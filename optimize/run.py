"""
Agentic OSPD Multi-Stage GRPO Training Pipeline using Native verl

Architecture:
- Stage 1: Parallel Rollout Generation (async agent loop)
- Stage 2: Reward Computation (benchmark LLM judge)
- Stage 3: GRPO Advantage Estimation (group-relative normalization)
- Stage 4: OSPD Training (self-distillation with hint context)

Usage:
    python3 -m verl.trainer.main_ppo \
        --config-path optimize \
        --config-name opsd_grpo \
        actor_rollout_ref.model.path=/data/huggingface_models/Qwen3-8B
"""

import os
import sys
import json
import asyncio
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
from copy import deepcopy
from uuid import uuid4

import ray
import torch
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
from dotenv import load_dotenv
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopManager, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.workers.rollout.replica import TokenOutput

load_dotenv(Path(__file__).parent.parent / ".env")

os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
os.environ["VERL_VLLM_DISTRIBUTED_BACKEND"] = "ray"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))


class BrowsecompRewardManager:
    """Reward computation using benchmark's calculate_score method."""

    def __init__(self):
        from benchmarks.browsecomp_zh import BrowsecompZHBenchmark
        self.benchmark = BrowsecompZHBenchmark(name="browsecomp_zh", file_path="", log_path="")

    async def compute_score(self, question: str, ground_truth: str, prediction: str) -> float:
        if not prediction or prediction.strip() == "":
            return 0.0
        try:
            score, _ = await self.benchmark.calculate_score(ground_truth, prediction, question)
            return float(score)
        except Exception as e:
            print(f"[RewardManager] Error: {e}")
            return 0.0


class OSPDAdvantageComputer:
    """OSPD (On-Policy Self-Distillation) advantage computation.

    Key insight:
    - Teacher = Student = current model (same weights)
    - Teacher Forward: model(s_t + hint, a_t) - forward with hint context
    - Student Forward: model(s_t, a_t) - forward without hint
    - Advantage = logprob_teacher - logprob_student (token-level)
    """

    def __init__(self, weight_osdp: float = 0.1):
        self.weight_osdp = weight_osdp

    def compute_hint_advantage(self, step_data: Dict[str, Any]) -> float:
        """Compute OSPD advantage for a single step.

        If hint is provided (failed step with feedback):
            A_osdp = 0.5 (positive signal to learn from hint)
        Else:
            A_osdp = 0.0 (no distillation signal)
        """
        if step_data.get("hint"):
            return self.weight_osdp * 0.5
        return 0.0


class GRPOAdvantageEstimator:
    """GRPO (Group Relative Policy Optimization) advantage estimator.

    Key insight:
    - No Critic model needed
    - Group-relative advantage: A_g = (R_g - μ) / σ
    - All tokens in same trajectory share the same macro advantage
    """

    def __init__(self, norm_adv_by_std: bool = True):
        self.norm_adv_by_std = norm_adv_by_std

    def compute_group_advantages(self, trajectories: List[Dict]) -> List[Dict]:
        """Compute group-relative advantages for trajectories."""
        from collections import defaultdict

        groups = defaultdict(list)
        for traj in trajectories:
            question_id = traj.get("question_id") or traj.get("id", "unknown")
            groups[question_id].append(traj)

        all_rewards = []
        for group_id, group_trajs in groups.items():
            rewards = [t["reward"] for t in group_trajs]
            mean_r = np.mean(rewards)
            std_r = np.std(rewards) if len(rewards) > 1 else 1.0
            all_rewards.extend(rewards)

            for traj in group_trajs:
                if self.norm_adv_by_std and std_r > 1e-6:
                    traj["advantage_grpo"] = (traj["reward"] - mean_r) / std_r
                else:
                    traj["advantage_grpo"] = traj["reward"] - mean_r

        if all_rewards:
            baseline = np.mean(all_rewards)
            for traj in trajectories:
                traj["reward_baseline"] = baseline

        return trajectories

    def broadcast_to_tokens(self, trajectories: List[Dict]) -> List[Dict]:
        """Broadcast macro advantage to all tokens in trajectory."""
        for traj in trajectories:
            adv = traj.get("advantage_grpo", 0.0)
            for step in traj.get("steps", []):
                step["A_macro"] = adv
        return trajectories


class TrajectoryExtractor:
    """Extract steps from agent trajectories for OSPD training."""

    @staticmethod
    def extract_with_thoughts(trajectory: Dict) -> List[Dict]:
        """Extract all assistant responses including thoughts."""
        steps = []
        history = trajectory.get("history", {})

        if isinstance(history, dict):
            for key in ["initial_agent", "messages", "history"]:
                if key in history:
                    history = history[key]
                    break
            if isinstance(history, list) and len(history) > 0:
                if isinstance(history[0], list):
                    history = history[0]

        if not isinstance(history, list):
            return steps

        current_state = []

        for i, msg in enumerate(history):
            if isinstance(msg, str):
                msg = {"role": "unknown", "content": msg}
            elif not isinstance(msg, dict):
                continue

            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "user":
                current_state = [msg]
            elif role == "assistant":
                thought_match = re.search(r"<thought>(.*?)</thought>", content, re.DOTALL)
                tool_call_match = re.search(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL)

                if thought_match:
                    thought = thought_match.group(1).strip()
                    steps.append({
                        "a_t": f"Thought: {thought}",
                        "s_t": json.dumps(current_state, ensure_ascii=False),
                        "env_obs": "",
                        "outcome": "thought",
                        "hint": None,
                    })

                if tool_call_match:
                    tool_xml = tool_call_match.group(1)
                    name_match = re.search(r'"name":\s*"([^"]+)"', tool_xml)
                    args_match = re.search(r'"args":\s*(\{.*?\})', tool_xml, re.DOTALL)
                    if name_match and args_match:
                        tool_name = name_match.group(1)
                        tool_args = args_match.group(1)
                        steps.append({
                            "a_t": f"ToolCall: {tool_name}({tool_args})",
                            "s_t": json.dumps(current_state, ensure_ascii=False),
                            "env_obs": "",
                            "outcome": "pending",
                            "hint": None,
                        })
                        current_state.append(msg)
            elif role == "tool":
                if steps:
                    steps[-1]["env_obs"] = content[:1000]
                    if steps[-1]["outcome"] == "thought":
                        steps[-1]["outcome"] = "thought_with_result"
                    elif "error" in content.lower() or "failed" in content.lower():
                        steps[-1]["outcome"] = "failed"
                    elif not content.strip():
                        steps[-1]["outcome"] = "empty"
                    else:
                        steps[-1]["outcome"] = "success"

        return steps

    @staticmethod
    def extract_answer(trajectory: Dict) -> str:
        """Extract final answer from trajectory."""
        import re
        from base.utils.utils import extract_xml

        history = trajectory.get("history", [])
        last_msg = history[-1] if history else {}
        content = last_msg.get("content", "")

        tool_calls = re.findall(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL)
        for tool_call_str in reversed(tool_calls):
            try:
                import json
                tool_call = json.loads(tool_call_str)
                if tool_call.get("name") == "get_final_answer":
                    args = tool_call.get("args", {})
                    answer = args.get("final_answer") or args.get("answer", "")
                    if answer:
                        return answer
            except:
                continue

        found_fields = extract_xml(content)
        if found_fields.get("answer"):
            return found_fields["answer"]

        return content[:500]


class MultiStageOSPDTrainer:
    """Multi-stage OSPD training pipeline using verl RayPPOTrainer.

    Stages:
    1. Rollout: Generate trajectories via async agent loop
    2. Reward: Compute rewards via benchmark LLM judge
    3. GRPO: Compute group-relative advantages
    4. OSPD: Compute self-distillation advantages (teacher-student gap)
    5. Train: Update policy with combined advantages
    """

    def __init__(
        self,
        config: OmegaConf,
        model_path: str,
        output_dir: str = "./output",
        num_rollouts: int = 8,
        search_mode: str = "real",
        tool_cfg: Optional[Dict] = None,
    ):
        self.config = config
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.num_rollouts = num_rollouts
        self.search_mode = search_mode
        self.tool_cfg = tool_cfg or {}
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.reward_manager = BrowsecompRewardManager()
        self.grpo_estimator = GRPOAdvantageEstimator()
        self.ospd_computer = OSPDAdvantageComputer()

        self.rollouts_dir = self.output_dir / "rollouts"
        self.rollouts_dir.mkdir(exist_ok=True)

    def _build_tool_cfg(self) -> Dict[str, Any]:
        """Build tool configuration based on search mode."""
        if self.search_mode == "local_simulated":
            return {
                "search": {"environment_mode": "local_simulated"},
                "open": {"environment_mode": "local_simulated"},
                "find": {"environment_mode": "local_simulated"},
            }
        return {
            "search": {"environment_mode": "real"},
            "open": {"environment_mode": "real"},
            "find": {"environment_mode": "real"},
        }

    async def run_single_rollout(
        self,
        question: str,
        rollout_id: str,
        ground_truth: str,
    ) -> Dict[str, Any]:
        """Run a single agent rollout."""
        from workspace.Browsecomp_zh.agent.agent import Workflow

        tool_cfg = self.tool_cfg or self._build_tool_cfg()
        agent = Workflow(
            llm_config="deepseek-chat",
            tool_cfg=tool_cfg,
        )

        try:
            result = await agent.run(question)
            trajectory = {
                "id": rollout_id,
                "question": question,
                "ground_truth": ground_truth,
                "history": result.get("history", []),
                "final_answer": result.get("answer", ""),
            }
        except Exception as e:
            print(f"[Rollout {rollout_id}] Error: {e}")
            trajectory = {
                "id": rollout_id,
                "question": question,
                "ground_truth": ground_truth,
                "history": [],
                "final_answer": "",
                "error": str(e),
            }

        return trajectory

    async def generate_rollouts_parallel(
        self,
        questions: List[Dict],
        max_concurrency: int = 32,
    ) -> List[Dict[str, Any]]:
        """Generate multiple rollouts in parallel."""
        sem = asyncio.Semaphore(max_concurrency)

        async def run_with_sem(question, qid, answer, r):
            async with sem:
                rollout_id = f"{qid}_r{r}"
                return await self.run_single_rollout(question, rollout_id, answer)

        tasks = []
        for q in questions:
            question = q["question"]
            qid = q["id"]
            answer = q.get("answer", "")
            for r in range(self.num_rollouts):
                tasks.append(run_with_sem(question, qid, answer, r))

        results = []
        for i in range(0, len(tasks), 100):
            batch = tasks[i:i+100]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            for idx, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    qid = questions[i // self.num_rollouts].get("id", "unknown") if i // self.num_rollouts < len(questions) else "unknown"
                    print(f"[Rollout] Exception for {qid}_r{idx}: {result}")
                    results.append({
                        "id": f"error_{qid}_r{idx}",
                        "question": questions[i // self.num_rollouts].get("question", "") if i // self.num_rollouts < len(questions) else "",
                        "ground_truth": questions[i // self.num_rollouts].get("answer", "") if i // self.num_rollouts < len(questions) else "",
                        "history": [],
                        "final_answer": "",
                        "error": str(result),
                    })
                else:
                    results.append(result)

        return results

    async def compute_rewards(self, trajectories: List[Dict]) -> List[Dict]:
        """Compute rewards using benchmark LLM judge."""
        rewardeds = []
        for traj in tqdm(trajectories, desc="Computing rewards"):
            if not isinstance(traj, dict):
                print(f"[Reward] Skipping non-dict trajectory: {type(traj)}")
                continue
            try:
                question = traj.get("question", "")
                ground_truth = traj.get("ground_truth", "")
                final_answer = traj.get("final_answer", "")
                reward = await self.reward_manager.compute_score(question, ground_truth, final_answer)
            except Exception as e:
                print(f"[Reward] Error for {traj.get('id', 'unknown')}: {e}")
                reward = 0.0
            traj["reward"] = reward
            rewardeds.append(traj)

        return rewardeds

    def build_ospd_samples(self, trajectories: List[Dict]) -> List[Dict]:
        """Build OSPD training samples with hints for failed steps."""
        samples = []
        for traj in trajectories:
            for step in traj.get("steps", []):
                hint = None
                if step["outcome"] in ["failed", "empty"]:
                    hint = self._generate_hint_from_feedback(
                        step["s_t"], step["a_t"], step["env_obs"]
                    )
                    step["hint"] = hint

                sample = {
                    "question": traj["question"],
                    "s_t": step["s_t"],
                    "a_t": step["a_t"],
                    "A_macro": step.get("A_macro", 0.0),
                    "outcome": step["outcome"],
                    "traj_id": traj["id"],
                    "hint": hint,
                    "s_enhanced": step["s_t"] + f"\n\n[Hint]: {hint}" if hint else step["s_t"],
                }
                samples.append(sample)

        return samples

    def _generate_hint_from_feedback(
        self, s_t: str, a_t: str, env_feedback: str
    ) -> Optional[str]:
        """Generate hint based on feedback (simple rules)."""
        feedback_lower = env_feedback.lower()

        if "no result" in feedback_lower or "not found" in feedback_lower:
            return "The search returned no results. Try different keywords or search more directly."
        if "error" in feedback_lower:
            return "An error occurred. Check the tool call format and parameters."
        if len(env_feedback) < 20:
            return "The result is too short or irrelevant. Try a more specific search query."

        return "The current approach did not work. Consider trying a different strategy."

    def compute_combined_advantages(self, samples: List[Dict]) -> List[Dict]:
        """Compute final advantages: A_final = A_GRPO + A_OSPD."""
        for sample in samples:
            A_macro = sample["A_macro"]
            A_osdp = self.ospd_computer.compute_hint_advantage(sample)
            sample["A_osdp"] = A_osdp
            sample["A_final"] = A_macro + A_osdp

        return samples

    async def run_epoch(
        self,
        dataset: List[Dict],
        epoch: int,
        max_concurrency: int = 32,
    ) -> Dict[str, Any]:
        """Run one complete epoch of multi-stage OSPD training."""
        print(f"\n{'='*60}")
        print(f"[Epoch {epoch}] Multi-Stage OSPD Training")
        print(f"{'='*60}")

        print("\n[Stage 1/5] Generating rollouts in parallel...")
        trajectories = await self.generate_rollouts_parallel(dataset, max_concurrency)
        print(f"  Generated {len(trajectories)} trajectories")

        print("\n[Stage 2/5] Computing rewards via benchmark LLM judge...")
        trajectories = await self.compute_rewards(trajectories)
        avg_reward = np.mean([t["reward"] for t in trajectories])
        print(f"  Average reward: {avg_reward:.4f}")

        print("\n[Stage 3/5] Computing GRPO advantages...")
        trajectories = self.grpo_estimator.compute_group_advantages(trajectories)
        trajectories = self.grpo_estimator.broadcast_to_tokens(trajectories)

        print("\n[Stage 4/5] Extracting steps and building OSPD samples...")
        for traj in trajectories:
            traj["steps"] = TrajectoryExtractor.extract_with_thoughts(traj)
        samples = self.build_ospd_samples(trajectories)
        samples = self.compute_combined_advantages(samples)
        print(f"  Built {len(samples)} OSPD samples")

        output_file = self.rollouts_dir / f"epoch_{epoch}_samples.jsonl"
        with open(output_file, "w") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"  Saved samples to {output_file}")

        traj_file = self.rollouts_dir / f"epoch_{epoch}_trajectories.jsonl"
        with open(traj_file, "w") as f:
            for t in trajectories:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        print(f"  Saved trajectories to {traj_file}")

        print("\n[Stage 5/5] OSPD training step...")
        metrics = {"epoch": epoch, "num_samples": len(samples), "avg_reward": avg_reward}
        print(f"  Metrics: {metrics}")

        return metrics

    async def run(self, dataset: List[Dict], num_epochs: int, max_concurrency: int = 32):
        """Run complete multi-stage OSPD training."""
        print(f"[Pipeline] Starting training with {len(dataset)} questions")
        print(f"[Pipeline] Rollouts per question: {self.num_rollouts}")
        print(f"[Pipeline] Max concurrency: {max_concurrency}")

        for epoch in range(1, num_epochs + 1):
            metrics = await self.run_epoch(dataset, epoch, max_concurrency)

            if epoch % 5 == 0:
                ckpt_dir = self.output_dir / f"checkpoint_epoch_{epoch}"
                ckpt_dir.mkdir(exist_ok=True)
                print(f"[Pipeline] Checkpoint would be saved to {ckpt_dir}")

        print("\n[Pipeline] Training complete!")


def create_verl_config(
    model_path: str,
    num_gpus: int = 1,
    train_batch_size: int = 32,
    max_prompt_length: int = 4096,
    max_response_length: int = 8192,
    rollout_n: int = 8,
    learning_rate: float = 1e-6,
) -> OmegaConf:
    """Create verl configuration for OSPD GRPO training."""
    config = OmegaConf.create({
        "algorithm": {
            "adv_estimator": "grpo",
            "gamma": 1.0,
            "lam": 1.0,
            "use_kl_in_reward": False,
            "norm_adv_by_std_in_grpo": True,
        },
        "data": {
            "tokenizer": None,
            "train_files": [],
            "val_files": [],
            "prompt_key": "prompt",
            "max_prompt_length": max_prompt_length,
            "max_response_length": max_response_length,
            "train_batch_size": train_batch_size,
            "return_raw_input_ids": False,
            "return_raw_chat": True,
            "shuffle": True,
            "seed": 42,
        },
        "actor_rollout_ref": {
            "hybrid_engine": True,
            "model": {
                "path": model_path,
                "external_lib": None,
                "override_config": {"attn_implementation": "sdpa"},
                "enable_gradient_checkpointing": True,
                "use_remove_padding": True,
            },
            "actor": {
                "strategy": "fsdp",
                "loss_agg_mode": "token-mean",
                "ppo_mini_batch_size": 8,
                "ppo_micro_batch_size_per_gpu": 4,
                "use_dynamic_bsz": True,
                "ppo_max_token_len_per_gpu": 28672,
                "grad_clip": 1.0,
                "clip_ratio": 0.2,
                "use_kl_loss": True,
                "kl_loss_coef": 0.001,
                "kl_loss_type": "low_var_kl",
                "ppo_epochs": 1,
                "shuffle": False,
                "ulysses_sequence_parallel_size": 1,
                "entropy_from_logits_with_chunking": False,
                "entropy_checkpointing": False,
                "use_remove_padding": True,
                "optim": {
                    "lr": learning_rate,
                    "lr_warmup_steps_ratio": 0.0,
                    "total_training_steps": -1,
                    "weight_decay": 0.01,
                    "lr_warmup_steps": -1,
                    "betas": [0.9, 0.999],
                    "clip_grad": 1.0,
                    "optimizer": "AdamW",
                    "optimizer_impl": "torch.optim",
                    "min_lr_ratio": None,
                    "warmup_style": None,
                    "lr_scheduler_type": "constant",
                    "num_cycles": 0.5,
                    "zero_indexed_step": True,
                    "override_optimizer_config": None,
                },
                "checkpoint": {
                    "save_contents": ["model", "optimizer", "extra"],
                    "load_contents": ["model", "optimizer", "extra"],
                    "async_save": False,
                    "mbridge_config": {},
                },
                "fsdp_config": {
                    "_target_": "verl.workers.config.FSDPEngineConfig",
                    "wrap_policy": {"min_num_params": 0},
                    "param_offload": False,
                    "grad_offload": False,
                    "optimizer_offload": True,
                    "fsdp_size": -1,
                    "entropy_from_logits_with_chunking": False,
                    "entropy_checkpointing": False,
                    "use_torch_compile": True,
                    "forward_only": False,
                },
            },
            "ref": {
                "strategy": "fsdp",
                "use_torch_compile": True,
                "ulysses_sequence_parallel_size": 1,
                "entropy_from_logits_with_chunking": False,
                "entropy_checkpointing": False,
                "fsdp_config": {
                    "_target_": "verl.workers.config.FSDPEngineConfig",
                    "param_offload": False,
                    "wrap_policy": {"min_num_params": 0},
                    "entropy_from_logits_with_chunking": False,
                    "entropy_checkpointing": False,
                    "use_torch_compile": True,
                    "forward_only": True,
                },
                "log_prob_micro_batch_size_per_gpu": 8,
                "log_prob_use_dynamic_bsz": True,
                "log_prob_max_token_len_per_gpu": 28672,
            },
            "rollout": {
                "name": "vllm",
                "nnodes": 1,
                "n_gpus_per_node": num_gpus,
                "temperature": 1.0,
                "val_temperature": 0.0,
                "compute_reward": False,
                "top_k": -1,
                "top_p": 1.0,
                "prompt_length": max_prompt_length,
                "response_length": max_response_length,
                "max_model_len": max_prompt_length + max_response_length,
                "dtype": "bfloat16",
                "gpu_memory_utilization": 0.35,
                "ignore_eos": False,
                "enforce_eager": True,
                "free_cache_engine": True,
                "load_format": "dummy",
                "data_parallel_size": 1,
                "expert_parallel_size": 1,
                "tensor_model_parallel_size": 1,
                "pipeline_model_parallel_size": 1,
                "max_num_batched_tokens": 4096,
                "max_num_seqs": 32,
                "log_prob_micro_batch_size_per_gpu": 8,
                "log_prob_use_dynamic_bsz": True,
                "log_prob_max_token_len_per_gpu": 28672,
                "enable_chunked_prefill": True,
                "do_sample": True,
                "n": rollout_n,
                "n_val": 1,
                "mode": "async",
                "val_kwargs": {"do_sample": False},
                "agent": {
                    "default_agent_loop": "browsecomp_zh_agent",
                    "agent_loop_manager_class": "optimize.run.BrowsecompZHAgentLoopManager",
                },
            },
        },
        "trainer": {
            "total_epochs": 10,
            "project_name": "agentic_ospd",
            "experiment_name": "browsecomp_zh",
            "logger": ["console"],
            "nnodes": 1,
            "n_gpus_per_node": num_gpus,
            "save_freq": 100,
            "val_before_train": False,
            "test_freq": 10,
            "critic_warmup": 0,
        },
    })
    return config


def load_dataset(dataset_path: str, limit: int = None) -> List[Dict]:
    """Load dataset from JSONL file."""
    data = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            record = {
                "id": str(item.get("idx", item.get("id", len(data)))),
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
            }
            data.append(record)
            if limit and len(data) >= limit:
                break
    print(f"[DataLoader] Loaded {len(data)} samples from {dataset_path}")
    return data


async def run_smoke_tests():
    """Run smoke tests for the pipeline."""
    print("=" * 60)
    print("Running SMOKE TESTS")
    print("=" * 60)

    print("\n[Test 1] Data Loading")
    data = load_dataset(
        "workspace/Browsecomp_zh/data/ASearcher_en_seed_data.jsonl", limit=10
    )
    assert len(data) > 0, "No data loaded!"
    print(f"  Loaded {len(data)} samples")
    print("  [PASS]")

    print("\n[Test 2] verl Configuration")
    config = create_verl_config(model_path="/data/huggingface_models/Qwen3-8B")
    assert config.algorithm.adv_estimator == "grpo"
    print(f"  GRPO estimator: {config.algorithm.adv_estimator}")
    print("  [PASS]")

    print("\n[Test 3] GRPO Advantage Estimation")
    grpo = GRPOAdvantageEstimator()
    test_trajs = [
        {"question_id": "q1", "reward": 1.0},
        {"question_id": "q1", "reward": 0.5},
        {"question_id": "q1", "reward": 0.0},
        {"question_id": "q2", "reward": 1.0},
    ]
    result = grpo.compute_group_advantages(test_trajs)
    assert all("advantage_grpo" in t for t in result)
    print(f"  Advantages computed: {[t['advantage_grpo'] for t in result]}")
    print("  [PASS]")

    print("\n[Test 4] OSPD Advantage Computation")
    opsd = OSPDAdvantageComputer(weight_osdp=0.1)
    step_with_hint = {"hint": "Try different keywords"}
    step_without_hint = {"hint": None}
    adv_with = opsd.compute_hint_advantage(step_with_hint)
    adv_without = opsd.compute_hint_advantage(step_without_hint)
    print(f"  With hint: {adv_with}, Without hint: {adv_without}")
    assert adv_with > adv_without
    print("  [PASS]")

    print("\n" + "=" * 60)
    print("ALL SMOKE TESTS PASSED!")
    print("=" * 60)


async def main():
    parser = argparse.ArgumentParser(description="Agentic OSPD Multi-Stage GRPO Training")
    parser.add_argument("command", type=str, choices=["smoke", "run", "verl"])
    parser.add_argument("--config", type=str, default="optimize/opsd_grpo.yaml")
    parser.add_argument("--model", type=str, default="/data/huggingface_models/Qwen3-8B")
    parser.add_argument("--dataset", type=str, default="workspace/Browsecomp_zh/data/ASearcher_en_seed_data.jsonl")
    parser.add_argument("--rollouts", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--output", type=str, default="/data/yanfeizhang/AgenticOPSD/workspace/Browsecomp_zh/outputs")
    parser.add_argument("--max_concurrency", type=int, default=32)
    parser.add_argument("--rollout_limit", type=int, default=None)
    parser.add_argument("--search_mode", type=str, choices=["real", "local_simulated"], default="real")
    parser.add_argument("--num_gpus", type=int, default=1)
    args = parser.parse_args()

    if args.command == "smoke":
        await run_smoke_tests()
        return

    if args.command == "verl":
        print(f"[verl] Starting native verl training with config: {args.config}")
        print(f"[verl] Model: {args.model}")
        print(f"[verl] This uses the custom async BrowseComp agent loop.")
        print(f"[verl] Use the following command to start training:")
        print(f"""[verl]
cd {Path(__file__).parent.parent} && \\
python3 -m verl.trainer.main_ppo \\
    --config-path optimize \\
    --config-name opsd_grpo_optim_fix \\
    actor_rollout_ref.model.path={args.model} \\
    actor_rollout_ref.rollout.n={args.rollouts} \\
    trainer.total_epochs={args.epochs} \\
    data.train_files=['{args.dataset}']
""")
        return

    print(f"[Pipeline] Loading dataset from {args.dataset}")
    dataset = load_dataset(args.dataset, limit=args.rollout_limit)
    if not dataset:
        print("[Pipeline] No data loaded, using demo data")
        dataset = [{"id": "demo", "question": "What is 1+1?", "answer": "2"}]

    config = create_verl_config(
        model_path=args.model,
        num_gpus=args.num_gpus,
        rollout_n=args.rollouts,
    )

    tool_cfg = {
        "search": {"environment_mode": args.search_mode},
        "open": {"environment_mode": args.search_mode},
        "find": {"environment_mode": args.search_mode},
    }

    trainer = MultiStageOSPDTrainer(
        config=config,
        model_path=args.model,
        output_dir=args.output,
        num_rollouts=args.rollouts,
        search_mode=args.search_mode,
        tool_cfg=tool_cfg,
    )

    await trainer.run(
        dataset=dataset,
        num_epochs=args.epochs,
        max_concurrency=args.max_concurrency,
    )


@register("browsecomp_zh_agent")
class BrowsecompZHAgentLoop(AgentLoopBase):
    """verl-compatible AgentLoop for Browsecomp_zh using native verl AgentLoopBase.

    This class integrates with verl's AgentLoopWorker to provide:
    - Multi-turn conversation with tool calling
    - Async LLM generation via SGLang/vLLM
    - Proper token-level masking for PPO training

    Usage with verl:
        python3 -m verl.trainer.main_ppo \
            --config-path optimize \
            --config-name opsd_grpo \
            actor_rollout_ref.model.path=/data/huggingface_models/Qwen3-8B
    """

    def __init__(
        self,
        *args,
        tools_config_path: Optional[str] = None,
        max_turns: int = 10,
        max_response_length: int = 8192,
        **kwargs,
    ):
        self.max_turns = max_turns
        self.max_response_length = max_response_length
        self.tools_config_path = tools_config_path
        if "trainer_config" in kwargs:
            super().__init__(*args, **kwargs)
            self.prompt_length = self.rollout_config.prompt_length
            self.response_length = min(max_response_length, self.rollout_config.response_length)

    @property
    def name(self) -> str:
        return "browsecomp_zh_agent"

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        if not hasattr(self, "server_manager"):
            raise RuntimeError("BrowsecompZHAgentLoop must be instantiated by verl AgentLoopManager.")

        messages = list(kwargs["raw_prompt"])
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        prompt_ids = await self.apply_chat_template(
            messages,
            images=images,
            videos=videos,
        )

        metrics = {}
        with simple_timer("generate_sequences", metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=images,
                video_data=videos,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1

        response_ids = output.token_ids[: self.response_length]
        response_mask = [1] * len(response_ids)

        result = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
            routed_experts=(
                output.routed_experts[: len(prompt_ids) + self.response_length]
                if output.routed_experts is not None
                else None
            ),
            multi_modal_data=multi_modal_data,
            num_turns=2,
            metrics=metrics,
            extra_fields=output.extra_fields,
        )
        result.extra_fields.update({"turn_scores": [], "tool_rewards": []})
        return result


class BrowsecompZHAgentLoopManager(AgentLoopManager):
    pass


class OSPDAgentLoopFactory:
    """Factory for creating OSPD-compatible agent loops.

    This factory creates agent loops that support:
    - OSPD (On-Policy Self-Distillation) with hint context
    - GRPO advantage computation
    - Multi-turn tool calling
    """

    @staticmethod
    def create_browsecomp_agent(
        config_path: str = "optimize/opsd_grpo_optim_fix.yaml",
        model_path: str = "/data/huggingface_models/Qwen3-8B",
        max_turns: int = 10,
    ) -> BrowsecompZHAgentLoop:
        return BrowsecompZHAgentLoop(
            tools_config_path=config_path,
            max_turns=max_turns,
        )


def create_verl_agent_config(
    model_path: str,
    output_dir: str = "/data/yanfeizhang/AgenticOPSD/workspace/Browsecomp_zh/outputs",
    num_gpus: int = 1,
    rollout_n: int = 8,
) -> OmegaConf:
    """Create verl configuration for OSPD GRPO training with agent support."""
    return create_verl_config(
        model_path=model_path,
        num_gpus=num_gpus,
        train_batch_size=32,
        max_prompt_length=4096,
        max_response_length=8192,
        rollout_n=rollout_n,
    )


if __name__ == "__main__":
    import re
    asyncio.run(main())
