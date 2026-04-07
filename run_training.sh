#!/bin/bash

# ==============================================================================
# Agentic OSPD GRPO Training Script
# Environment: 7 x A800 GPUs
# ==============================================================================

# Stop script on error
set -e

# Set working directory to project root
cd /data/yanfeizhang/OPSD_experiment

source /data/anaconda3/etc/profile.d/conda.sh
conda activate OPSD_experiment

# Limit visible GPUs to cards 0-6 (7 GPUs total)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6

# Set Ray temporary directory to avoid read-only errors from other users' paths (e.g., wuhanwei)
export RAY_TMPDIR="/data/yanfeizhang/OPSD_experiment/ray_tmp"
mkdir -p "$RAY_TMPDIR"
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0

if [ -n "$ip_head" ]; then
    export RAY_ADDRESS="$ip_head"
else
    LOCAL_RAY_IP=$(python3 - <<'PY'
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    s.connect(("8.8.8.8", 80))
    print(s.getsockname()[0])
except Exception:
    print(socket.gethostbyname(socket.gethostname()))
finally:
    s.close()
PY
)
    ray stop --force >/dev/null 2>&1 || true
    ray start --head \
        --node-ip-address "$LOCAL_RAY_IP" \
        --port=6379 \
        --include-dashboard=false \
        --disable-usage-stats \
        --temp-dir "$RAY_TMPDIR" \
        --num-gpus=7
    export RAY_ADDRESS="${LOCAL_RAY_IP}:6379"
fi

# Set Hugging Face cache directory to a local writable path
export HF_HOME="/data/yanfeizhang/OPSD_experiment/.cache/huggingface"
export HF_DATASETS_CACHE="/data/yanfeizhang/OPSD_experiment/.cache/huggingface/datasets"
export TMPDIR="/data/yanfeizhang/OPSD_experiment/.tmp"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TMPDIR"

# Set Python path to ensure imports work correctly
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Note: We use python-dotenv in the Python script directly, 
# so we don't need to source the .env file in bash which might fail due to syntax.

echo "============================================================"
echo "Starting Agentic OSPD GRPO Training"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Working Directory: $(pwd)"
echo "============================================================"

# Ensure output directory exists to avoid Hydra errors
mkdir -p workspace/Browsecomp_zh/outputs

# Run verl PPO trainer
# Note: To completely disable Hydra's default output directory in the project root,
# we use hydra.run.dir to redirect it to our workspace outputs.
python3 -m verl.trainer.main_ppo \
    --config-path $(pwd)/optimize \
    --config-name opsd_grpo_optim_fix \
    actor_rollout_ref.model.path=/data/huggingface_models/Qwen3-8B \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.load_format=dummy \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.max_model_len=32768 \
    actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
    actor_rollout_ref.rollout.max_num_seqs=32 \
    trainer.total_epochs=1 \
    data.train_files=['workspace/Browsecomp_zh/data/ASearcher_en_seed_data.jsonl'] \
    data.val_files=['workspace/Browsecomp_zh/data/val_data.jsonl'] \
    hydra.run.dir=/data/yanfeizhang/OPSD_experiment/workspace/Browsecomp_zh/outputs/hydra_logs/\${now:%Y-%m-%d}/\${now:%H-%M-%S}

echo "Training completed!"
