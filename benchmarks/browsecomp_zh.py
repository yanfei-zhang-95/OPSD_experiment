import re
import string
from collections import Counter
from typing import Callable, List, Tuple

from pydantic import BaseModel, Field

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from benchmarks.benchmark import BaseBenchmark
from base.engine.logs import logger

from base.engine.async_llm import create_llm_instance
from base.engine.formatter import XmlFormatter, FormatError

class LLMJudge(BaseModel):
    right_or_wrong: int = Field(default=0, description="right_or_wrong")
    reason: str = Field(default="", description="reason")

class BrowsecompZHBenchmark(BaseBenchmark):
    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)
        self.llm_judge_formatter = XmlFormatter.from_model(LLMJudge)

    async def calculate_score(self, ground_truth: str, prediction: str, question: str) -> Tuple[float, str]:
        # Create a fresh LLM instance for each call to avoid cross-thread/cross-loop issues
        llm_judge = create_llm_instance("deepseek-chat")
        prompt = f"""
You are an LLM agent to find out whether the question and the answer are correct.
There is always an answer to the question, so if the ground truth and the prediction cannot match, you cannot specify it as correct.
The prediction that cannot specify the exact answer is inaccurate and cannot be specified.
The question is: {question}
The ground truth is: {ground_truth}
The prediction is: {prediction}

Sometimes the ground truth is in Chinese, while the preiction is in English or Pinyin, please note when the predicted Pinyin or English meaning matches the ground truth, the answer should also be correct.
Sometimes the ground truth/predicion might contain some special characters that are  (e.g. Jesús<->Jesus), please note this and give the correct answer.
For person names, especially Spanish/Latin names, acceptable variations include:
1. Missing maternal surname (e.g., "Luis de Jesús Rodríguez" is correct for "Luis de Jesus Rodriguez Gutierrez").
2. Differences in accents/diacritics (e.g., "Jesús" vs "Jesus").
3. As long as the prediction uniquely identifies the same person as the ground truth, it should be marked as correct (1).

You should give your answer as:
<right_or_wrong>a integer stating whether the prediction is correct with respect to the question and the ground truth, 1 stands for correct and 0 stands for incorrect, besides, "no answer" states that no answer was given after iterative seraches, which should also get 0</right_or_wrong>
<reason>a string stating the reason for your answer</reason>
        """
        try:
            raw_response = await llm_judge.call_with_format(
                prompt,
                self.llm_judge_formatter
            )
            return int(raw_response["right_or_wrong"]), prediction
        finally:
            # Prevent noisy shutdown warnings from AsyncOpenAI/httpx when event loop closes.
            aclose = getattr(llm_judge, "aclose", None)
            if callable(aclose):
                try:
                    await aclose()
                except Exception:
                    pass

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, graph, input_text):
        result = await graph(request=input_text)
        # ReActAgent.run() returns a dict with 'answer' and 'history'
        output = result.get("answer", "")
        history = result.get("history", [])
        return output, history

    def _slice_history(self, history):
        """Helper to slice history (remove system prompt) handling both flat and nested structures."""
        if isinstance(history, dict):
            sliced_history = {}
            for k, v in history.items():
                if isinstance(v, list):
                    # Check if it's a list of lists (multi-dimensional history from agent)
                    if len(v) > 0 and isinstance(v[0], list):
                        sliced_history[k] = [sub[1:] if len(sub) > 0 else sub for sub in v]
                    else:
                        sliced_history[k] = v[1:]
                else:
                    sliced_history[k] = v
            return sliced_history
        elif isinstance(history, list):
            # Check if it's a list of lists
            if len(history) > 0 and isinstance(history[0], list):
                return [sub[1:] if len(sub) > 0 else sub for sub in history]
            else:
                return history[1:]
        return history

    async def evaluate_problem(self, problem: dict, graph: Callable) -> Tuple[str, str, str, str, float, float]:
        input_text = problem["question"]
        expected_output = problem["answer"]
        inputs = f"Question: {input_text}\n\nAnswer:"

        try:
            output, history = await self._generate_output(graph, inputs)
            score, extracted_output = await self.calculate_score(expected_output, output, input_text)

            if (
                score < 0.3
            ):  # We set the threshold for collecting incorrect questions to 0.3, as F1 Score cannot be simply judged using 0-1
                sliced_history = self._slice_history(history)
                self.log_mismatch(input_text, expected_output, output, extracted_output, history=sliced_history)

            return input_text, history, output, expected_output, score

        except Exception as e:
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {e}")
            error_message = f"Maximum retries reached. Skipping this sample. Error: {e}"
            
            sliced_history = []
            if 'history' in locals():
                sliced_history = self._slice_history(history)

            self.log_mismatch(input_text, expected_output, error_message, error_message, history=sliced_history)
            return input_text, "no history given an error occured", str(e), expected_output, 0

    def get_result_columns(self) -> List[str]:
        return ["question", "history", "prediction", "expected_output", "score"]
