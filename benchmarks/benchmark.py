import asyncio
import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Tuple

import aiofiles
import pandas as pd
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from tqdm.contrib.logging import logging_redirect_tqdm

from scripts.logs import logger
from scripts.utils.common import write_json_file


class BaseBenchmark(ABC):
    PASS = "PASS"
    FAIL = "FAIL"

    def __init__(self, name: str, file_path: str, log_path: str, batch_size: int = None):
        self.name = name
        self.file_path = file_path
        self.log_path = os.environ.get('EXPERIMENT_DIR', log_path)
        self.session_time = datetime.now().strftime("%m_%d_%H_%M")
        self.batch_size = batch_size
        self.batch_results = []
        os.makedirs(self.log_path, exist_ok=True)

    async def load_data(self, specific_indices: List[int] = None) -> List[dict]:
        data = []
        async with aiofiles.open(self.file_path, mode="r", encoding="utf-8") as file:
            async for line in file:
                data.append(json.loads(line))
        if specific_indices is not None:
            return [data[i] for i in specific_indices if i < len(data)]
        return data

    def save_batch_to_jsonl(self, results: List[Tuple[Any, ...]], columns: List[str], batch_num: int):
        if not results:
            return

        df = pd.DataFrame(results, columns=columns)
        avg_score = df["score"].mean() if "score" in df.columns else 0.0
        input_tokens = df["input_tokens"].sum() if "input_tokens" in df.columns else 0
        output_tokens = df["output_tokens"].sum() if "output_tokens" in df.columns else 0
        total_tokens = df["total_tokens"].sum() if "total_tokens" in df.columns else 0
        total_calls = df["calls"].sum() if "calls" in df.columns else 0
        avg_tokens = df["total_tokens"].mean() if "total_tokens" in df.columns else 0.0
        avg_calls = df["calls"].mean() if "calls" in df.columns else 0.0

        timestamp = datetime.now().strftime("%m_%d_%H_%M")
        base_filename = f"batch_{batch_num:02d}_score_{avg_score:.3f}_{timestamp}"
        jsonl_filename = f"{base_filename}.jsonl"
        summary_filename = "batch_summary.txt"

        jsonl_output_file = os.path.join(self.log_path, jsonl_filename)
        summary_output_file = os.path.join(self.log_path, summary_filename)

        with open(jsonl_output_file, "w", encoding="utf-8") as f:
            for record in df.to_dict('records'):
                if 'question' in record:
                    record['problem'] = record.pop('question')
                if 'expected_output' in record:
                    record['solution'] = record.pop('expected_output')
                f.write(json.dumps(record) + '\n')

        partial_summary_info = (
            f"\nBatch {batch_num}\n"
            f"Avg score: {avg_score:.3f}\n"
            f"Tokens: {input_tokens} + {output_tokens} = {total_tokens}\n"
            f"Total calls: {total_calls}\n"
            f"Avg tokens: {avg_tokens:.3f}\n"
            f"Avg calls: {avg_calls:.3f}\n"
        )

        with open(summary_output_file, "a+", encoding="utf-8") as f:
            f.write(partial_summary_info + '\n')

        logger.info(f"Batch {batch_num} saved: {jsonl_filename} (avg_score: {avg_score:.3f})")
        logger.info(f"Summary for batch {batch_num} saved to {summary_filename}")

        return jsonl_output_file

    def cleanup_batch_files(self):
        if not self.batch_size:
            return
        import glob
        batch_files = glob.glob(os.path.join(self.log_path, "batch_*.jsonl"))
        for batch_file in batch_files:
            try:
                os.remove(batch_file)
                logger.info(f"Deleted batch file: {os.path.basename(batch_file)}")
            except Exception as e:
                logger.warning(f"Failed to delete {batch_file}: {e}")

    def print_and_save_results_to_jsonl(self, results: List[Tuple[Any, ...]], columns: List[str]):
        df = pd.DataFrame(results, columns=columns)
        avg_score = df["score"].mean()
        input_tokens = df["input_tokens"].sum()
        output_tokens = df["output_tokens"].sum()
        total_tokens = df["total_tokens"].sum()
        avg_tokens = df["total_tokens"].mean()
        avg_calls = df["calls"].mean()
        total_calls = df["calls"].sum()

        if 'question' in df.columns:
            df.rename(columns={'question': 'problem'}, inplace=True)

        jsonl_output_file = os.path.join(self.log_path, f"Detailed_{avg_score:.3f}.jsonl")
        with open(jsonl_output_file, "w", encoding="utf-8") as f:
            for record in df.to_dict('records'):
                f.write(json.dumps(record) + '\n')

        summary_output_file = os.path.join(self.log_path, f"Summary_{avg_score:.3f}.txt")
        summary_info = (
            f"\nAverage score on {self.name} dataset: {avg_score:.5f}\n"
            f"Tokens: {input_tokens} + {output_tokens} = {total_tokens}\n"
            f"Total calls: {total_calls}\n"
            f"Avg tokens: {avg_tokens:.5f}\n"
            f"Avg calls: {avg_calls:.5f}\n"
        )

        with open(summary_output_file, "w", encoding="utf-8") as f:
            f.write(summary_info)

        logger.info(summary_info)
        logger.info(f"Results saved to {jsonl_output_file}")
        logger.info(f"Summary saved to {summary_output_file}")

        return avg_score, avg_tokens, input_tokens, output_tokens, total_tokens

    def log_mismatch(self, problem: str, expected_output: Any, prediction: str, extracted_output: Any, extract_answer_code: str = "None", **data):
        log_data = {
            **data,
            "problem": problem,
            "expected_output": expected_output,
            "model_output": prediction,
            "extracted_output": extracted_output,
            "extract_answer_code": extract_answer_code,
        }

        session_time = os.environ.get('EXPERIMENT_TIMESTAMP', self.session_time)
        log_file = Path(self.log_path) / "Failed_samples.jsonl"

        if log_file.exists():
            try:
                with log_file.open("r", encoding="utf-8") as f:
                    existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = []
        else:
            existing_data = []

        existing_data.append(log_data)
        write_json_file(log_file, existing_data, encoding="utf-8", indent=4)
        logger.info(f"Saved {len(existing_data)} failed samples to {log_file}")

    @abstractmethod
    async def evaluate_problem(self, problem: dict, agent: Callable) -> Tuple[Any, ...]:
        pass

    @abstractmethod
    def calculate_score(self, expected_output: Any, prediction: Any) -> Tuple[float, Any]:
        pass

    @abstractmethod
    def get_result_columns(self) -> List[str]:
        pass

    async def evaluate_all_problems(self, data: List[dict], agent: Callable, max_concurrent_tasks: int = 20):
        semaphore = asyncio.Semaphore(max_concurrent_tasks)
        columns = self.get_result_columns()
        all_results = []
        batch_num = 0

        async def sem_evaluate(problem):
            async with semaphore:
                return await self.evaluate_problem(problem, agent)

        tasks = [sem_evaluate(problem) for problem in data]

        if self.batch_size:
            for i in tqdm(range(0, len(tasks), self.batch_size), desc="Batch Progress"):
                batch_tasks = tasks[i:i + self.batch_size]
                batch_results = await tqdm_asyncio.gather(*batch_tasks, desc=f"Evaluating {self.name} problems", total=len(batch_tasks))
                all_results.extend(batch_results)
                self.save_batch_to_jsonl(batch_results, columns, batch_num)
                batch_num += 1
        else:
            all_results = await tqdm_asyncio.gather(*tasks, desc=f"Evaluating {self.name} problems", total=len(tasks))

        return all_results

    async def run_evaluation(self, agent: Callable, va_list: List[int], max_concurrent_tasks: int = 20):
        data = await self.load_data(va_list)
        with logging_redirect_tqdm():
            results = await self.evaluate_all_problems(data, agent, max_concurrent_tasks)
        columns = self.get_result_columns()
        return self.print_and_save_results_to_jsonl(results, columns)

    async def run_baseline(self, agent: Callable, max_concurrent_tasks: int = 50):
        data = await self.load_data()
        results = await self.evaluate_all_problems(data, agent, max_concurrent_tasks)
        columns = self.get_result_columns()
        return self.print_and_save_results_to_jsonl(results, columns)