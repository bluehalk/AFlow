import asyncio
import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Tuple

import aiofiles
import pandas as pd
from tqdm.asyncio import tqdm_asyncio

from scripts.logs import logger
from scripts.utils.common import write_json_file


class BaseBenchmark(ABC):
    def __init__(self, name: str, file_path: str, log_path: str):
        self.name = name
        self.file_path = file_path
        self.log_path = log_path

    PASS = "PASS"
    FAIL = "FAIL"

    async def load_data(self, specific_indices: List[int] = None) -> List[dict]:
        data = []
        async with aiofiles.open(self.file_path, mode="r", encoding="utf-8") as file:
            async for line in file:
                data.append(json.loads(line))
        if specific_indices is not None:
            filtered_data = [data[i] for i in specific_indices if i < len(data)]
            return filtered_data
        return data

    def print_and_save_results_to_csv(self, results: List[Tuple[Any, ...]], columns: List[str]):
        df = pd.DataFrame(results, columns=columns)
        avg_score = df["score"].mean()
        total_tokens = df["total_tokens"].sum()
        avg_tokens = df["total_tokens"].mean()
        avg_calls = df["calls"].mean()
        total_calls = df["calls"].sum()

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{avg_score:.5f}_{current_time}.csv"
        output_file = os.path.join(self.log_path, filename)
        df.to_csv(output_file, index=False)

        logger.info(f"Average score on {self.name} dataset: {avg_score:.5f}")
        logger.info(f"Total tokens: {total_tokens}")
        logger.info(f"Avg tokens:{avg_tokens}")
        
        # 将总token数和平均token数写入文件
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"Total tokens: {total_tokens}\n") 
            f.write(f"Total calls: {total_calls}\n")

            f.write(f"Avg tokens: {avg_tokens}\n")
            f.write(f"Avg calls: {avg_calls}\n")

            f.write(f"Avg score: {avg_score}\n")

        logger.info(f"Results saved to {output_file}")

        return avg_score, avg_tokens, total_tokens

    def log_mismatch(
        self,
        problem: str,
        expected_output: Any,
        prediction: str,
        extracted_output: Any,
        extract_answer_code: str = "None",
    ):
        log_data = {
            "question": problem,
            "right_answer": expected_output,
            "model_output": prediction,
            "extracted_output": extracted_output,
            "extract_answer_code": extract_answer_code,
        }
        log_file = Path(self.log_path) / "log.json"
        if log_file.exists():
            with log_file.open("r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
        else:
            data = []
        data.append(log_data)
        write_json_file(log_file, data, encoding="utf-8", indent=4)

    @abstractmethod
    async def evaluate_problem(self, problem: dict, agent: Callable) -> Tuple[Any, ...]:
        pass

    @abstractmethod
    def calculate_score(self, expected_output: Any, prediction: Any) -> Tuple[float, Any]:
        pass

    @abstractmethod
    def get_result_columns(self) -> List[str]:
        pass

    async def evaluate_all_problems(self, data: List[dict], agent: Callable, max_concurrent_tasks: int = 15):
        semaphore = asyncio.Semaphore(max_concurrent_tasks)

        async def sem_evaluate(problem):
            async with semaphore:
                return await self.evaluate_problem(problem, agent)

        tasks = [sem_evaluate(problem) for problem in data]
        return await tqdm_asyncio.gather(*tasks, desc=f"Evaluating {self.name} problems", total=len(data))

    async def run_evaluation(self, agent: Callable, va_list: List[int], max_concurrent_tasks: int = 15):
        # NOTE(sjh): va_list = [0,1,2,3,4,5,6,7,8,9] indices of the dataset
        data = await self.load_data(va_list)

        #NOTE results: list of tuples, each tuple is a result of one problem
        results = await self.evaluate_all_problems(data, agent, max_concurrent_tasks) 
        columns = self.get_result_columns()
        
        average_score, average_tokens, total_tokens = self.print_and_save_results_to_csv(results, columns)
        
        # 如果是 MBPP benchmark，保存失败样本
        if hasattr(self, 'save_failed_samples'):
            self.save_failed_samples()

        return average_score, average_tokens, total_tokens  # 返回平均分数、平均token数、总token数
    

    async def run_baseline(self, agent: Callable, max_concurrent_tasks: int = 50):
        data = await self.load_data()
        results = await self.evaluate_all_problems(data, agent, max_concurrent_tasks)
        columns = self.get_result_columns()
        average_score, average_tokens, total_tokens = self.print_and_save_results_to_csv(results, columns)
        return average_score, average_tokens, total_tokens

