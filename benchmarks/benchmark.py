import asyncio
import json
import os
import sys
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
    def __init__(self, name: str, file_path: str, log_path: str, batch_size: int = None):
        self.name = name
        self.file_path = file_path
        # 如果环境变量中有实验文件夹，优先使用它
        self.log_path = os.environ.get('EXPERIMENT_DIR', log_path)
        self.session_time = datetime.now().strftime("%m_%d_%H_%M")
        
        # batch保存相关
        self.batch_size = batch_size
        self.batch_results = []
        
        # 确保log_path目录存在
        os.makedirs(self.log_path, exist_ok=True)

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

    def save_batch_to_csv(self, results: List[Tuple[Any, ...]], columns: List[str], batch_num: int):

        """保存一个batch的结果到CSV"""
        if not results:
            return
            
        df = pd.DataFrame(results, columns=columns)
        avg_score = df["score"].mean() if "score" in df.columns else 0.0
        partial_summary_info = f"""
        Batch {batch_num}
        Avg score: {avg_score:.3f}
        Tokens: {df["input_tokens"].sum()} + {df["output_tokens"].sum()} = {df["total_tokens"].sum()}
        Total calls: {df["calls"].sum()}
        Avg tokens: {df["total_tokens"].mean()}
        Avg calls: {df["calls"].mean()}
        """
        # 生成batch文件名
        timestamp = datetime.now().strftime("%m_%d_%H_%M")
        csv_filename = f"batch_{batch_num:02d}_score_{avg_score:.3f}_{timestamp}.csv"
        csv_output_file = os.path.join(self.log_path, csv_filename)
        
        # 清理长文本字段
        for col in ['prediction', 'expected_output', 'question']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('\n', '<br>')
        
        df.to_csv(csv_output_file, index=False)
        logger.info(f"Batch {batch_num} saved: {csv_filename} (avg_score: {avg_score:.3f})")
        with open(csv_output_file, "a") as f:
            f.write(partial_summary_info)

        return csv_output_file

    def cleanup_batch_files(self):
        """清理batch文件，只保留最终的汇总文件"""
        import glob
        
        # 如果没有使用batch保存，直接返回
        if not self.batch_size:
            return
        
        # 查找所有batch文件
        batch_pattern = os.path.join(self.log_path, "batch_*.csv")
        batch_files = glob.glob(batch_pattern)
        
        if not batch_files:
            logger.info("没有找到batch文件需要清理")
            return
        
        # 删除所有batch文件
        deleted_count = 0
        for batch_file in batch_files:
            try:
                os.remove(batch_file)
                logger.info(f"已删除batch文件: {os.path.basename(batch_file)}")
                deleted_count += 1
            except Exception as e:
                logger.warning(f"删除batch文件失败: {batch_file}, 错误: {e}")
        
        logger.info(f"已清理 {deleted_count} 个batch文件，保留最终汇总文件")

    def print_and_save_results_to_csv(self, results: List[Tuple[Any, ...]], columns: List[str]):
        df = pd.DataFrame(results, columns=columns)
        avg_score = df["score"].mean()
        total_tokens = df["total_tokens"].sum()
        avg_tokens = df["total_tokens"].mean()
        avg_calls = df["calls"].mean()
        total_calls = df["calls"].sum()
        input_tokens = df["input_tokens"].sum()
        output_tokens = df["output_tokens"].sum()

        # 使用环境变量中的时间戳，如果有的话
        experiment_timestamp = os.environ.get('EXPERIMENT_TIMESTAMP')
        if experiment_timestamp:
            current_time = experiment_timestamp
        else:
            current_time = datetime.now().strftime("%m_%d_%H_%M")

        for col in ['prediction', 'expected_output', 'question']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('\n', '<br>')
            
        csv_filename = f"Detailed_{avg_score:.3f}.csv"
        csv_output_file = os.path.join(self.log_path, csv_filename)
        df.to_csv(csv_output_file, index=False)

        summary_filename = f"Summary_{avg_score:.3f}.txt"
        summary_output_file = os.path.join(self.log_path, summary_filename)

        summary_info = f"""
        Average score on {self.name} dataset: {avg_score:.5f}
        Tokens: {input_tokens} + {output_tokens} = {total_tokens}
        Total calls: {total_calls}
        Avg tokens: {avg_tokens:.5f}
        Avg calls: {avg_calls:.5f}
        """

        # 将总token数和平均token数写入文件
        with open(summary_output_file, "w", encoding="utf-8") as f:
            f.write(summary_info)

        logger.info(summary_info)
        logger.info(f"Results saved to {csv_output_file}")
        logger.info(f"Summary saved to {summary_output_file}")

        return avg_score, avg_tokens, input_tokens, output_tokens, total_tokens

    def log_mismatch(
        self,
        problem: str,
        expected_output: Any,
        prediction: str,
        extracted_output: Any,
        extract_answer_code: str = "None",
        **data,
    ):
        log_data = {
            **data,
            "problem": problem,
            "expected_output": expected_output,
            "model_output": prediction,
            "extracted_output": extracted_output,
            "extract_answer_code": extract_answer_code,
        }
        
        # 使用环境变量中的时间戳，如果有的话
        experiment_timestamp = os.environ.get('EXPERIMENT_TIMESTAMP')
        if experiment_timestamp:
            session_time = experiment_timestamp
        else:
            session_time = self.session_time
            
        log_file = Path(self.log_path) / f"Failed_samples.jsonl"
     
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
        logger.info(f"Saved {len(data)} failed samples to {log_file}")

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
        batch_results = []
        batch_num = 0

        async def sem_evaluate(problem):
            async with semaphore:
                return await self.evaluate_problem(problem, agent)

        # 创建所有任务
        tasks = [sem_evaluate(problem) for problem in data]

        if self.batch_size:
            for i in tqdm(range(0, len(tasks), self.batch_size), desc=f"Batch Progress", total=len(tasks)//self.batch_size + (1 if len(tasks) % self.batch_size else 0)):
                batch_tasks = tasks[i:i+self.batch_size]
                batch_results = await tqdm_asyncio.gather(*batch_tasks, desc=f"Evaluating {self.name} problems", total=len(batch_tasks))
                all_results.extend(batch_results)
                self.save_batch_to_csv(batch_results, columns, batch_num)
                batch_num += 1
        else:
            all_results = await tqdm_asyncio.gather(*tasks, desc=f"Evaluating {self.name} problems", total=len(tasks))

        # 使用tqdm.asyncio显示进度，同时保持顺序
        # with tqdm(total=len(tasks), desc=f"Evaluating {self.name} problems") as pbar:
        #     for i, task in enumerate(tasks):
        #         result = await task
        #         all_results.append(result)
        #         batch_results.append(result)
        #         pbar.update(1)
                
        #         # 如果启用了batch保存，且达到了batch_size
        #         if self.batch_size and len(batch_results) >= self.batch_size:
        #             batch_num += 1
        #             self.save_batch_to_csv(batch_results, columns, batch_num)
        #             batch_results = []  # 清空当前batch
        
        # # 保存剩余的batch
        # if self.batch_size and batch_results:
        #     batch_num += 1
        #     self.save_batch_to_csv(batch_results, columns, batch_num)
        
        return all_results

    async def run_evaluation(self, agent: Callable, va_list: List[int], max_concurrent_tasks: int = 20):
        # NOTE(sjh): va_list = [0,1,2,3,4,5,6,7,8,9] indices of the dataset
        data = await self.load_data(va_list)

        #NOTE results: list of tuples, each tuple is a result of one problem
        with logging_redirect_tqdm():
            results = await self.evaluate_all_problems(data, agent, max_concurrent_tasks)
        columns = self.get_result_columns()
        
        average_score, average_tokens, input_tokens, output_tokens, total_tokens = self.print_and_save_results_to_csv(results, columns)
        # self.cleanup_batch_files()
        return average_score, average_tokens, input_tokens, output_tokens, total_tokens
    

    async def run_baseline(self, agent: Callable, max_concurrent_tasks: int = 50):
        data = await self.load_data()
        results = await self.evaluate_all_problems(data, agent, max_concurrent_tasks)
        columns = self.get_result_columns()
        average_score, average_tokens, input_tokens, output_tokens, total_tokens = self.print_and_save_results_to_csv(results, columns)
        return average_score, average_tokens, input_tokens, output_tokens, total_tokens

