import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import json
import os
from pathlib import Path

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from benchmarks.benchmark import BaseBenchmark
from scripts.logs import logger
from scripts.utils.sanitize import sanitize




class MBPPBenchmark(BaseBenchmark):
    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)
        self.failed_samples = []  # 收集失败样本

    class TimeoutError(Exception):
        pass

    def run_with_timeout(self, func, timeout):
        result = []
        stop_event = threading.Event()

        def target():
            try:
                result.append(func())
            except Exception as e:
                result.append(e)
            finally:
                stop_event.set()

        thread = threading.Thread(target=target)
        thread.start()
        is_timeout = not stop_event.wait(timeout)

        if is_timeout:
            raise self.TimeoutError("Function execution timed out")

        if not result:
            return None
        if isinstance(result[0], Exception):
            raise result[0]
        return result[0]

    def check_solution(self, solution, test, entry_point):
        solution = sanitize(code=solution, entrypoint=entry_point)
        try:
            global_dict = {
                "math": __import__("math"),
                "hashlib": __import__("hashlib"),
                "re": __import__("re"),
                "List": List,
                "Dict": Dict,
                "Tuple": Tuple,
                "Optional": Optional,
                "Any": Any,
            }

            exec(solution, global_dict)

            if entry_point not in global_dict:
                raise ValueError(f"Function {entry_point} is not defined in the solution.")

            exec(test, global_dict)

            check = global_dict["check"]

            result = self.run_with_timeout(check, 15)

            if result is None:
                result = (self.PASS, "The solution passed all test cases.")

        except self.TimeoutError:
            result = (
                self.FAIL,
                "Execution timed out. Please check if your solution contains infinite loops or overly time-consuming operations.",
            )
        except Exception as e:
            # 为了更清晰地定位错误，记录异常类型与完整回溯
            import traceback
            tb = traceback.format_exc()
            error_message = (
                f"{type(e).__name__}: {str(e)}\n"
                f"Traceback:\n{tb}\n"
                f"Solution:\n{solution}\n"
                f"Test Code:\n{test}"
            )
            result = (self.FAIL, error_message)
        return result

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=True) 
    async def _generate_output(self, graph, prompt, entry_point):
        return await graph(prompt, entry_point)

    async def evaluate_problem(self, data: dict, graph: Callable) -> Tuple[str, str, str, str, float, dict]:
        input_text = data["prompt"]
        expected_output = "\nCorrect Solution:\ndef " + data["code"]
        
        try:
            # Generate prediction using the graph function
            # graph 返回 (prediction:str, input_tokens:int, output_tokens:int, calls:int)
            prediction, input_tokens, output_tokens, calls = await self._generate_output(
                graph, input_text, data["entry_point"]
            )

            # Check the solution
            ret = self.check_solution(prediction, data["test"], data["entry_point"])
            error_message = ret[1]
            correct_solution = data["code"]

            # Calculate score based on the check result
            score = 1.0 if ret[0] == self.PASS else 0.0

            # Log mismatch if the score is 0
            if score == 0:
                self.log_mismatch(input_text, correct_solution, prediction, error_message)
                # 收集失败样本到内存
                self.failed_samples.append({**data, "error_message": error_message, "prediction": prediction})

            return (
                input_text + '\n',
                prediction + '\n',
                error_message + '\n',
                correct_solution + '\n',
                score,
                input_tokens,
                output_tokens,
                input_tokens + output_tokens,
                calls,
            )

        except Exception as e:
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {e}")
            # 异常情况下也收集失败样本
            self.failed_samples.append({**data, "error_message": str(e), "prediction": str(e)})
            return (
                input_text + '\n',
                str(e) + '\n',
                str(e) + '\n',
                data.get("code", "") + '\n',
                0.0,
                0,   
                0,
                0,
                0,
            )

    def save_failed_samples(self):
        """将失败样本保存到 jsonl 文件"""
        if not self.failed_samples:
            logger.info("No failed samples to save.")
            return
        
        failed_file = os.path.join(self.log_path, "failed_samples.jsonl")
        with open(failed_file, 'w', encoding='utf-8') as f:
            for sample in self.failed_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(self.failed_samples)} failed samples to {failed_file}")

    def calculate_score(self, expected_output: str, prediction: str) -> Tuple[float, str]:
        # The scoring logic for MBPP is already implemented in evaluate_problem, this is just to conform to the interface
        return 0.0, prediction

    def get_result_columns(self) -> List[str]:
        # 调整列顺序并加入 token 统计列
        return [
            "inputs",           # 题目描述
            "prediction",       # 生成代码
            "error_message",    # 失败时的详细错误 / 通过时的 PASS 信息
            "correct_solution", # 官方参考实现
            "score",            # 1/0
            "input_tokens",     # prompt tokens
            "output_tokens",    # completion tokens
            "total_tokens",     # 两者之和
            "calls",            # 调用次数
        ]
