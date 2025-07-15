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
    """
    MBPP (Mostly Basic Python Problems) 基准测试类
    
    用于评估AI模型在Python编程任务上的表现，包括：
    - 代码生成能力
    - 代码执行正确性
    - 超时处理
    - 错误诊断
    """
    
    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)
        self.failed_samples = []  # 收集失败样本，用于后续分析

    class TimeoutError(Exception):
        """自定义超时异常类"""
        pass

    def run_with_timeout(self, func, timeout):
        """
        改进的超时处理机制，使用守护线程避免需要手动Ctrl+C
        """
        result = []
        stop_event = threading.Event()  # 线程同步事件

        def target():
            """目标函数，在独立线程中执行"""
            try:
                result.append(func())  # 执行函数并保存结果
            except Exception as e:
                result.append(e)  # 保存异常信息
            finally:
                stop_event.set()  # 标记执行完成

        # 创建并启动守护线程 - 关键改进！
        thread = threading.Thread(target=target)
        thread.daemon = True  # 设置为守护线程，主程序结束时自动停止
        thread.start()
        
        # 等待执行完成或超时
        is_timeout = not stop_event.wait(timeout)

        if is_timeout:
            # 超时处理改进：尝试优雅地停止线程
            stop_event.set()  # 设置停止标志
            thread.join(timeout=1)  # 等待最多1秒让线程结束
            
            if thread.is_alive():
                # 线程仍在运行，但因为是守护线程，主程序可以继续
                # 记录警告但不阻塞主程序
                import warnings
                warnings.warn(f"Thread for function {func.__name__} is still running after timeout, but continuing due to daemon thread")
            
            raise self.TimeoutError("Function execution timed out")

        if not result:
            return None
        if isinstance(result[0], Exception):
            raise result[0]  # 重新抛出捕获的异常
        return result[0]

    def check_solution(self, solution, test, entry_point):
        solution = sanitize(code=solution, entrypoint=entry_point)
        try:
            # 创建安全的执行环境，包含常用模块和类型注解
            global_dict = {
                "math": __import__("math"),      # 数学函数
                "hashlib": __import__("hashlib"), # 哈希函数
                "re": __import__("re"),          # 正则表达式
                "List": List,                    # 类型注解
                "Dict": Dict,
                "Tuple": Tuple,
                "Optional": Optional,
                "Any": Any,
            }

            # 执行生成的解决方案代码
            exec(solution, global_dict)

            # 检查入口函数是否已定义
            if entry_point not in global_dict:
                raise ValueError(f"Function {entry_point} is not defined in the solution.")

            # 执行测试代码
            exec(test, global_dict)

            # 获取测试检查函数
            check = global_dict["check"]

            # 在15秒超时内运行测试
            result = self.run_with_timeout(check, 15)

            if result is None:
                result = (self.PASS, "The solution passed all test cases.")

        except self.TimeoutError:
            # 处理超时情况
            result = (
                self.FAIL,
                "Execution timed out. Please check if your solution contains infinite loops or overly time-consuming operations.",
            )
        except Exception as e:
            # 处理其他异常，提供详细的错误信息和堆栈跟踪
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
        """
        使用重试机制生成代码输出
        重试策略：
        - 最多重试5次
        - 每次重试间隔1秒
        """
        return await graph(prompt, entry_point)

    async def evaluate_problem(self, data: dict, graph: Callable) -> Tuple[str, str, str, str, float, dict]:
        input_text = data["prompt"]
        expected_output = "\nCorrect Solution:\ndef " + data["code"]
        
        try:
            # 使用图函数生成代码预测
            # graph 返回 (prediction:str, input_tokens:int, output_tokens:int, calls:int)
            prediction, input_tokens, output_tokens, calls = await self._generate_output(
                graph, input_text, data["entry_point"]
            )

            # 检查生成的解决方案
            ret = self.check_solution(prediction, data["test"], data["entry_point"])
            error_message = ret[1]
            correct_solution = data["code"]

            score = 1.0 if ret[0] == self.PASS else 0.0

            if score == 0:
                self.log_mismatch(problem=input_text, expected_output=correct_solution, prediction=prediction, extracted_output=error_message, **data)

            return (
                input_text + '\n',           # 输入文本
                prediction + '\n',           # 生成的代码
                error_message + '\n',        # 错误信息或成功消息
                correct_solution + '\n',     # 正确答案
                score,                       # 分数
                input_tokens,                # 输入token数
                output_tokens,               # 输出token数
                input_tokens + output_tokens, # 总token数
                calls,                       # 调用次数
            )

        except Exception as e:
            # 处理生成过程中的异常
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {e}")
            # 异常情况下也收集失败样本
            self.log_mismatch(problem=input_text, expected_output=correct_solution, prediction=prediction, extracted_output=error_message, **data)
            return (
                input_text + '\n',
                str(e) + '\n',
                str(e) + '\n',
                data.get("code", "") + '\n',
                0.0,  # 异常情况下得分为0
                0,    # token统计为0
                0,
                0,
                0,
            )

    # def save_failed_samples(self):
    #     """
    #     将失败样本保存到 jsonl 文件，用于后续分析和调试
    #     """
    #     if not self.failed_samples:
    #         logger.info("No failed samples to save.")
    #         return
            
    #     # 生成带时间戳的文件名
    #     from datetime import datetime
    #     time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     failed_file = os.path.join(self.log_path, f"failed_samples_{time_str}.jsonl")
        
    #     # 保存失败样本到JSONL格式文件
    #     with open(failed_file, 'w', encoding='utf-8') as f:
    #         for sample in self.failed_samples:
    #             f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
    #     logger.info(f"Saved {len(self.failed_samples)} failed samples to {failed_file}")

    def calculate_score(self, expected_output: str, prediction: str) -> Tuple[float, str]:
        """
        计算分数的方法（MBPP的评分逻辑已在evaluate_problem中实现）
        此方法仅为了符合接口要求
        """
        return 0.0, prediction

    def get_result_columns(self) -> List[str]:
        """
        获取结果列名列表，定义了评估结果的输出格式
        """
        return [
            "inputs",           # 题目描述
            "prediction",       # 生成代码
            "error_message",    # 失败时的详细错误 / 通过时的 PASS 信息
            "correct_solution", # 官方参考实现
            "score",            # 1/0 分数
            "input_tokens",     # prompt tokens 数量
            "output_tokens",    # completion tokens 数量
            "total_tokens",     # 两者之和
            "calls",            # 调用次数
        ]
