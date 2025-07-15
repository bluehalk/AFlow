# -*- coding: utf-8 -*-
# @Date    : 2025-03-31
# @Author  : didi & zhaoyang
# @Desc    : operator demo of aflow

import asyncio
import concurrent.futures
import random
import sys
import traceback
from collections import Counter
from typing import Any, Dict, List, Tuple, Optional
import signal
import re
from tenacity import retry, stop_after_attempt, wait_fixed
import regex
from math import isclose
from sympy import simplify, N, parse_expr
import asyncio
import shlex
import os
from scripts.utils.math_equivalence import judge_symbolic_equality


# 尝试导入LaTeX解析器，如果失败则使用备用方案
try:
    from sympy.parsing.latex import parse_latex
    LATEX_AVAILABLE = True
except ImportError:
    LATEX_AVAILABLE = False
    parse_latex = None

from scripts.async_llm import AsyncLLM
from scripts.logs import logger
from scripts.formatter import BaseFormatter, FormatError, XmlFormatter, TextFormatter, CodeFormatter
from scripts.operator_an import (
    AnswerGenerateOp,
    CodeGenerateOp,
    FormatOp,
    GenerateOp,
    MdEnsembleOp,
    ReflectionTestOp,
    ReviewOp,
    ReviseOp,
    ScEnsembleOp,
    DualAnswerGenerateOp,
) # All BaseModel

from scripts.prompts.prompt import (
    ANSWER_GENERATION_PROMPT,
    FORMAT_PROMPT,
    MD_ENSEMBLE_PROMPT,
    PYTHON_CODE_VERIFIER_PROMPT,
    REFLECTION_ON_PUBLIC_TEST_PROMPT,
    REVIEW_PROMPT,
    REVISE_PROMPT,
    SC_ENSEMBLE_PROMPT,
)
from scripts.utils.code import (
    extract_test_cases_from_jsonl,
    test_case_2_test_function,
    exec_code,
    extract_python_code,
)

class Operator:
    def __init__(self, llm: AsyncLLM, name: str):
        self.name = name
        self.llm = llm
        self.log_path = os.environ.get('EXPERIMENT_DIR', "")

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    async def _fill_node(self, op_class, prompt, mode=None, problem="", **extra_kwargs):
        # Create appropriate formatter based on mode
        formatter = self._create_formatter(op_class, mode)
        
        try:
            # Use the formatter with AsyncLLM
            if formatter:
                response = await self.llm.call_with_format(prompt, formatter, problem)
            else:
                # Fallback to direct call if no formatter is needed
                response = await self.llm(prompt, problem)
                

            # Convert to expected format based on the original implementation
            if isinstance(response, dict):
                return response
            else:
                return {"response": response}
        except FormatError as e:
            print(f"Format error in {self.name}: {str(e)}")
            return {"error": str(e)}
    
    def _create_formatter(self, op_class, mode=None) -> Optional[BaseFormatter]:
        """Create appropriate formatter based on operation class and mode"""
        if mode == "xml_fill":
            # 通过类方法来实例化
            return XmlFormatter.from_model(op_class) 
        elif mode == "code_fill":
            return CodeFormatter()
        elif mode == "single_fill":
            return TextFormatter()
        else:
            # Return None if no specific formatter is needed
            return None


class Custom(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "Custom"):
        super().__init__(llm, name)

    async def __call__(self, input, instruction):
        prompt = instruction + input
        response = await self._fill_node(GenerateOp, prompt, mode="single_fill")
        return response
        
class OneshotCustom(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "OneshotCustom"):
        super().__init__(llm, name)
    
    async def __call__(self, input, instruction):
        prompt = instruction + input
        response = await self._fill_node(OneshotCustomOp, prompt, mode="xml_fill")
        return response

class Judger(Operator):
    """
    一个独立的裁判，用于分析和判断两个不一致的答案。
    如果建议中包含可执行代码，会先执行并返回结果。
    """
    def __init__(self, llm: AsyncLLM, name: str = "Judger", judger_prompt: str = ""):
        super().__init__(llm, name)
        self.judger_prompt = judger_prompt

    def _extract_python_code(self, text: str) -> Optional[str]:
        """从markdown中提取python代码"""
        return extract_python_code(text)

    @retry(stop=stop_after_attempt(2), wait=wait_fixed(1))
    async def __call__(self, problem: str, math_solution: str, code_solution: str, math_answer: str, code_answer: str) -> dict:
        """
        调用Judger来分析不一致性。
        """
        prompt = self.judger_prompt.format(
            problem=problem,
            math_solution=math_solution,
            code_solution=code_solution,
            math_answer=math_answer,
            code_answer=code_answer,
        )
        response = await self._fill_node(None, prompt, mode=None)
        response_text = response.get("response", "")
        
        # 新增逻辑：检测并执行建议中的代码
        executable_code = self._extract_python_code(response_text)
        execution_result = None
        if executable_code:
            logger.info("👨‍⚖️ Judger provided executable feedback. Executing...")
            # 使用独立的exec_code函数，对于Judger的调试代码需要看到所有print输出
            status, output = await exec_code(executable_code, timeout=5, only_solve_function=False)
            if status == "Success":
                execution_result = output
                logger.info(f"✅ Executable feedback ran successfully. Output: {output}")
            else:
                execution_result = f"Error executing feedback code: {output}"
                logger.warning(f"⚠️ Executable feedback failed to run. Error: {output}")
        
        response["executable_feedback_result"] = execution_result
        return response


class MathCodeDualVerifier(Operator):
    """
    Operator that performs dual verification using both mathematical reasoning and code execution.
    It first generates a response containing both a mathematical solution and a Python code solution.
    Then, it executes the code and compares its output with the mathematical answer.
    If they are consistent, it returns the answer.
    If not, it enters a recheck loop to correct the discrepancy.
    """

    def __init__(
        self,
        llm: AsyncLLM,
        name: str = "MathCodeDualVerifier",
        dual_answer_generation_prompt: str = "",
        recheck_prompt: str = "",
        recheck_with_judgement_prompt: str = "",
        judger_prompt: str = "",
    ):
        super().__init__(llm, name)
        self.dual_answer_generation_prompt = dual_answer_generation_prompt
        self.recheck_with_judgement_prompt = recheck_with_judgement_prompt
        self.recheck_prompt = recheck_prompt
        self.judger_prompt = judger_prompt
        self.judger = Judger(llm, name="Judger", judger_prompt=judger_prompt)

    def extract_model_answer(self, text: str) -> str:
        pattern = r"\\boxed{((?:[^{}]|{[^{}]*})*)}"
        boxed_matches = re.findall(pattern, text, re.DOTALL)
        if boxed_matches:
            return boxed_matches[-1].strip()

        sentence_end_pattern = r"(?<!\d)[.!?]\s+"
        sentences = re.split(sentence_end_pattern, text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences[-1] if sentences else ""

    async def judge_consistency(self, problem: str, math_answer: str, code_answer: str) -> Tuple[int, str]:
        extracted_math_answer = self.extract_model_answer(math_answer)
        extracted_code_answer = self.extract_model_answer(code_answer)

        log_dir= os.environ.get('EXPERIMENT_DIR', self.log_path)
        log_path = os.path.join(log_dir, "consistency_log.txt")

        logger.info(f"🔍 Cross-validating answers:")
        logger.info(f"  Math answer: {extracted_math_answer}")
        logger.info(f"  Code answer: {extracted_code_answer}")

        with open(log_path, "a") as f:
            f.write(f"Problem: {problem[:66]}\n")
            f.write(f"Compare math_answer: {extracted_math_answer} and code_answer: {extracted_code_answer} ")

        judge_1 = self.math_equal(extracted_code_answer, extracted_math_answer)
        judge_2 = judge_symbolic_equality(extracted_code_answer, extracted_math_answer)


        if judge_1 or judge_2:
            with open(log_path, "a") as f:
                f.write(f"✅ Answers are equivalent\n")
            logger.info(f"✅ Answers are equivalent\n")
            return 1, extracted_math_answer, extracted_code_answer
        else:
            with open(log_path, "a") as f:
                f.write(f"❌ Answers are not equivalent\n")
            logger.info(f"❌ Answers are not equivalent\n")
            return 0, extracted_math_answer, extracted_code_answer

    def math_equal(self, prediction: Any, reference: Any) -> bool:
        if str(prediction) == str(reference):
            return True

        try:
            if self.is_digit(prediction) and self.is_digit(reference):
                prediction = self.parse_digits(prediction)
                reference = self.parse_digits(reference)
                return isclose(prediction, reference, abs_tol=1e-3)
        except:
            pass

        try:
            return self.symbolic_equal(prediction, reference)
        except:
            pass

        return False
    def is_digit(self, num):
        return self.parse_digits(num) is not None

    def parse_digits(self, num):
        num = regex.sub(",", "", str(num))
        try:
            return float(num)
        except:
            if num.endswith("%"):
                num = num[:-1]
                if num.endswith("\\"):
                    num = num[:-1]
                try:
                    return float(num) / 100
                except:
                    pass
        return None

    def symbolic_equal(self, a, b):
        def _parse(s):
            for f in [parse_latex, parse_expr]:
                try:
                    return f(s)
                except:
                    pass
            return s

        a = _parse(a)
        b = _parse(b)

        try:
            if simplify(a - b) == 0:
                return True
        except:
            pass

        try:
            if isclose(N(a), N(b), abs_tol=1e-3):
                return True
        except:
            pass
        return False

    async def dual_answer_generate(self, prompt, mode):
        response = await self._fill_node(None, prompt, mode)
        return response

    async def validate_response(self, response: str) -> Tuple[bool, dict]:

        pattern = r"<(\w+)>(.*?)</\1>"
        matches = re.findall(pattern, response, re.DOTALL)

        #NOTE(sjh) 字段名为键，字段值为值
        found_fields = {match[0]: match[1].strip() for match in matches}
        if "python_code" in found_fields and "math_solution" in found_fields:
            return True, found_fields
        else:
            return False, found_fields


    @retry(stop=stop_after_attempt(1), wait=wait_fixed(2))
    async def __call__(self, problem: str):
        code = None
        code_answer = None
        feedback = ""

        # 初始Prompt
        system_prompt = self.dual_answer_generation_prompt.format(
            problem=problem,
        )
        current_prompt = system_prompt

        for i in range(3):
            # 在每次迭代中，我们都使用精心构建的 current_prompt
            # 而不是一个不断增长的 history 列表

            logger.info(f"\n 🔄 current prompt: {current_prompt}\n\n")

            if i == 0:
                dual_answers = await self.dual_answer_generate([{"role": "user", "content": current_prompt}], mode=None)
            else:
                dual_answers = await self.dual_answer_generate([{"role": "system", "content": system_prompt}, {"role": "user", "content": current_prompt}], mode=None)

            is_valid, dual_answer_response = await self.validate_response(dual_answers["response"])
            
            if not is_valid:
                # 如果格式无效，构建一个简单的重试请求
                feedback = f"\n\nThe answer should strictly follow the required format. Please try again."
                current_prompt = current_prompt + feedback 
                continue

            code = dual_answer_response.get("python_code", "")
            math_answer = dual_answer_response.get("math_solution", "")
        
            # 执行代码并检查一致性  
            status, code_answer = await exec_code(code, timeout=5, only_solve_function=True)

            if status == "Success":
                uni_score, extracted_math_answer, extracted_code_answer = await self.judge_consistency(problem, math_answer, code_answer)
                if uni_score == 1:
                    return {"response": code_answer}
                else:
                    # 答案不一致，调用 Judger 并构建选择性历史
                    logger.info("🚨 Answers are inconsistent. Calling Judger for analysis...")
                    judgement_response = await self.judger(
                        problem=problem,
                        math_solution=math_answer,
                        code_solution=code,
                        math_answer=extracted_math_answer,
                        code_answer=extracted_code_answer
                    )
                    
                    judgement_text = judgement_response.get("response", "No analysis provided.")
                    executable_feedback_result = judgement_response.get("executable_feedback_result")
                    
                    # 准备Prompt的上下文
                    judgement_context = {
                        "problem": problem,
                        "math_answer": extracted_math_answer,
                        "code_answer": extracted_code_answer,
                        "judgement": judgement_text,
                        "executable_feedback_result": "" # 默认为空字符串
                    }
                    
                    if executable_feedback_result:
                        judgement_context["executable_feedback_result"] = (
                            "\n**Judge's Executable Suggestion Result**:\n"
                            "The judge provided a code snippet in the suggestion. We executed it and here is the result:\n"
                            f"---\n{executable_feedback_result}\n---\n"
                        )

                    # 构建新的、精简的 prompt
                    current_prompt = self.recheck_with_judgement_prompt.format(**judgement_context)

            else:
                # 代码执行失败，构建反馈
                logger.info(f"Execution error on attempt {i + 1}, error message: {code_answer}")
                feedback = f"\nThe code execution failed with error message: {code_answer}. Please ensure your code is correct and can be executed."
                current_prompt = current_prompt + feedback 

        return {"response": dual_answers["response"]} # output就是执行的结果




class AnswerGenerate(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "AnswerGenerate"):
        super().__init__(llm, name)

    async def __call__(self, input: str) -> Tuple[str, str]:
        prompt = ANSWER_GENERATION_PROMPT.format(input=input)
        response = await self._fill_node(AnswerGenerateOp, prompt, mode="xml_fill")
        return response


class CustomCodeGenerate(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "CustomCodeGenerate"):
        super().__init__(llm, name)

    async def __call__(self, problem, entry_point, instruction):
        prompt = instruction + problem
        response = await self._fill_node(GenerateOp, prompt, mode="code_fill", function_name=entry_point)
        return response


class ScEnsemble(Operator):
    """
    Paper: Self-Consistency Improves Chain of Thought Reasoning in Language Models
    Link: https://arxiv.org/abs/2203.11171
    Paper: Universal Self-Consistency for Large Language Model Generation
    Link: https://arxiv.org/abs/2311.17311
    """

    def __init__(self, llm: AsyncLLM, name: str = "ScEnsemble"):
        super().__init__(llm, name)

    async def __call__(self, solutions: List[str], problem: str):
        answer_mapping = {}
        solution_text = ""
        for index, solution in enumerate(solutions):
            answer_mapping[chr(65 + index)] = index
            solution_text += f"{chr(65 + index)}: \n{str(solution)}\n\n\n"

        prompt = SC_ENSEMBLE_PROMPT.format(question=problem, solutions=solution_text)
        response = await self._fill_node(ScEnsembleOp, prompt, mode="xml_fill")

        answer = response.get("solution_letter", "")
        answer = answer.strip().upper()

        return {"response": solutions[answer_mapping[answer]]}


def run_code(code):
    try:
        # Create a new global namespace
        global_namespace = {}

        disallowed_imports = [
            "os",
            "sys",
            "subprocess",
            "multiprocessing",
            "matplotlib",
            "seaborn",
            "plotly",
            "bokeh",
            "ggplot",
            "pylab",
            "tkinter",
            "PyQt5",
            "wx",
            "pyglet",
        ]

        # Check for prohibited imports
        for lib in disallowed_imports:
            if f"import {lib}" in code or f"from {lib}" in code:
                logger.info("Detected prohibited import: %s", lib)
                return "Error", f"Prohibited import: {lib} and graphing functionalities"

        # Use exec to execute the code
        exec(code, global_namespace) # 整个代码都会执行，包括import
        # Assume the code defines a function named 'solve'
        if "solve" in global_namespace and callable(global_namespace["solve"]): # 确保solve可执行
            result = global_namespace["solve"]() # 再次执行solve，提取结果
            return "Success", str(result)
        else:
            return "Error", "Function 'solve' not found"
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tb_str = traceback.format_exception(exc_type, exc_value, exc_traceback)
        return "Error", f"Execution error: {str(e)}\n{''.join(tb_str)}"


class Programmer(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "Programmer"):
        super().__init__(llm, name)

    async def exec_code(self, code: str, timeout: int = 5) -> Tuple[str, str]:
        """
        Execute code in a subprocess with timeout.
        Returns:
            Tuple[status, output], where status is "Success" or "Error".
        """
        return await exec_code(code, timeout=timeout, only_solve_function=True)

    async def code_generate(self, problem, analysis, feedback, mode):
        """
        Asynchronous method to generate code.
        """
        prompt = PYTHON_CODE_VERIFIER_PROMPT.format(
            problem=problem,
            analysis=analysis,
            feedback=feedback
        )
        response = await self._fill_node(CodeGenerateOp, prompt, mode, problem, function_name="solve")
        return response

    @retry(stop=stop_after_attempt(2), wait=wait_fixed(2))
    async def __call__(self, problem: str, analysis: str = "None"):
        """
        Call method, generate code and execute, retry up to 3 times.
        """
        code = None
        output = None
        feedback = ""
        for i in range(3):
            code_response = await self.code_generate(problem, analysis, feedback, mode="code_fill") # 为什么这里不维护历史信息呢？
            code = code_response.get("code")
            if not code:
                return {"code": code, "output": "No code generated"}
            status, output = await self.exec_code(code, timeout=5)
            if status == "Success":
                return {"code": code, "output": output}
            else:
                logger.info(f"Execution error on attempt {i + 1}, error message: {output}, problem: {problem}")
                feedback = (
                    f"\nThe result of the error from the code you wrote in the previous round:\n"
                    f"Code: {code}\n\nStatus: {status}, {output}"
                )

        return {"code": code, "output": output} # output就是执行的结果


# test#  # test
#     @retry(stop=stop_after_attempt(2), wait=wait_fixed(2)) # 只有
#     async def __call__(self, code: str):
#         status, output = await self.exec_code(code, timeout=5)
#         return {"code": code, "output": output}



def run_test_code(test_code):
    """在独立进程中执行测试代码"""
    try:
        exec(test_code, globals())
        return "Success", "Test passed"
    except AssertionError as e:
        return "Error", f"AssertionError: {str(e)}"
    except Exception as e:
        return "Error", f"Exception: {str(e)}"

class Test(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "Test"):
        super().__init__(llm, name)
        # 创建进程池，参考Programmer的实现
        self.process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=1)

    def __del__(self):
        """确保进程池在对象销毁时关闭"""
        if hasattr(self, 'process_pool'):
            self.process_pool.shutdown(wait=True)

    async def exec_code_with_timeout(self, test_code, timeout=10):
        """异步执行代码，带超时保护，参考Programmer的实现"""
        loop = asyncio.get_running_loop()
        
        try:
            # 使用进程池执行测试代码
            future = loop.run_in_executor(self.process_pool, run_test_code, test_code)
            # 等待任务完成或超时
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            # 取消future
            future.cancel()

            # 强制关闭并重新创建进程池来终止挂起的进程
            self.process_pool.shutdown(wait=False, cancel_futures=True)
            self.process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=1)

            import gc
            gc.collect()
            return "Error", "Code execution timed out"
        except concurrent.futures.process.BrokenProcessPool:
            # 如果进程池损坏，重新创建
            self.process_pool.shutdown(wait=False)
            self.process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=1)
            return "Error", "Process pool broken, try again"
        except Exception as e:
            return "Error", f"Unknown error: {str(e)}"

    async def exec_code(self, solution, entry_point):
        test_cases = extract_test_cases_from_jsonl(entry_point)
        print(f"test_cases: {test_cases}")
        fail_cases = []
        
        for test_case in test_cases:
            test_code = test_case_2_test_function(solution, test_case, entry_point)
            
            # 使用进程池执行测试代码
            status, message = await self.exec_code_with_timeout(test_code, timeout=10)
            
            if status == "Error":
                if "AssertionError" in message:
                    # 断言失败
                    with open("tester.txt", "a") as f:
                        f.write("test_error of " + entry_point + "\n")
                    fail_cases.append({"test_fail_case": {"error_message": message}})
                else:
                    # 超时或其他异常
                    with open("tester.txt", "a") as f:
                        f.write(entry_point + " " + message + "\n")
                    return {"exec_fail_case": message}
        
        if fail_cases != []:
            return fail_cases
        else:
            return "no error"

    async def __call__(self, problem, solution, entry_point, test_loop: int = 3):
        """
        "Test": {
        "description": "Test the solution with test cases, if the solution is correct, return 'no error', if the solution is incorrect, return reflect on the soluion and the error information",
        "interface": "test(problem: str, solution: str, entry_point: str) -> str"
        }
        """
        for dual_answers in range(test_loop):
            result = await self.exec_code(solution, entry_point)

            if result == "no error":
                return {"result": True, "solution": solution}

            elif "exec_fail_case" in result:
                result = result["exec_fail_case"]
                prompt = REFLECTION_ON_PUBLIC_TEST_PROMPT.format(
                    problem=problem,
                    solution=solution,
                    exec_pass=f"executed unsuccessfully, error: \n {result}",
                    test_fail="executed unsucessfully",
                )
                response = await self._fill_node(ReflectionTestOp, prompt, mode="code_fill")
                solution = response["code"]
            else:
                # import pdb; pdb.set_trace()
                prompt = REFLECTION_ON_PUBLIC_TEST_PROMPT.format(
                    problem=problem,
                    solution=solution,
                    exec_pass="executed successfully",
                    test_fail=result,
                )
                response = await self._fill_node(ReflectionTestOp, prompt, mode="code_fill")
                solution = response["code"]

        result = await self.exec_code(solution, entry_point)
        if result == "no error":
            return {"result": True, "solution": solution}
        else:
            return {"result": False, "solution": solution}


class Format(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "Format"):
        super().__init__(llm, name)

    async def __call__(self, problem, solution, mode: str = None):
        prompt = FORMAT_PROMPT.format(problem_description=problem, solution=solution)
        response = await self._fill_node(FormatOp, prompt, mode)
        return response


class Review(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "Review"):
        super().__init__(llm, name)

    async def __call__(self, problem, solution, mode: str = None):
        prompt = REVIEW_PROMPT.format(problem=problem, solution=solution)
        response = await self._fill_node(ReviewOp, prompt, mode="xml_fill")
        return response


class Revise(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "Revise"):
        super().__init__(llm, name)

    async def __call__(self, problem, solution, feedback, mode: str = None):
        prompt = REVISE_PROMPT.format(problem=problem, solution=solution, feedback=feedback)
        response = await self._fill_node(ReviseOp, prompt, mode="xml_fill")
        return response


class MdEnsemble(Operator):
    """
    Paper: Can Generalist Foundation Models Outcompete Special-Purpose Tuning? Case Study in Medicine
    Link: https://arxiv.org/abs/2311.16452
    """

    def __init__(self, llm: AsyncLLM, name: str = "MdEnsemble", vote_count: int = 5):
        super().__init__(llm, name)
        self.vote_count = vote_count

    @staticmethod
    def shuffle_answers(solutions: List[str]) -> Tuple[List[str], Dict[str, str]]:
        shuffled_solutions = solutions.copy()
        random.shuffle(shuffled_solutions)
        answer_mapping = {chr(65 + i): solutions.index(solution) for i, solution in enumerate(shuffled_solutions)}
        return shuffled_solutions, answer_mapping

    async def __call__(self, solutions: List[str], problem: str, mode: str = None):
        logger.info(f"solution count: {len(solutions)}")
        all_responses = []

        for dual_answers in range(self.vote_count):
            shuffled_solutions, answer_mapping = self.shuffle_answers(solutions)

            solution_text = ""
            for index, solution in enumerate(shuffled_solutions):
                solution_text += f"{chr(65 + index)}: \n{str(solution)}\n\n\n"

            prompt = MD_ENSEMBLE_PROMPT.format(solutions=solution_text, question=problem)
            response = await self._fill_node(MdEnsembleOp, prompt, mode="xml_fill")

            answer = response.get("solution_letter", "A")
            answer = answer.strip().upper()

            if answer in answer_mapping:
                original_index = answer_mapping[answer]
                all_responses.append(original_index)

        most_frequent_index = Counter(all_responses).most_common(1)[0][0]
        final_answer = solutions[most_frequent_index]
        return {"solution": final_answer}
