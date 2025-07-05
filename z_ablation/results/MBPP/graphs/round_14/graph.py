from typing import Literal
from scripts.operators import *
from . import prompt
from scripts.async_llm import create_llm_instance
DatasetType = Literal["HumanEval", "MBPP", "GSM8K", "MATH", "HotpotQA", "DROP"]

class Workflow:
    def __init__(
        self,
        name: str,
        llm_config,
        dataset: DatasetType,
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.llm_config = llm_config

    async def __call__(self, problem: str, entry_point: str):
        # 为每个样本创建独立的LLM实例，避免并发冲突
        llm = create_llm_instance(self.llm_config)
        custom = Custom(llm)
        custom_code_generate = CustomCodeGenerate(llm)
        test = Test(llm)
        sc_ensemble = ScEnsemble(llm)
        
        # 重置计数器
        llm.usage_tracker.overall_input_tokens = 0
        llm.usage_tracker.overall_output_tokens = 0
        llm.usage_tracker.call_count = 0
        llm.usage_tracker.usage_history = []
        
        solutions = []
        for _ in range(3):  # Generate 3 solutions
            # print("="*100)
            # print(f"==Generate solution {_}==")
            solution = await custom_code_generate(problem=problem, entry_point=entry_point, instruction=prompt.CODE_GENERATE_PROMPT)
            solutions.append(solution['code'])
        
        best_solution = await sc_ensemble(solutions=solutions, problem=problem)

        test_result = await test(problem=problem, solution=best_solution['response'], entry_point=entry_point)
        # print(test_result) # result, solution

        # 获取这个样本的token统计
        usage_summary = llm.usage_tracker.get_summary()
        input_tokens = usage_summary['overall_input_tokens']
        output_tokens = usage_summary['overall_output_tokens']
        call_count = usage_summary['call_count']

        if test_result['result']:
            # print("="*100, "test success", "="*100)
            return test_result['solution'], input_tokens, output_tokens, call_count
        else:
            # print("="*100, "test failed", "="*100)
            # If the test fails, try to fix the solution
            fixed_solution = await custom(input=f"Problem: {problem}\nFailed solution: {best_solution['response']}\nError: {test_result['solution']}", instruction=prompt.FIX_CODE_PROMPT)
            # print("="*100, "fixed_solution", "="*100)
            # print(fixed_solution)
            
            # 获取最终的token统计（包括修复步骤）
            final_usage = llm.usage_tracker.get_summary()
            final_input_tokens = final_usage['overall_input_tokens']
            final_output_tokens = final_usage['overall_output_tokens']
            final_call_count = final_usage['call_count']
            
            return fixed_solution['response'], final_input_tokens, final_output_tokens, final_call_count