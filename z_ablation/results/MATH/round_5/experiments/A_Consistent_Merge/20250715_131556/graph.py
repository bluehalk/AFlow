from typing import Literal
from scripts.operators import Custom, Programmer, ScEnsemble, MathCodeDualVerifier
from .prompt import MERGE_PROMPT, RECHECK_PROMPT, RECHECK_PROMPT_WITH_JUDGEMENT, JUDGER_PROMPT
from scripts.async_llm import create_llm_instance
from scripts.logs import logger

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
    async def __call__(self, problem: str):
        """
        Implementation of the graph
        """
        llm = create_llm_instance(self.llm_config)
        math_code_dual_verifier = MathCodeDualVerifier(
            llm, 
            name="MathCodeDualVerifier", 
            dual_answer_generation_prompt=MERGE_PROMPT, 
            recheck_with_judgement_prompt=RECHECK_PROMPT_WITH_JUDGEMENT,
            judger_prompt=JUDGER_PROMPT
        )

        llm.usage_tracker.overall_input_tokens = 0
        llm.usage_tracker.overall_output_tokens = 0
        llm.usage_tracker.call_count = 0
        llm.usage_tracker.usage_history = []

        # Use Programmer to generate and execute Python code
        # code_solution = await programmer(problem=problem) # Programmer会自动生成代码并执行，循环三次。如果执行失败，会返回错误信息
        
        # refined_solution = await custom(input=problem + f"\nCode: {code_solution['code']}\n Code output: {code_solution['output']}", instruction=REFINE_ANSWER_PROMPT)
        
        detailed_solution = await math_code_dual_verifier(problem=problem)
        
        # solutions = [
        #     refined_solution['response'],
        #     detailed_solution['response']
        # ]
        # for _ in range(2):
        #     solution = await custom(input=problem, instruction=GENERATE_SOLUTION_PROMPT)
        #     solutions.append(solution['response'])
        
        # Use ScEnsemble to select the best solution
        # final_solution = await sc_ensemble(solutions=solutions, problem=problem)
        
        usage_summary = llm.usage_tracker.get_summary() 
        input_tokens = usage_summary['overall_input_tokens']
        output_tokens = usage_summary['overall_output_tokens']
        call_count = usage_summary['call_count']

        return detailed_solution['response'], input_tokens, output_tokens, call_count
                    