from typing import Literal
from scripts.operators import *
import prompt
from metagpt.provider.llm_provider_registry import create_llm_instance
from metagpt.utils.cost_manager import CostManager

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
        self.llm = create_llm_instance(llm_config)
        self.llm.cost_manager = CostManager()
        self.custom = Custom(self.llm)
        self.custom_code_generate = CustomCodeGenerate(self.llm)
        self.test = Test(self.llm)
        self.sc_ensemble = ScEnsemble(self.llm)

    async def __call__(self, problem: str, entry_point: str):
        """
        Implementation of the graph
        Custom operator to generate anything you want.
        But when you want to get standard code, you should use custom_code_generate operator.
        """
        solutions = []
        for _ in range(3):  # Generate 3 solutions
            solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction=prompt.CODE_GENERATE_PROMPT)
            solutions.append(solution['response'])
        
        best_solution = await self.sc_ensemble(solutions=solutions, problem=problem)
        
        test_result = await self.test(problem=problem, solution=best_solution['response'], entry_point=entry_point)
        
        if test_result['result']:
            return test_result['solution'], self.llm.cost_manager.total_cost
        else:
            # If the test fails, try to fix the solution
            fixed_solution = await self.custom(input=f"Problem: {problem}\nFailed solution: {best_solution['response']}\nError: {test_result['solution']}", instruction=prompt.FIX_CODE_PROMPT)
            return fixed_solution['response'], self.llm.cost_manager.total_cost
