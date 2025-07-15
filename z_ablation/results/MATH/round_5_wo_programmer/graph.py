from typing import Literal
from scripts.operators import Custom, Programmer, ScEnsemble
from .prompt import REFINE_ANSWER_PROMPT, DETAILED_SOLUTION_PROMPT, GENERATE_SOLUTION_PROMPT
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
    async def __call__(self, problem: str):
        """
        Implementation of the graph
        """
        llm = create_llm_instance(self.llm_config)
        custom = Custom(llm)
        sc_ensemble = ScEnsemble(llm)

        llm.usage_tracker.overall_input_tokens = 0
        llm.usage_tracker.overall_output_tokens = 0
        llm.usage_tracker.call_count = 0
        llm.usage_tracker.usage_history = []


        # Generate a detailed step-by-step solution using Custom
        detailed_solution = await custom(input=problem, instruction=DETAILED_SOLUTION_PROMPT)
        
        # Generate multiple solutions using Custom
        solutions = [
            detailed_solution['response']
        ]
        for _ in range(2):
            solution = await custom(input=problem, instruction=GENERATE_SOLUTION_PROMPT)
            solutions.append(solution['response'])
        
        # Use ScEnsemble to select the best solution
        final_solution = await sc_ensemble(solutions=solutions, problem=problem)

         solution = await custom(input=problem, instruction=GENERATE_SOLUTION_PROMPT)
        
        usage_summary = llm.usage_tracker.get_summary() 
        input_tokens = usage_summary['overall_input_tokens']
        output_tokens = usage_summary['overall_output_tokens']
        call_count = usage_summary['call_count']


        return final_solution['response'], input_tokens, output_tokens, call_count
                    