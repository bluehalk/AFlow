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
        programmer = Programmer(llm)
        sc_ensemble = ScEnsemble(llm)

        llm.usage_tracker.overall_input_tokens = 0
        llm.usage_tracker.overall_output_tokens = 0
        llm.usage_tracker.call_count = 0
        llm.usage_tracker.usage_history = []

        # # Use Programmer to generate and execute Python code
        # code_solution = await programmer(problem=problem)
        
        # # Use Custom to refine and format the answer
        # refined_solution = await custom(input=problem + f"\nCode output: {code_solution['output']}", instruction=REFINE_ANSWER_PROMPT)
        
        # # Generate a detailed step-by-step solution using Custom
        # detailed_solution = await custom(input=problem, instruction=DETAILED_SOLUTION_PROMPT)
        
        # # Generate multiple solutions using Custom
        # solutions = [
        #     refined_solution['response'],
        #     detailed_solution['response']
        # ]
        final_solution = await custom(input=problem, instruction="Let's think step by step.")
        
        usage_summary = llm.usage_tracker.get_summary() 
        input_tokens = usage_summary['overall_input_tokens']
        output_tokens = usage_summary['overall_output_tokens']
        call_count = usage_summary['call_count']


        return final_solution['response'], input_tokens, output_tokens, call_count
                    