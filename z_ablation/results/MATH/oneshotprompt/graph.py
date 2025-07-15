from typing import Literal
from scripts.operators import Custom, Programmer, ScEnsemble
from scripts.async_llm import create_llm_instance
from .prompt import ONESHOT_PROMPT_3

DatasetType = Literal["HumanEval", "MBPP", "GSM8K", "MATH", "HotpotQA", "DROP"]


ONESHOT_PROMPT_DEEP_WIDE_SIMPLE = """
You are an expert mathematician. Solve this problem using a two-phase approach:

## Phase 1: THINK DEEP
- Find the most direct and rigorous mathematical solution
- Execute with complete precision and detailed reasoning  
- Show every step with proper LaTeX notation and clear logic
- Ensure mathematical rigor in every detail

## Phase 2: THINK WIDE  
- Verify your solution using a different approach or perspective
- Check if your answer makes intuitive and mathematical sense
- Connect to broader mathematical principles and insights

## Final Answer
- Integrate insights from both phases
- Present final answer in \\boxed{} notation
- State your confidence level based on the two-phase verification

Focus first on depth and rigor, then on breadth and verification.
"""


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

        llm.usage_tracker.overall_input_tokens = 0
        llm.usage_tracker.overall_output_tokens = 0
        llm.usage_tracker.call_count = 0
        llm.usage_tracker.usage_history = []

        final_solution = await custom(input=problem, instruction=ONESHOT_PROMPT_DEEP_WIDE_SIMPLE)
        # ONESHOT_PROMPT_3
        usage_summary = llm.usage_tracker.get_summary() 
        input_tokens = usage_summary['overall_input_tokens']
        output_tokens = usage_summary['overall_output_tokens']
        call_count = usage_summary['call_count']


        return final_solution['response'], input_tokens, output_tokens, call_count
                    