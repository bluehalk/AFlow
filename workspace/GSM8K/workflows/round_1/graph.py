from typing import Literal
import workspace.GSM8K.workflows.template.operator as operator
import workspace.GSM8K.workflows.round_1.prompt as prompt_custom
from scripts.async_llm import create_llm_instance

DatasetType = Literal["HumanEval", "MBPP", "GSM8K", "MATH", "HotpotQA", "DROP"]

class Workflow:
    def __init__(
        self,
        name: str, # NOTE(sjh): name = "GSM8K"
        llm_config, # NOTE(sjh): llm_config = {"model": "gpt-4o-mini", "temperature": 0.0, "max_tokens": 1000}
        dataset: DatasetType, # NOTE(sjh): dataset = "GSM8K"
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        solution = await self.custom(input=problem, instruction="")
        return solution['response'], self.llm.get_usage_summary()["total_cost"]