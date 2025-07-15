from typing import Literal
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.operators import Custom, AnswerGenerate, ScEnsemble
from scripts.async_llm import create_llm_instance
import z_ablation.results.DROP.graphs.round_3.prompt as prompt_custom

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
        answer_generate = AnswerGenerate(llm)
        custom = Custom(llm)
        # 重置计数器
        llm.usage_tracker.overall_input_tokens = 0
        llm.usage_tracker.overall_output_tokens = 0
        llm.usage_tracker.call_count = 0
        llm.usage_tracker.usage_history = []

        step_by_step = await answer_generate(input=problem)
        final_solution = await custom(input=f"Question: {problem}\nBest solution: {step_by_step['answer']}", instruction=prompt_custom.REFINE_ANSWER_PROMPT)

        usage_summary = llm.usage_tracker.get_summary() 
        input_tokens = usage_summary['overall_input_tokens']
        output_tokens = usage_summary['overall_output_tokens']
        call_count = usage_summary['call_count']

        return step_by_step['answer'], input_tokens, output_tokens, call_count