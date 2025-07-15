from typing import Literal
from scripts.operators import *
from . import prompt_2
from . import prompt_final
from . import prompt_final_improved
from . import prompt_final_v2
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
        custom_code_generate = CustomCodeGenerate(llm)
        custom = Custom(llm)
        # 重置计数器
        llm.usage_tracker.overall_input_tokens = 0
        llm.usage_tracker.overall_output_tokens = 0
        llm.usage_tracker.call_count = 0
        llm.usage_tracker.usage_history = []

        solution = await custom_code_generate(problem=problem, entry_point=entry_point, instruction=prompt_final_v2.FINAL_OPTIMIZED_PROMPT_V2)
        # solution = await custom(problem, entry_point)

        # 获取这个样本的token统计
        usage_summary = llm.usage_tracker.get_summary()
        input_tokens = usage_summary['overall_input_tokens']
        output_tokens = usage_summary['overall_output_tokens']
        call_count = usage_summary['call_count']
            
        return solution['code'], input_tokens, output_tokens, call_count