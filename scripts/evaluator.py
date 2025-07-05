# -*- coding: utf-8 -*-
# @Date    : 8/23/2024 10:00 AM
# @Author  : all
# @Desc    : Evaluation for different datasets

from typing import Dict, Literal, Tuple, List, Optional

from benchmarks.benchmark import BaseBenchmark
from benchmarks.drop import DROPBenchmark
from benchmarks.gsm8k import GSM8KBenchmark
from benchmarks.hotpotqa import HotpotQABenchmark
from benchmarks.humaneval import HumanEvalBenchmark
from benchmarks.math import MATHBenchmark
from benchmarks.mbpp import MBPPBenchmark

# If you want to customize tasks, add task types here and provide evaluation functions, just like the ones given above
DatasetType = Literal["HumanEval", "MBPP", "GSM8K", "MATH", "HotpotQA", "DROP"]


class Evaluator:
    """
    Complete the evaluation for different datasets here
    """

    def __init__(self, eval_path: str):
        self.eval_path = eval_path
        self.dataset_configs: Dict[DatasetType, BaseBenchmark] = {
            "GSM8K": GSM8KBenchmark,
            "MATH": MATHBenchmark,
            "HumanEval": HumanEvalBenchmark,
            "HotpotQA": HotpotQABenchmark,
            "MBPP": MBPPBenchmark,
            "DROP": DROPBenchmark,
        }

    async def graph_evaluate(
        self,
        dataset: DatasetType,
        graph,
        params: dict,
        path: str,
        is_test: bool = False,
        sample_indices: Optional[List[int]] = None,
    ) -> Tuple[float, float, float]:
        if dataset not in self.dataset_configs:
            raise ValueError(f"Unsupported dataset: {dataset}")

        #NOTE(sjh): 这里根据是否是test指定了数据集的文件路径
        data_path = self._get_data_path(dataset, is_test)

        benchmark_class = self.dataset_configs[dataset]
        benchmark = benchmark_class(name=dataset, file_path=data_path, log_path=path)

        # Use params to configure the graph and benchmark
        # NOTE(sjh)返回一个实例化后的Workflow对象
        configured_graph = await self._configure_graph(dataset, graph, params)
        # logger.info(f"configured_graph: {type(configured_graph)}: {configured_graph}")

        # Determine which sample indices to evaluate
        if sample_indices is not None:
            va_list = list(sample_indices)
        else:
            if is_test:
                va_list = None  # 默认测试集样本数量
            else:
                va_list = list(range(10))  # 默认验证集前10个样本
        return await benchmark.run_evaluation(configured_graph, va_list)

    async def _configure_graph(self, dataset, graph, params: dict):
        # Here you can configure the graph based on params
        # For example: set LLM configuration, dataset configuration, etc.
        dataset_config = params.get("dataset", {})
        llm_config = params.get("llm_config", {})
        return graph(name=dataset, llm_config=llm_config, dataset=dataset_config)

    def _get_data_path(self, dataset: DatasetType, test: bool) -> str:
        base_path = f"data/datasets/{dataset.lower()}"
        return f"{base_path}_test.jsonl" if test else f"{base_path}_validate.jsonl"
