#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GSM8K测试脚本 - 使用一半测试数据
适配API限制: RPM=30, TPM=6000
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Literal

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.operators import Custom, ScEnsemble, Programmer
from scripts.async_llm import LLMsConfig, create_llm_instance
from benchmarks.gsm8k import GSM8KBenchmark

DatasetType = Literal["HumanEval", "MBPP", "GSM8K", "MATH", "HotpotQA", "DROP"]

# 最优工作流的prompt
MATH_SOLVE_PROMPT = """
You are a highly skilled mathematician tasked with solving a math problem. Follow these steps carefully:

1. Read and understand the problem thoroughly.
2. Identify all key information, variables, and relationships.
3. Determine the appropriate mathematical concepts, formulas, or equations to use.
4. Solve the problem step-by-step, showing all your work clearly.
5. Double-check your calculations and reasoning at each step.
6. Provide a clear and concise final answer.
7. Verify your solution by plugging it back into the original problem or using an alternative method if possible.

Format your answer as follows:
- Use LaTeX notation for mathematical expressions where appropriate.
- Show each step of your solution process clearly.
- Clearly state your final answer at the end of your solution.
- Express numerical answers as precise values (avoid rounding unless specified).
- Ensure that your final answer is a single numerical value without any units or additional text.
- Do not include any explanatory text with your final answer, just the number itself.

For example, if the final answer is 42.5, your response should end with just:
42.5

Here's the problem to solve:

"""

class OptimalWorkflow:
    """最优的GSM8K工作流 (准确率93.744%)"""
    
    def __init__(self, name: str, llm_config, dataset: DatasetType) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = Custom(self.llm)
        self.sc_ensemble = ScEnsemble(self.llm)
        self.programmer = Programmer(self.llm)

    async def __call__(self, problem: str):
        """最优工作流实现"""
        solutions = []
        
        # 生成5个解决方案 (但为了控制API使用，我们减少到3个)
        for i in range(3):  # 减少数量以控制API使用
            print(f"  生成解决方案 {i+1}/3...")
            solution = await self.custom(input=problem, instruction=MATH_SOLVE_PROMPT)
            solutions.append(solution['response'])
            
            # 添加延迟以控制RPM
            if i < 2:  # 最后一个不需要延迟
                await asyncio.sleep(2)  # 2秒延迟
        
        # 使用自一致性集成选择最佳答案
        print("  进行自一致性集成...")
        final_solution = await self.sc_ensemble(solutions=solutions, problem=problem)
        
        # 使用编程器验证 (可选，为了节省API调用可以注释掉)
        # print("  代码验证...")
        # verification = await self.programmer(problem=problem, analysis=final_solution['response'])
        # 
        # if verification['output']:
        #     return verification['output'], self.llm.get_usage_summary()["total_cost"]
        
        return final_solution['response'], self.llm.get_usage_summary()["total_cost"]

async def test_gsm8k_half():
    """测试GSM8K的一半数据"""
    print("🚀 开始GSM8K测试 (一半测试数据)")
    print("=" * 50)
    
    # 配置模型
    models_config = LLMsConfig.default()
    llm_config = models_config.get('llama3-70b-8192')
    
    print(f"✅ 模型配置: {llm_config.model}")
    print(f"🔧 API限制: RPM=30, TPM=6000")
    
    # 创建工作流
    workflow = OptimalWorkflow(
        name="GSM8K_optimal", 
        llm_config=llm_config, 
        dataset="GSM8K"
    )
    
    # 创建benchmark
    benchmark = GSM8KBenchmark(
        name="GSM8K", 
        file_path="data/datasets/gsm8k_test.jsonl",  # 使用测试集
        log_path="workspace/GSM8K_test"
    )
    
    # 加载测试数据
    print("📂 加载GSM8K测试数据...")
    all_data = await benchmark.load_data()  # 加载全部测试数据
    total_samples = len(all_data)
    half_samples = total_samples // 2
    
    print(f"📊 测试集总样本数: {total_samples}")
    print(f"🎯 使用样本数: {half_samples} (一半)")
    
    # 使用前一半数据
    test_data = all_data[:half_samples]
    
    # 控制并发数 (考虑API限制)
    max_concurrent = 3  # 很低的并发数
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_sample(i, sample):
        async with semaphore:
            try:
                print(f"🔄 处理样本 {i+1}/{half_samples}: {sample['question'][:50]}...")
                
                # 调用工作流
                prediction, cost = await workflow(sample['question'])
                
                # 评估结果
                expected = benchmark.extract_number(sample['answer'])
                predicted = benchmark.extract_number(prediction)
                score, _ = benchmark.calculate_score(expected, predicted)
                
                print(f"✅ 样本 {i+1} - 得分: {score:.1f}, 成本: ${cost:.6f}")
                
                # 添加延迟以控制RPM (30 RPM = 每2秒1个请求)
                await asyncio.sleep(3)  # 3秒延迟确保不超过RPM限制
                
                return {
                    "sample_id": i,
                    "question": sample['question'],
                    "expected": expected,
                    "predicted": predicted,
                    "prediction_text": prediction,
                    "score": score,
                    "cost": cost
                }
                
            except Exception as e:
                print(f"❌ 样本 {i+1} 处理失败: {str(e)}")
                return {
                    "sample_id": i,
                    "question": sample['question'],
                    "error": str(e),
                    "score": 0.0,
                    "cost": 0.0
                }
    
    # 串行处理以严格控制API使用
    print(f"\n🔄 开始处理 {half_samples} 个样本 (串行处理)...")
    start_time = time.time()
    results = []
    
    for i, sample in enumerate(test_data):
        result = await process_sample(i, sample)
        results.append(result)
        
        # 每10个样本显示进度
        if (i + 1) % 10 == 0:
            avg_score = sum(r.get('score', 0) for r in results) / len(results)
            total_cost = sum(r.get('cost', 0) for r in results)
            elapsed = time.time() - start_time
            print(f"📈 进度: {i+1}/{half_samples}, 平均准确率: {avg_score:.3f}, 总成本: ${total_cost:.4f}, 用时: {elapsed:.1f}s")
    
    # 计算最终结果
    final_scores = [r.get('score', 0) for r in results]
    final_costs = [r.get('cost', 0) for r in results]
    
    avg_score = sum(final_scores) / len(final_scores)
    total_cost = sum(final_costs)
    total_time = time.time() - start_time
    
    print("\n" + "=" * 50)
    print("🎉 GSM8K测试完成!")
    print(f"📊 样本数: {half_samples}")
    print(f"🎯 平均准确率: {avg_score:.4f} ({avg_score*100:.2f}%)")
    print(f"💰 总成本: ${total_cost:.6f}")
    print(f"⏱️  总用时: {total_time:.1f}秒")
    print(f"🔥 与论文结果对比: 论文93.74% vs 当前{avg_score*100:.2f}%")
    
    # 保存结果
    output_dir = Path("workspace/GSM8K_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result_file = output_dir / f"test_results_{avg_score:.4f}.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": {
                "total_samples": half_samples,
                "average_score": avg_score,
                "total_cost": total_cost,
                "total_time": total_time,
                "paper_score": 0.9374
            },
            "detailed_results": results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"📁 详细结果已保存到: {result_file}")

if __name__ == "__main__":
    asyncio.run(test_gsm8k_half()) 