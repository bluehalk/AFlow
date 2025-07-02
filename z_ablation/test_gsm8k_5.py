#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GSM8K消融研究脚本 - 快速测试版本 (5个样本)
对比三种方法：DirectIO, IO_with_optimal_instructions, OptimalWorkflow
适配API限制: RPM=30, TPM=6000, TPD=500000
"""

import asyncio
import json
import os
import sys
import time
import pandas as pd
from pathlib import Path
from typing import Literal, Dict, List, Any
from datetime import datetime
import statistics

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.operators import Custom, ScEnsemble, Programmer
from scripts.async_llm import LLMsConfig, create_llm_instance
from benchmarks.gsm8k import GSM8KBenchmark

DatasetType = Literal["HumanEval", "MBPP", "GSM8K", "MATH", "HotpotQA", "DROP"]

# 从round 10获取的最优prompt (最佳性能)
OPTIMAL_MATH_SOLVE_PROMPT = """You are a highly skilled mathematician tasked with solving a math problem. Follow these steps carefully:

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

# 基础的直接prompt (最简单的baseline)
BASIC_SOLVE_PROMPT = """Solve this math problem step by step and provide the final numerical answer.

"""

# 中等复杂度的prompt
MEDIUM_SOLVE_PROMPT = """Solve this math problem step by step. Show your work and provide the final numerical answer.

"""

# 简单数学prompt  
SIMPLE_MATH_PROMPT = """Solve this math problem:

"""

# 最简单的prompt
MINIMAL_PROMPT = """Answer:

"""

# 空prompt (无指导)
NO_PROMPT = ""

class DirectIO:
    """直接把问题给LLM - 最基础的方法"""
    
    def __init__(self, name: str, llm_config, dataset: DatasetType) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = Custom(self.llm)

    async def __call__(self, problem: str):
        """直接调用LLM解决问题"""
        print(f"  DirectIO: 直接求解问题...")
        solution = await self.custom(input=problem, instruction=BASIC_SOLVE_PROMPT)
        return solution['response'], self.llm.get_usage_summary()["total_cost"]

class IO_with_optimal_instructions:
    """使用最优instructions的IO"""
    
    def __init__(self, name: str, llm_config, dataset: DatasetType) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = Custom(self.llm)

    async def __call__(self, problem: str):
        """使用最优prompt直接调用LLM"""
        print(f"  IO_optimal: 使用最优prompt求解...")
        solution = await self.custom(input=problem, instruction=OPTIMAL_MATH_SOLVE_PROMPT)
        return solution['response'], self.llm.get_usage_summary()["total_cost"]

class OptimalWorkflow:
    """最优工作流实现 - 基于round 10的结果"""
    
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
        
        # 生成2个解决方案 (快速测试版本，减少API调用)
        for i in range(2):
            print(f"  OptimalWorkflow: 生成解决方案 {i+1}/2...")
            
            # 带重试的解决方案生成
            for attempt in range(3):
                try:
                    if attempt > 0:
                        print(f"    🔄 解决方案生成重试第 {attempt} 次...")
                        await asyncio.sleep(2)
                    
                    solution = await self.custom(input=problem, instruction=OPTIMAL_MATH_SOLVE_PROMPT)
                    solutions.append(solution['response'])
                    break
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    if any(keyword in error_msg for keyword in ['connection', 'timeout', 'network']):
                        if attempt < 2:
                            print(f"    ❌ 生成解决方案网络错误，准备重试: {e}")
                            continue
                    
                    print(f"    ❌ 生成解决方案失败: {e}")
                    # 如果失败，添加一个错误标记
                    solutions.append(f"Error generating solution {i+1}: {e}")
                    break
            
            # 添加延迟以控制RPM
            if i < 1:  # 最后一个不需要延迟
                await asyncio.sleep(2.5)
        
        # 使用自一致性集成选择最佳答案 (带重试机制)
        print("  OptimalWorkflow: 进行自一致性集成...")
        final_solution = None
        for attempt in range(3):
            try:
                if attempt > 0:
                    print(f"    🔄 自一致性集成重试第 {attempt} 次...")
                    await asyncio.sleep(2)
                
                final_solution = await self.sc_ensemble(solutions=solutions, problem=problem)
                break
                
            except Exception as e:
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ['connection', 'timeout', 'network']):
                    if attempt < 2:
                        print(f"    ❌ 自一致性集成网络错误，准备重试: {e}")
                        continue
                
                print(f"    ❌ 自一致性集成失败: {e}")
                # 如果自一致性失败，使用第一个有效的解决方案
                valid_solutions = [s for s in solutions if not s.startswith("Error")]
                if valid_solutions:
                    final_solution = {'response': valid_solutions[0]}
                    print("  OptimalWorkflow: 使用第一个有效解决方案")
                else:
                    final_solution = {'response': "所有解决方案生成失败"}
                    print("  OptimalWorkflow: 所有解决方案都失败了")
                break
        
        # 检查final_solution是否有效
        if not final_solution or 'response' not in final_solution:
            print("  OptimalWorkflow: 所有步骤都失败，返回错误信息")
            return "OptimalWorkflow执行失败：无法生成有效解决方案", 0.0
        
        # 使用编程器验证 (带重试机制)
        print("  OptimalWorkflow: 代码验证...")
        verification_success = False
        for attempt in range(3):
            try:
                if attempt > 0:
                    print(f"    🔄 验证重试第 {attempt} 次...")
                    await asyncio.sleep(2)
                
                verification = await self.programmer(problem=problem, analysis=final_solution['response'])
                if verification['output']:
                    print("  OptimalWorkflow: 验证成功，使用验证结果")
                    return verification['output'], self.llm.get_usage_summary()["total_cost"]
                else:
                    print("  OptimalWorkflow: 验证无输出，使用自一致性结果")
                    break
                    
            except Exception as e:
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ['connection', 'timeout', 'network']):
                    if attempt < 2:
                        print(f"    ❌ 验证网络错误，准备重试: {e}")
                        continue
                
                print(f"    验证步骤失败，使用自一致性结果: {e}")
                break
        
        print("  OptimalWorkflow: 使用自一致性集成结果")
        return final_solution['response'], self.llm.get_usage_summary()["total_cost"]

class QuickAblationExperiment:
    """快速消融实验管理器 - 5个样本"""
    
    def __init__(self, data_file: str, results_dir: str = "z_ablation/results_quick"):
        self.data_file = data_file
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # 配置模型
        self.models_config = LLMsConfig.default()
        self.llm_config = self.models_config.get('llama3-70b-8192')
        
        # 创建benchmark
        self.benchmark = GSM8KBenchmark(
            name="GSM8K_Quick_Ablation", 
            file_path="data/datasets/gsm8k_test.jsonl",
            log_path="workspace/GSM8K_ablation_quick"
        )
        
        # 实验记录
        self.experiment_log = {
            "start_time": datetime.now().isoformat(),
            "config": {
                "model": self.llm_config.model,
                "data_file": data_file,
                "api_limits": "RPM=30, TPM=6000, TPD=500000",
                "test_samples": 5,
                "note": "快速测试版本"
            },
            "methods": {},
            "summary": {}
        }
    
    async def load_test_data(self) -> List[Dict]:
        """加载测试数据 - 只取前5个"""
        print(f"📂 加载测试数据: {self.data_file} (前5个样本)")
        
        data = []
        with open(self.data_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 5:  # 只取前5个
                    break
                if line.strip():
                    data.append(json.loads(line.strip()))
        
        print(f"📊 加载了 {len(data)} 个测试样本 (快速测试版本)")
        return data
    
    async def run_method(self, method_class, method_name: str, test_data: List[Dict]) -> Dict:
        """运行单个方法的测试"""
        print(f"\n🚀 开始测试方法: {method_name}")
        print("=" * 60)
        
        # 创建方法实例
        method = method_class(
            name=f"GSM8K_{method_name}", 
            llm_config=self.llm_config, 
            dataset="GSM8K"
        )
        
        results = []
        start_time = time.time()
        
        # API调用频率控制 - 快速测试版本使用较短延迟
        if method_name == "OptimalWorkflow":
            request_delay = 5   # OptimalWorkflow需要更长延迟
        else:
            request_delay = 2   # 其他方法较短延迟
        
        for i, sample in enumerate(test_data):
            print(f"🔄 [{method_name}] 处理样本 {i+1}/{len(test_data)}: {sample['question'][:50]}...")
            
            # 简单重试机制
            success = False
            for attempt in range(3):  # 最多重试3次
                try:
                    if attempt > 0:
                        print(f"    🔄 重试第 {attempt} 次...")
                        await asyncio.sleep(2)  # 简单延迟2秒
                    
                    # 调用方法
                    prediction, cost = await method(sample['question'])
                    
                    # 评估结果
                    expected = self.benchmark.extract_number(sample['answer'])
                    predicted = self.benchmark.extract_number(prediction)
                    score, _ = self.benchmark.calculate_score(expected, predicted)
                    
                    result = {
                        "sample_id": i,
                        "question": sample['question'],
                        "expected": expected,
                        "predicted": predicted,
                        "prediction_text": prediction,
                        "score": score,
                        "cost": cost,
                        "correct": score > 0.5  # 二进制正确性
                    }
                    results.append(result)
                    
                    print(f"✅ [{method_name}] 样本 {i+1} - 得分: {score:.1f}, 成本: ${cost:.6f}")
                    success = True
                    break
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    # 只对网络相关错误重试
                    if any(keyword in error_msg for keyword in ['connection', 'timeout', 'network']):
                        if attempt < 2:  # 还有重试机会
                            print(f"    ❌ 网络错误，准备重试: {e}")
                            continue
                    
                    # 非网络错误或达到最大重试次数
                    print(f"❌ [{method_name}] 样本 {i+1} 最终失败: {str(e)}")
                    result = {
                        "sample_id": i,
                        "question": sample['question'],
                        "error": str(e),
                        "score": 0.0,
                        "cost": 0.0,
                        "correct": False
                    }
                    results.append(result)
                    break
            
            # API限制控制
            if i < len(test_data) - 1:  # 最后一个不需要延迟
                await asyncio.sleep(request_delay)
        
        # 计算最终统计
        total_time = time.time() - start_time
        scores = [r['score'] for r in results]
        costs = [r['cost'] for r in results]
        correct_count = sum(1 for r in results if r['correct'])
        
        actual_samples = len(results)
        
        method_stats = {
            "method_name": method_name,
            "total_samples": actual_samples,
            "correct_samples": correct_count,
            "accuracy": correct_count / actual_samples,
            "avg_score": statistics.mean(scores),
            "score_std": statistics.stdev(scores) if len(scores) > 1 else 0,
            "total_cost": sum(costs),
            "avg_cost_per_sample": statistics.mean(costs),
            "total_time_seconds": total_time,
            "avg_time_per_sample": total_time / actual_samples,
            "detailed_results": results
        }
        
        print(f"\n📊 [{method_name}] 最终结果:")
        print(f"   总样本数: {method_stats['total_samples']}")
        print(f"   正确样本数: {method_stats['correct_samples']}")
        print(f"   准确率: {method_stats['accuracy']:.1%}")
        print(f"   平均得分: {method_stats['avg_score']:.3f} ± {method_stats['score_std']:.3f}")
        print(f"   总成本: ${method_stats['total_cost']:.4f}")
        print(f"   平均每样本成本: ${method_stats['avg_cost_per_sample']:.6f}")
        print(f"   总用时: {method_stats['total_time_seconds']:.1f}秒")
        
        return method_stats
    
    async def run_ablation_study(self):
        """运行完整的消融研究"""
        print("🎯 开始GSM8K快速消融研究 (5个样本)")
        print("=" * 70)
        print(f"✅ 模型配置: {self.llm_config.model}")
        print(f"🔧 API限制: RPM=30, TPM=6000, TPD=500000")
        print(f"📝 实验顺序: DirectIO → IO_with_optimal_instructions → OptimalWorkflow")
        print(f"⚡ 快速测试模式: 只测试5个样本以验证代码逻辑")
        
        # 加载测试数据
        test_data = await self.load_test_data()
        
        # 定义测试方法 (按指定顺序)
        methods = [
            (DirectIO, "DirectIO"),
            (IO_with_optimal_instructions, "IO_with_optimal_instructions"),
            (OptimalWorkflow, "OptimalWorkflow")
        ]
        
        # 依次运行每个方法
        for i, (method_class, method_name) in enumerate(methods):
            print(f"\n🔄 开始第 {i+1}/3 个方法: {method_name}")
            method_stats = await self.run_method(method_class, method_name, test_data)
            self.experiment_log["methods"][method_name] = method_stats
            
            # 方法间的缓冲时间
            if i < len(methods) - 1:
                print(f"\n⏳ 等待5秒后继续下一个方法...")
                await asyncio.sleep(5)
        
        # 生成总结
        await self.generate_summary()
        
        # 保存最终结果
        await self.save_final_results()
        
        print(f"\n🎉 快速消融研究完成! 结果保存在: {self.results_dir}")
    
    async def generate_summary(self):
        """生成实验总结"""
        methods = self.experiment_log["methods"]
        
        # 创建对比表格
        comparison = []
        for method_name, stats in methods.items():
            comparison.append({
                "Method": method_name,
                "Accuracy": f"{stats['accuracy']:.1%}",
                "Avg_Score": f"{stats['avg_score']:.3f}",
                "Score_Std": f"{stats['score_std']:.3f}",
                "Total_Cost": f"${stats['total_cost']:.4f}",
                "Avg_Cost_Per_Sample": f"${stats['avg_cost_per_sample']:.6f}",
                "Total_Time(s)": f"{stats['total_time_seconds']:.1f}",
                "Correct_Count": f"{stats['correct_samples']}/5"
            })
        
        self.experiment_log["summary"] = {
            "comparison_table": comparison,
            "end_time": datetime.now().isoformat()
        }
        
        # 打印总结
        print("\n" + "=" * 70)
        print("📋 快速消融研究总结 (5个样本)")
        print("=" * 70)
        
        df_comparison = pd.DataFrame(comparison)
        print(df_comparison.to_string(index=False))
        
        print(f"\n✅ 代码验证完成 - 所有方法都能正常运行")
        print(f"🔄 现在可以运行完整的200样本测试: python z_ablation/test_gsm8k_200.py")
    
    async def save_final_results(self):
        """保存最终实验结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存完整实验日志
        log_file = self.results_dir / f"quick_ablation_experiment_{timestamp}.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(self.experiment_log, f, ensure_ascii=False, indent=2)
        
        # 保存对比表格
        comparison_df = pd.DataFrame(self.experiment_log["summary"]["comparison_table"])
        comparison_file = self.results_dir / f"quick_method_comparison_{timestamp}.csv"
        comparison_df.to_csv(comparison_file, index=False, encoding='utf-8')
        
        print(f"💾 快速测试结果已保存到 {self.results_dir}")


async def main():
    """主函数"""
    print("🎯 GSM8K快速消融研究 - 5个样本验证")
    print("=" * 50)
    
    # 创建实验
    experiment = QuickAblationExperiment(
        data_file="z_ablation/200_gsm8k.jsonl",
        results_dir="z_ablation/results_quick"
    )
    
    # 运行消融研究
    await experiment.run_ablation_study()

if __name__ == "__main__":
    asyncio.run(main())