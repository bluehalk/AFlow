#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GSM8K消融研究脚本 - 200个测试样本 - 付费API版本
只测试OptimalWorkflow方法
适配付费API限制: RPM=100, RPD=50000, TPM=30000
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

class OptimalWorkflow:
    """最优工作流实现 - 基于round 10的结果 - 付费API优化版本"""
    
    def __init__(self, name: str, llm_config, dataset: DatasetType, max_retries: int = 3) -> None:
        self.name = name
        self.dataset = dataset
        self.llm_config = llm_config
        self.max_retries = max_retries
        self.llm = create_llm_instance(llm_config)
        self.custom = Custom(self.llm)
        self.sc_ensemble = ScEnsemble(self.llm)
        self.programmer = Programmer(self.llm)

    async def _call_with_simple_retry(self, func, *args, **kwargs):
        """简单重试辅助函数 - 付费API优化版本"""
        for attempt in range(self.max_retries):
            try:
                if attempt > 0:
                    await asyncio.sleep(1)  # 付费API重试延迟减少
                return await func(*args, **kwargs)
            except Exception as e:
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ['connection', 'timeout', 'network', 'rate limit']):
                    if attempt < self.max_retries - 1:
                        print(f"    ⚠️  API错误，重试中: {e}")
                        continue
                raise e

    async def __call__(self, problem: str):
        """最优工作流实现 - 付费API优化版本"""
        solutions = []
        
        # 生成3个解决方案 - 并发执行以提高速度
        print(f"  OptimalWorkflow: 并发生成3个解决方案...")
        tasks = []
        for i in range(3):
            task = self._call_with_simple_retry(
                self.custom, input=problem, instruction=OPTIMAL_MATH_SOLVE_PROMPT
            )
            tasks.append(task)
        
        # 并发执行，但有轻微延迟以避免瞬间过多请求
        try:
            # 使用gather但添加短延迟
            for i, task in enumerate(tasks):
                if i > 0:
                    await asyncio.sleep(0.3)  # 付费API可以更快
                solutions.append(await task)
            
            # 提取响应
            solution_texts = [sol['response'] for sol in solutions]
            
        except Exception as e:
            print(f"    ❌ 解决方案生成失败: {e}")
            # 降级到顺序执行
            solution_texts = []
            for i in range(3):
                try:
                    solution = await self._call_with_simple_retry(
                        self.custom, input=problem, instruction=OPTIMAL_MATH_SOLVE_PROMPT
                    )
                    solution_texts.append(solution['response'])
                except Exception as e:
                    print(f"    解决方案{i+1}生成失败: {e}")
                    solution_texts.append(f"Error: {e}")
                
                if i < 2:
                    await asyncio.sleep(0.5)
        
        # 使用自一致性集成
        print("  OptimalWorkflow: 进行自一致性集成...")
        valid_solutions = [s for s in solution_texts if not str(s).startswith("Error")]
        if not valid_solutions:
            print("    ❌ 没有有效解决方案进行集成")
            return "OptimalWorkflow执行失败", 0.0
            
        try:
            final_solution = await self._call_with_simple_retry(
                self.sc_ensemble, solutions=valid_solutions, problem=problem
            )
            print(f"    ✅ 集成成功: {final_solution['response'][:80]}...")
        except Exception as e:
            print(f"    ⚠️  自一致性集成失败，使用第一个有效解决方案: {e}")
            final_solution = {'response': valid_solutions[0]}
        
        # 编程器验证
        print("  OptimalWorkflow: 代码验证...")
        try:
            verification = await self._call_with_simple_retry(
                self.programmer, problem=problem, analysis=final_solution['response']
            )
            # 检查验证结果是否有效
            output = verification.get('output', '')
            if output and not str(output).startswith("Error") and str(output).strip():
                print(f"    ✅ 验证成功，使用代码结果: {output}")
                return str(output).strip(), self.llm.get_usage_summary()["total_cost"]
            else:
                print(f"    ⚠️  验证结果无效，使用自一致性结果: {output}")
        except Exception as e:
            print(f"    ❌ 验证失败，使用自一致性结果: {e}")
        
        return final_solution['response'], self.llm.get_usage_summary()["total_cost"]

class OptimalWorkflowExperiment:
    """OptimalWorkflow实验管理器 - 付费API版本"""
    
    def __init__(self, data_file: str, results_dir: str = "results", llm_config: str = "meta-llama/llama-3-70b-instruct"):
        self.data_file = data_file
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 配置模型
        self.models_config = LLMsConfig.default()
        self.llm_config = self.models_config.get(llm_config)
        
        # 创建benchmark
        self.benchmark = GSM8KBenchmark(
            name="GSM8K_OptimalWorkflow", 
            file_path="data/datasets/gsm8k_test.jsonl",
            log_path="workspace/GSM8K_optimal"
        )
        
        # 实验记录
        self.experiment_log = {
            "start_time": datetime.now().isoformat(),
            "config": {
                "model": self.llm_config.model,
                "data_file": data_file,
                "api_limits": "RPM=100, RPD=50000, TPM=30000 (付费API)",
                "total_samples": 200,
                "max_retries": 3,
                "optimization": "并发解决方案生成，减少延迟",
                "method": "OptimalWorkflow only"
            },
            "results": {},
            "failed_samples": []
        }
    
    async def load_test_data(self) -> List[Dict]:
        """加载测试数据"""
        print(f"📂 加载测试数据: {self.data_file}")
        
        data = []
        with open(self.data_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))
        
        print(f"📊 加载了 {len(data)} 个测试样本")
        if len(data) != 200:
            print(f"⚠️  警告: 期望200个样本，实际加载了{len(data)}个")
        return data
    
    async def run_optimal_workflow(self, test_data: List[Dict]) -> Dict:
        """运行OptimalWorkflow测试 - 付费API优化版本"""
        method_name = "OptimalWorkflow"
        print(f"\n🚀 开始测试方法: {method_name} (付费API优化版本)")
        print("=" * 60)
        
        # 创建方法实例
        method = OptimalWorkflow(
            name=f"GSM8K_{method_name}", 
            llm_config=self.llm_config, 
            dataset="GSM8K",
            max_retries=3
        )
        
        results = []
        failed_samples = []
        start_time = time.time()
        
        # 付费API的延迟配置 - 大幅度减少延迟
        # RPM=100意味着每0.6秒一个请求，但OptimalWorkflow内部有约5个请求
        # 所以每个样本大约需要3秒，为了安全起见设置4秒延迟
        request_delay = 4.0  # 相比免费版本的10秒大幅减少
        
        for i, sample in enumerate(test_data):
            sample_start_time = time.time()
            try:
                print(f"🔄 [OptimalWorkflow] 处理样本 {i+1}/{len(test_data)}: {sample['question'][:50]}...")
                
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
                    "correct": score > 0.5,
                    "processing_time": time.time() - sample_start_time
                }
                results.append(result)
                
                print(f"✅ [OptimalWorkflow] 样本 {i+1} - 得分: {score:.1f}, 成本: ${cost:.6f}, 用时: {result['processing_time']:.1f}s")
                
            except Exception as e:
                processing_time = time.time() - sample_start_time
                error_msg = str(e)
                
                print(f"❌ [OptimalWorkflow] 样本 {i+1} 最终失败: {error_msg}")
                
                failed_sample = {
                    "sample_id": i,
                    "question": sample['question'],
                    "error": error_msg,
                    "processing_time": processing_time
                }
                failed_samples.append(failed_sample)
                
                result = {
                    "sample_id": i,
                    "question": sample['question'],
                    "error": error_msg,
                    "score": 0.0,
                    "cost": 0.0,
                    "correct": False,
                    "processing_time": processing_time
                }
                results.append(result)
            
            # API限制控制 - 付费版本延迟大幅减少
            if i < len(test_data) - 1:
                await asyncio.sleep(request_delay)
            
            # 每10个样本显示进度 (频率提高)
            if (i + 1) % 10 == 0:
                successful_results = [r for r in results if 'error' not in r]
                if successful_results:
                    avg_score = sum(r['score'] for r in successful_results) / len(successful_results)
                    total_cost = sum(r['cost'] for r in results)
                    elapsed_time = time.time() - start_time
                    estimated_total_time = elapsed_time * len(test_data) / (i + 1)
                    remaining_time = estimated_total_time - elapsed_time
                    print(f"📈 [OptimalWorkflow] 进度: {i+1}/{len(test_data)}, 准确率: {avg_score:.2f}, 成本: ${total_cost:.3f}")
                    print(f"⏱️  已用时: {elapsed_time/60:.1f}分钟, 预计剩余: {remaining_time/60:.1f}分钟")
        
        # 计算最终统计
        total_time = time.time() - start_time
        successful_results = [r for r in results if 'error' not in r]
        
        if successful_results:
            scores = [r['score'] for r in successful_results]
            costs = [r['cost'] for r in results]
            correct_count = sum(1 for r in successful_results if r['correct'])
            
            method_stats = {
                "method_name": method_name,
                "total_samples": len(results),
                "successful_samples": len(successful_results),
                "failed_samples": len(failed_samples),
                "correct_samples": correct_count,
                "accuracy": correct_count / len(successful_results) if successful_results else 0,
                "overall_success_rate": len(successful_results) / len(results),
                "avg_score": statistics.mean(scores) if scores else 0,
                "score_std": statistics.stdev(scores) if len(scores) > 1 else 0,
                "total_cost": sum(costs),
                "avg_cost_per_sample": statistics.mean(costs) if costs else 0,
                "total_time_seconds": total_time,
                "total_time_minutes": total_time / 60,
                "avg_time_per_sample": total_time / len(results),
                "detailed_results": results
            }
        else:
            method_stats = {
                "method_name": method_name,
                "total_samples": len(results),
                "successful_samples": 0,
                "failed_samples": len(failed_samples),
                "correct_samples": 0,
                "accuracy": 0,
                "overall_success_rate": 0,
                "avg_score": 0,
                "score_std": 0,
                "total_cost": 0,
                "avg_cost_per_sample": 0,
                "total_time_seconds": total_time,
                "total_time_minutes": total_time / 60,
                "avg_time_per_sample": total_time / len(results),
                "detailed_results": results
            }
        
        # 记录失败的样本
        if failed_samples:
            self.experiment_log["failed_samples"] = failed_samples
        
        print(f"\n📊 [OptimalWorkflow] 最终结果:")
        print(f"   准确率: {method_stats['accuracy']:.1%} ({method_stats['correct_samples']}/200)")
        if method_stats['failed_samples'] > 0:
            print(f"   失败样本: {method_stats['failed_samples']}")
        print(f"   总成本: ${method_stats['total_cost']:.4f}")
        print(f"   用时: {method_stats['total_time_minutes']:.1f}分钟")
        
        return method_stats
    
    async def run_experiment(self):
        """运行OptimalWorkflow实验"""
        print("🎯 开始OptimalWorkflow实验 (200个样本, 付费API)")
        print("=" * 60)
        print(f"✅ 模型配置: {self.llm_config.model}")
        print(f"🔧 API限制: RPM=100, RPD=50000, TPM=30000 (付费API)")
        print(f"🚀 优化: 并发解决方案生成，延迟减少60%")
        
        # 加载测试数据
        test_data = await self.load_test_data()
        
        # 运行OptimalWorkflow
        method_stats = await self.run_optimal_workflow(test_data)
        self.experiment_log["results"] = method_stats
        
        # 保存结果
        await self.save_results()
        
        print(f"\n🎉 OptimalWorkflow实验完成! 结果保存在: {self.results_dir}")
    
    async def save_results(self):
        """保存实验结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存详细结果到CSV
        df = pd.DataFrame(self.experiment_log["results"]["detailed_results"])
        csv_file = self.results_dir / f"OptimalWorkflow_paid_{timestamp}.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        # 保存完整实验日志
        log_file = self.results_dir / f"OptimalWorkflow_experiment_paid_{timestamp}.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(self.experiment_log, f, ensure_ascii=False, indent=2)
        
        # 如果有失败的样本，单独保存
        if self.experiment_log["failed_samples"]:
            failed_samples_file = self.results_dir / f"OptimalWorkflow_failed_samples_paid_{timestamp}.json"
            with open(failed_samples_file, 'w', encoding='utf-8') as f:
                json.dump(self.experiment_log["failed_samples"], f, ensure_ascii=False, indent=2)
        
        print(f"💾 结果已保存到 {self.results_dir}")
        print(f"📁 详细结果: OptimalWorkflow_paid_{timestamp}.csv")


async def main():
    """主函数"""
    print("🎯 GSM8K OptimalWorkflow实验 - 200个样本 (付费API)")
    print("=" * 60)
    
    # 创建实验
    experiment = OptimalWorkflowExperiment(
        data_file="z_ablation/200_gsm8k.jsonl",
        results_dir="z_ablation/results",
        llm_config="meta-llama/llama-3-70b-instruct"  # 付费API配置
    )
    
    # 运行实验
    await experiment.run_experiment()

if __name__ == "__main__":
    asyncio.run(main())