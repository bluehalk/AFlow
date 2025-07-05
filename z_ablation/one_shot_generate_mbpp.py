#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MBPP数据集 Graph Workflow 测试脚本
基于test_mbpp_graph.py的成功实现，完全复制graph.py的workflow逻辑
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

import scripts.operators as operator
from scripts.async_llm import LLMsConfig, create_llm_instance
from benchmarks.mbpp import MBPPBenchmark
import concurrent.futures

DatasetType = Literal["HumanEval", "MBPP", "GSM8K", "MATH", "HotpotQA", "DROP"]

# 使用和graph.py相同的prompt (从原始prompt.py复制)
CODE_GENERATE_PROMPT = """
Generate a Python function to solve the given problem. Ensure the function name matches the one specified in the problem. Include necessary imports. Use clear variable names and add comments for clarity.

Problem:
"""

FIX_CODE_PROMPT = """
You are a Python expert. Your task is to fix the failed solution based on the provided error information.

Please analyze the error and provide a corrected solution:
"""

# MBPP自一致性集成prompt
MBPP_SC_ENSEMBLE_PROMPT = """
Given the programming problem: {question}

Several Python solutions have been generated to address this problem:
{solutions}

Carefully evaluate these solutions and identify the code that is most likely to be correct. 
Consider factors like:
- Code correctness and logic
- Handling of edge cases
- Code quality and efficiency
- Adherence to the problem requirements

In the "thought" field, provide a detailed explanation of your evaluation process.
In the "solution_letter" field, output only the single letter ID (A, B, C, etc.) corresponding to the best solution.
Do not include any additional text or explanation in the "solution_letter" field.
"""



# MBPP综合workflow prompt（类似GSM8K的OVERALL_WORKFLOW_PROMPT）
MBPP_OVERALL_WORKFLOW_PROMPT = """
You are an expert Python programmer with advanced problem-solving capabilities. 
You will solve a programming problem using a comprehensive multi-approach workflow. Follow these steps carefully:

## STEP 1: GENERATE MULTIPLE SOLUTIONS
First, generate THREE different Python solutions to the problem using different approaches:

**Solution A (Direct Implementation):**
- Read and understand the problem requirements thoroughly
- Identify the core algorithm or logic needed
- Implement a straightforward solution
- Ensure proper handling of input/output format
- Add comments explaining the logic

**Solution B (Alternative Approach):**
- Consider a different algorithm or implementation strategy
- Use alternative data structures or methods if applicable
- Focus on efficiency or different edge case handling
- Provide clear code comments

**Solution C (Optimized/Robust Approach):**
- Implement with focus on optimization or robustness
- Consider performance improvements or comprehensive error handling
- Use advanced Python features if beneficial
- Ensure code quality and readability

## STEP 2: SELF-CONSISTENCY EVALUATION
Carefully evaluate the three solutions above:
- Compare the logic and implementation quality of each solution
- Identify which solution best addresses the problem requirements
- Check for potential bugs or edge case issues
- Consider code efficiency and readability
- Select the most reliable and well-implemented solution

## STEP 3: CODE VERIFICATION
Test your selected solution mentally or with examples:
- Walk through the code with sample inputs
- Verify the logic handles edge cases properly
- Ensure the function signature matches requirements
- Check that the return type and format are correct

## STEP 4: FINAL SOLUTION
Provide your final, best solution:
- Use the most reliable solution from your analysis
- Ensure it's complete and ready to run
- Include proper function signature as specified
- Add brief comments if helpful

## FORMAT YOUR RESPONSE AS:

### Solution A:
```python
# Your first solution here
```

### Solution B:
```python
# Your second solution here
```

### Solution C:
```python
# Your third solution here
```

### Self-Consistency Evaluation:
[Compare the three solutions and select the best one]

### Final Solution:
```python
# Your final, best solution here
```

Now solve this programming problem:

"""

class TokenTracker:
    """Token 使用统计器"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.calls = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        self.total_cost = 0.0

    def add_call(self, operation: str, input_tokens: int, output_tokens: int, cost: float):
        self.calls.append({
            "operation": operation,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost": cost,
        })
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_tokens += input_tokens + output_tokens
        self.total_cost += cost

    def get_summary(self):
        return {
            "total_calls": len(self.calls),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "calls_detail": self.calls,
        }

class Vanilla_Workflow:
    """完全复制graph.py的Workflow类"""
    def __init__(
        self,
        name: str,
        llm_config,
        dataset: DatasetType,
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)
        self.custom_code_generate = operator.CustomCodeGenerate(self.llm)
        self.test = operator.Test(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)
        
        # 添加token追踪器
        self.token_tracker = TokenTracker()

    def _get_tokens_and_cost(self):
        """获取当前的tokens和cost"""
        if hasattr(self.llm, 'usage_tracker'):
            return (
                self.llm.usage_tracker.total_input_tokens,
                self.llm.usage_tracker.total_output_tokens, 
                self.llm.usage_tracker.total_cost
            )
        else:
            return 0, 0, 0.0
    
    def _track_operation(self, operation: str, initial_input: int, initial_output: int, initial_cost: float):
        """追踪一次操作的token使用"""
        current_input, current_output, current_cost = self._get_tokens_and_cost()
        delta_input = current_input - initial_input
        delta_output = current_output - initial_output
        delta_cost = current_cost - initial_cost
        
        self.token_tracker.add_call(operation, delta_input, delta_output, delta_cost)
        return delta_input, delta_output, delta_cost

    async def __call__(self, problem: str, entry_point: str):
        """
        Implementation of the graph - 完全复制原始graph.py
        Custom operator to generate anything you want.
        But when you want to get standard code, you should use custom_code_generate operator.
        """
        print(f"    🔄 开始Graph workflow: {entry_point}")
        
        # 重置token追踪器
        self.token_tracker.reset()
        
        # 第1-3次调用: Generate 3 solutions
        print(f"    📝 生成3个解决方案...")
        solutions = [] 
        for i in range(3):
            initial_input, initial_output, initial_cost = self._get_tokens_and_cost()
            
            solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction=CODE_GENERATE_PROMPT)
            # CustomCodeGenerate returns {'code': '...'}, not {'response': '...'}
            solutions.append(solution['code'])
            
            # 追踪此次调用
            self._track_operation(f"CustomCodeGenerate_{i+1}", initial_input, initial_output, initial_cost)
            print(f"      ✅ 解决方案 {i+1} 生成完成")
        
        # 第4次调用: Self-consistency ensemble
        print(f"    🔄 自一致性集成...")
        initial_input, initial_output, initial_cost = self._get_tokens_and_cost()
        
        best_solution = await self.sc_ensemble(solutions=solutions, problem=problem)
        
        # 追踪ScEnsemble调用
        self._track_operation("ScEnsemble", initial_input, initial_output, initial_cost)
        
        # Ensure sc_ensemble returned correct format
        if not isinstance(best_solution, dict) or 'response' not in best_solution:
            # Fallback to first solution
            best_solution = {"response": solutions[0]}
        
        # Test the solution
        print(f"    🧪 测试解决方案...")
        try:
            test_result = await self.test(problem=problem, solution=best_solution['response'], entry_point=entry_point)
        except Exception as e:
            print(f"    ⚠️ 测试过程出现异常: {str(e)}")
            # If test fails, return the solution without testing
            test_result = {"result": False, "solution": best_solution['response']}
        
        if test_result['result']:
            print(f"    ✅ 测试通过!")
            tokens_info = {
                "input_tokens": self.token_tracker.total_input_tokens,
                "output_tokens": self.token_tracker.total_output_tokens,
                "total_tokens": self.token_tracker.total_tokens,
            }
            return test_result['solution'], tokens_info, self.llm.usage_tracker.total_cost
        else:
            # 第5次调用: If the test fails, try to fix the solution
            print(f"    🔧 测试失败，尝试修复解决方案...")
            try:
                initial_input, initial_output, initial_cost = self._get_tokens_and_cost()
                
                fixed_solution = await self.custom(input=f"Problem: {problem}\nFailed solution: {best_solution['response']}\nError: {test_result['solution']}", instruction=FIX_CODE_PROMPT)
                
                # 追踪Custom修复调用
                self._track_operation("Custom_Fix", initial_input, initial_output, initial_cost)
                
                print(f"    ✅ 修复完成")
                tokens_info = {
                    "input_tokens": self.token_tracker.total_input_tokens,
                    "output_tokens": self.token_tracker.total_output_tokens,
                    "total_tokens": self.token_tracker.total_tokens,
                }
                return fixed_solution['response'], tokens_info, self.llm.usage_tracker.total_cost
            except Exception as e:
                print(f"    ❌ 修复失败: {str(e)}")
                # If fix also fails, return the original solution
                tokens_info = {
                    "input_tokens": self.token_tracker.total_input_tokens,
                    "output_tokens": self.token_tracker.total_output_tokens,
                    "total_tokens": self.token_tracker.total_tokens,
                }
                return best_solution['response'], tokens_info, self.llm.usage_tracker.total_cost

class MBPPExperiment:
    """MBPP实验管理器"""
    
    def __init__(self, data_file: str, results_dir: str = "results", llm_config: str = "openai/gpt-4o-mini"):
        self.data_file = data_file
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # 配置模型
        self.models_config = LLMsConfig.default()
        self.llm_config = self.models_config.get(llm_config)
        
        if self.llm_config is None:
            raise ValueError(f"Model '{llm_config}' not found in configuration")
        
        # 创建benchmark实例
        self.benchmark = MBPPBenchmark("MBPP", data_file, str(self.results_dir))
        
        # 结果存储
        self.results = {
            "optimal_workflow": {
                "predictions": [],
                "costs": [],
                "tokens": [],
                "times": [],
                "success_rate": 0.0,
                "total_cost": 0.0,
                "avg_tokens": 0.0,
                "total_tokens": 0,  # 添加总tokens统计
                "avg_time": 0.0
            }
        }
    
    async def load_test_data(self, max_samples: int = None) -> List[Dict]:
        """加载测试数据"""
        test_data = []
        with open(self.data_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                if line.strip():
                    test_data.append(json.loads(line.strip()))
        return test_data

    async def run_optimal_workflow(self, test_data: List[Dict]) -> Dict:
        """运行Vanilla_OptimalWorkflow测试"""
        print(f"\n{'='*60}")
        print(f"🚀 开始运行Vanilla_OptimalWorkflow测试")
        print(f"📊 测试样本数量: {len(test_data)}")
        print(f"{'='*60}")
        
        workflow = Vanilla_Workflow(
            name="MBPP_Vanilla_OptimalWorkflow", 
            llm_config=self.llm_config, 
            dataset="MBPP"
        )
        
        predictions = []
        costs = []
        tokens_list = []
        times = []
        success_count = 0
        successes = []
        
        for i, sample in enumerate(test_data):
            entry_point = sample.get('entry_point', f'problem_{i}')
            print(f"\n📝 处理问题 {i+1}/{len(test_data)}: {entry_point}")
            
            start_time = time.time()
            
            try:
                result, tokens_info, cost = await workflow(sample['prompt'], sample['entry_point'])
                
                processing_time = time.time() - start_time
                
                # 使用MBPPBenchmark检查结果正确性
                ret = self.benchmark.check_solution(result, sample['test'], sample['entry_point'])
                success = ret[0] == self.benchmark.PASS
                successes.append(success)
                
                predictions.append(result)
                costs.append(cost)
                tokens_list.append(tokens_info)
                times.append(processing_time)
                
                if success:
                    success_count += 1
                
                print(f"  ✅ 完成 - 成功: {'是' if success else '否'}")
                print(f"  💰 成本: ${cost:.6f}")
                print(f"  📊 Tokens: {tokens_info['input_tokens']} + {tokens_info['output_tokens']} = {tokens_info['total_tokens']}")
                print(f"  ⏱️  时间: {processing_time:.2f}s")
                
            except Exception as e:
                processing_time = time.time() - start_time
                print(f"  ❌ 错误: {str(e)}")
                
                predictions.append(f"ERROR: {str(e)}")
                costs.append(0.0)
                tokens_list.append({"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                times.append(processing_time)
                successes.append(False)
        
        # 计算统计数据
        total_cost = sum(costs)
        total_tokens = sum(t['total_tokens'] for t in tokens_list)  # 添加总tokens
        avg_tokens = total_tokens / len(tokens_list) if tokens_list else 0
        avg_time = sum(times) / len(times) if times else 0
        success_rate = success_count / len(test_data) if test_data else 0
        
        print(f"\n{'='*60}")
        print(f"📊 Vanilla_OptimalWorkflow测试完成!")
        print(f"✅ 成功率: {success_rate:.2%} ({success_count}/{len(test_data)})")
        print(f"💰 总成本: ${total_cost:.6f}")
        print(f"📈 总tokens: {total_tokens:,} (平均: {avg_tokens:.0f}/样本)")
        print(f"⏱️  平均时间: {avg_time:.2f}s")
        print(f"{'='*60}")
        
        # 保存详细日志
        log_data = []
        for i, (sample, pred, cost, tokens, time_taken, success) in enumerate(zip(test_data, predictions, costs, tokens_list, times, successes)):
            # 修正individual success判断
            individual_success = not pred.startswith("ERROR:") and not pred.startswith("Error:")
            log_entry = {
                "problem_id": i,
                "entry_point": sample.get('entry_point', ''),
                "problem": sample.get('prompt', sample.get('text', '')),
                "prediction": pred,
                "cost": cost,
                "tokens": tokens,
                "time": time_taken,
                "success": individual_success
            }
            log_data.append(log_entry)
        
        # 保存日志
        log_file = self.results_dir / "Vanilla_optimal_workflow_log.jsonl"
        with open(log_file, 'w', encoding='utf-8') as f:
            for entry in log_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        return {
            "predictions": predictions,
            "costs": costs,
            "tokens": tokens_list,
            "times": times,
            "success_rate": success_rate,
            "total_cost": total_cost,
            "avg_tokens": avg_tokens,
            "total_tokens": total_tokens,  # 添加总tokens
            "avg_time": avg_time,
            "log_file": str(log_file)
        }

    # 注释掉OverallWorkflow相关方法
    # async def run_overall_workflow(self, test_data: List[Dict]) -> Dict:
    #     """运行OverallWorkflow测试"""
    #     pass

    async def run_experiment(self, method_type: str = "optimal", max_samples: int = None):
        """运行实验"""
        print(f"🎯 加载测试数据...")
        test_data = await self.load_test_data(max_samples)
        
        if method_type == "optimal":
            self.results["optimal_workflow"] = await self.run_optimal_workflow(test_data)
        # elif method_type == "overall":
        #     self.results["overall_workflow"] = await self.run_overall_workflow(test_data)
        else:
            raise ValueError(f"目前只支持 optimal 方法，不支持: {method_type}")
    
    async def save_results(self, method_type: str):
        """保存实验结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        summary = {
            "experiment_info": {
                "data_file": str(self.data_file),
                "method_type": method_type,
                "timestamp": timestamp,
                "llm_config": str(self.llm_config.model)
            },
            "results": {}
        }
        
        for key, result in self.results.items():
            if result["predictions"]:  # 只保存有数据的结果
                summary["results"][key] = {
                    "success_rate": result["success_rate"],
                    "total_cost": result["total_cost"],
                    "avg_tokens": result["avg_tokens"],
                    "total_tokens": result["total_tokens"],  # 添加总tokens
                    "avg_time": result["avg_time"],
                    "sample_count": len(result["predictions"])
                }
        
        # 保存摘要
        summary_file = self.results_dir / f"mbpp_{method_type}_summary_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 结果已保存到: {summary_file}")

async def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='MBPP Vanilla_OptimalWorkflow测试')
    parser.add_argument('--data_file', default='data/datasets/mbpp_test.jsonl', help='测试数据文件路径')
    parser.add_argument('--method', choices=['optimal'], default='optimal', help='测试方法(目前只支持optimal)')
    parser.add_argument('--max_samples', type=int, default=None, help='最大测试样本数')
    parser.add_argument('--llm_config', default='openai/gpt-4o-mini', help='LLM配置')
    
    args = parser.parse_args()
    
    print(f"🎯 MBPP Vanilla_OptimalWorkflow 测试")
    print(f"📂 数据文件: {args.data_file}")
    print(f"🔧 方法: {args.method}")
    print(f"🔢 最大样本数: {args.max_samples}")
    print(f"🤖 LLM配置: {args.llm_config}")
    
    # 创建实验
    experiment = MBPPExperiment(
        data_file=args.data_file,
        results_dir="z_ablation/results/mbpp",
        llm_config=args.llm_config
    )
    
    # 运行实验
    await experiment.run_experiment(method_type=args.method, max_samples=args.max_samples)
    
    # 保存结果
    await experiment.save_results(args.method)

if __name__ == "__main__":
    asyncio.run(main())