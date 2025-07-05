#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MBPP Graph Workflow 测试脚本 - 完全复制graph.py的实现
包括完整的token统计和5次调用追踪
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Literal, Dict, List, Any
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.operators as operator
from scripts.async_llm import create_llm_instance, LLMsConfig

DatasetType = Literal["HumanEval", "MBPP", "GSM8K", "MATH", "HotpotQA", "DROP"]

# 使用和graph.py相同的prompt (从原始prompt.py复制)
CODE_GENERATE_PROMPT = """
Generate a Python function to solve the given problem. Ensure the function name matches the one specified in the problem. Include necessary imports. Use clear variable names and add comments for clarity.

Problem:
{problem}

Function signature:
{entry_point}

Generate the complete function below:
"""

FIX_CODE_PROMPT = """
The provided solution failed to pass the tests. Please analyze the error and fix the code. Ensure the function name and signature remain unchanged. If necessary, add or modify imports, correct logical errors, and improve the implementation.

Problem:
{input}

Provide the corrected function below:
"""

class TokenTracker:
    """Token使用统计器"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置统计"""
        self.calls = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        self.total_cost = 0.0
    
    def add_call(self, operation: str, input_tokens: int, output_tokens: int, cost: float):
        """添加一次调用的统计"""
        self.calls.append({
            "operation": operation,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost": cost
        })
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_tokens += input_tokens + output_tokens
        self.total_cost += cost
    
    def get_summary(self):
        """获取统计摘要"""
        return {
            "total_calls": len(self.calls),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "calls_detail": self.calls
        }

class Workflow:
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
            return test_result['solution'], self.llm.usage_tracker.total_cost
        else:
            # 第5次调用: If the test fails, try to fix the solution
            print(f"    🔧 测试失败，尝试修复解决方案...")
            try:
                initial_input, initial_output, initial_cost = self._get_tokens_and_cost()
                
                fixed_solution = await self.custom(input=f"Problem: {problem}\nFailed solution: {best_solution['response']}\nError: {test_result['solution']}", instruction=FIX_CODE_PROMPT)
                
                # 追踪Custom修复调用
                self._track_operation("Custom_Fix", initial_input, initial_output, initial_cost)
                
                print(f"    ✅ 修复完成")
                return fixed_solution['response'], self.llm.usage_tracker.total_cost
            except Exception as e:
                print(f"    ❌ 修复失败: {str(e)}")
                # If fix also fails, return the original solution
                return best_solution['response'], self.llm.usage_tracker.total_cost

class MBPPTestExperiment:
    """MBPP测试实验管理器"""
    
    def __init__(self, llm_config: str = "openai/gpt-4o-mini"):
        # 配置模型
        self.models_config = LLMsConfig.default()
        self.llm_config = self.models_config.get(llm_config)
        
        if self.llm_config is None:
            raise ValueError(f"Model '{llm_config}' not found in configuration")
    
    async def test_single_problem(self, problem_data: Dict):
        """测试单个问题"""
        problem = problem_data.get('prompt', problem_data.get('text', ''))
        entry_point = problem_data.get('entry_point', '')
        test_cases = problem_data.get('test_list', problem_data.get('test', []))
        
        print(f"\n🎯 测试问题: {entry_point}")
        print(f"📝 问题描述: {problem[:100]}...")
        print(f"🧪 测试用例数量: {len(test_cases) if isinstance(test_cases, list) else 'N/A'}")
        
        # 创建workflow
        workflow = Workflow(
            name="MBPP_Graph_Test",
            llm_config=self.llm_config,
            dataset="MBPP"
        )
        
        start_time = time.time()
        
        try:
            # 运行workflow
            result, cost = await workflow(problem, entry_point)
            
            processing_time = time.time() - start_time
            
            # 获取详细的token统计
            token_summary = workflow.token_tracker.get_summary()
            
            # print(f"\n📊 结果统计:")
            # print(f"   ✅ 处理完成!")
            # print(f"   💰 总成本: ${cost:.6f}")
            # print(f"   🕐 处理时间: {processing_time:.2f}秒")
            # print(f"   📞 总调用次数: {token_summary['total_calls']}")
            # print(f"   📈 总Input tokens: {token_summary['total_input_tokens']:,}")
            # print(f"   📉 总Output tokens: {token_summary['total_output_tokens']:,}")
            # print(f"   📊 总tokens: {token_summary['total_tokens']:,}") 
            
            # print(f"\n📋 详细调用统计:")
            # for i, call in enumerate(token_summary['calls_detail'], 1):
            #     print(f"   {i}. {call['operation']}:")
            #     print(f"      Input: {call['input_tokens']:,}, Output: {call['output_tokens']:,}, Total: {call['total_tokens']:,}")
            #     print(f"      Cost: ${call['cost']:.6f}")
            
            print(f"\n🔧 生成的最终代码:")
            print("="*60)
            print(result)
            print("="*60)
            
            return {
                "success": test_result['result'],
                "result": result,
                "cost": cost,
                "processing_time": processing_time,
                "token_summary": token_summary,
                "entry_point": entry_point
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"\n❌ 测试失败: {str(e)}")
            
            return {
                "success": False,
                "error": str(e),
                "processing_time": processing_time,
                "entry_point": entry_point
            }
    
    async def test_multiple_problems(self, max_samples: int = 3):
        """测试多个问题"""
        print(f"🚀 开始MBPP Graph Workflow测试 - 最多{max_samples}个样本")
        print("="*80)
        
        # 尝试不同的数据文件
        data_files = [
            "data/datasets/mbpp_test.jsonl", 
        ]
        
        test_data = []
        for data_file in data_files:
            try:
                with open(data_file, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i >= max_samples:
                            break
                        if line.strip():
                            test_data.append(json.loads(line.strip()))
                print(f"📂 成功加载 {data_file}")
                break
            except FileNotFoundError:
                continue
        
        if not test_data:
            print("❌ 未找到测试数据文件!")
            return []
        
        print(f"📂 加载了 {len(test_data)} 个测试样本")
        
        results = []
        total_start_time = time.time()
        
        for i, sample in enumerate(test_data):
            print(f"\n{'='*20} 样本 {i+1}/{len(test_data)} {'='*20}")
            
            result = await self.test_single_problem(sample)
            results.append(result)
            
            # 显示累计统计
            successful_results = [r for r in results if r['success']]
            if successful_results:
                total_cost = sum(r['cost'] for r in successful_results)
                total_calls = sum(r['token_summary']['total_calls'] for r in successful_results)
                total_input_tokens = sum(r['token_summary']['total_input_tokens'] for r in successful_results)
                total_output_tokens = sum(r['token_summary']['total_output_tokens'] for r in successful_results)
                total_tokens = sum(r['token_summary']['total_tokens'] for r in successful_results)
                
                print(f"\n📈 累计统计 ({i+1}/{len(test_data)}):")
                print(f"   成功: {len(successful_results)}, 失败: {len(results) - len(successful_results)}")
                print(f"   总调用: {total_calls}")
                print(f"   总成本: ${total_cost:.6f}")
                print(f"   总Input tokens: {total_input_tokens:,}")
                print(f"   总Output tokens: {total_output_tokens:,}")
                print(f"   总tokens: {total_tokens:,}")
        
        # 最终统计
        total_time = time.time() - total_start_time
        successful_results = [r for r in results if r['success']]
        
        print(f"\n🎉 测试完成! 总用时: {total_time/60:.2f}分钟")
        print("="*80)
        print("📊 最终统计:")
        print(f"   总样本: {len(results)}")
        print(f"   成功: {len(successful_results)}")
        print(f"   失败: {len(results) - len(successful_results)}")
        
        if successful_results:
            total_cost = sum(r['cost'] for r in successful_results)
            total_calls = sum(r['token_summary']['total_calls'] for r in successful_results)
            total_input_tokens = sum(r['token_summary']['total_input_tokens'] for r in successful_results)
            total_output_tokens = sum(r['token_summary']['total_output_tokens'] for r in successful_results)
            total_tokens = sum(r['token_summary']['total_tokens'] for r in successful_results)
            avg_time = sum(r['processing_time'] for r in successful_results) / len(successful_results)
            
            print(f"   总调用: {total_calls} (平均 {total_calls/len(successful_results):.1f}/样本)")
            print(f"   总成本: ${total_cost:.6f}")
            print(f"   平均成本/样本: ${total_cost/len(successful_results):.6f}")
            print(f"   总Input tokens: {total_input_tokens:,}")
            print(f"   总Output tokens: {total_output_tokens:,}")
            print(f"   总tokens: {total_tokens:,}")
            print(f"   平均tokens/样本: {total_tokens/len(successful_results):.0f}")
            print(f"   平均处理时间/样本: {avg_time:.2f}秒")
            
            # 显示每种操作的统计
            print(f"\n📋 操作类型统计:")
            operation_stats = {}
            for result in successful_results:
                for call in result['token_summary']['calls_detail']:
                    op = call['operation']
                    if op not in operation_stats:
                        operation_stats[op] = {'count': 0, 'input_tokens': 0, 'output_tokens': 0, 'cost': 0.0}
                    operation_stats[op]['count'] += 1
                    operation_stats[op]['input_tokens'] += call['input_tokens']
                    operation_stats[op]['output_tokens'] += call['output_tokens']
                    operation_stats[op]['cost'] += call['cost']
            
            for op, stats in operation_stats.items():
                avg_input = stats['input_tokens'] / stats['count']
                avg_output = stats['output_tokens'] / stats['count']
                avg_cost = stats['cost'] / stats['count']
                print(f"   {op}: {stats['count']}次")
                print(f"     平均Input: {avg_input:.0f}, 平均Output: {avg_output:.0f}")
                print(f"     平均成本: ${avg_cost:.6f}")
        
        return results

async def main():
    """主函数"""
    import sys
    
    # 检查命令行参数
    max_samples = 3  # 默认测试3个样本
    
    if len(sys.argv) > 1:
        try:
            max_samples = int(sys.argv[1])
        except ValueError:
            print("用法: python test_mbpp_graph.py [max_samples]")
            print("  max_samples: 最大测试样本数 (默认3)")
            return
    
    print(f"🎯 MBPP Graph Workflow 测试")
    print(f"📋 完全复制 graph.py 的实现逻辑")
    print(f"📊 包含完整的5次调用token统计")
    print(f"🔢 测试样本数: {max_samples}")
    
    # 创建实验
    experiment = MBPPTestExperiment(llm_config="openai/gpt-4o-mini")
    
    # 运行测试
    if max_samples == 1:
        # 单个测试 - 使用第一个样本
        data_files = [
            "data/datasets/mbpp_test.jsonl"
        ]
        
        sample = None
        for data_file in data_files:
            try:
                with open(data_file, 'r') as f:
                    sample = json.loads(f.readline().strip())
                break
            except FileNotFoundError:
                continue
        
        if sample:
            await experiment.test_single_problem(sample)
        else:
            print("❌ 未找到测试数据文件!")
    else:
        # 多个测试
        await experiment.test_multiple_problems(max_samples)

if __name__ == "__main__":
    asyncio.run(main()) 