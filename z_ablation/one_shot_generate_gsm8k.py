#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

from scripts.operators import Custom, ScEnsemble, Programmer, run_code
from scripts.async_llm import LLMsConfig, create_llm_instance
from benchmarks.gsm8k import GSM8KBenchmark
import concurrent.futures

DatasetType = Literal["HumanEval", "MBPP", "GSM8K", "MATH", "HotpotQA", "DROP"]

PYTHON_CODE_VERIFIER_PROMPT = """
You are a professional Python programmer. Your task is to write complete, self-contained code based on a given mathematical problem and output the answer. 
The code should include all necessary imports and dependencies, and be ready to run without additional setup or environment configuration.
Problem description: {problem}
Other analysis: {analysis}
{feedback}

Your code should:
1. Implement the calculation steps described in the problem.
2. Define a function named `solve` that performs the calculation and returns the result. 
The `solve` function should not require any input parameters; 
instead, it should obtain all necessary inputs from within the function or from globally defined variables.
3. `solve` function return the final calculation result.

Please ensure your code is efficient, well-commented, and follows Python best practices. 
The output should be limited to basic data types such as strings, integers, and floats. 
It is prohibited to transmit images or other file formats. The code output is intended for a text-based language model.
"""

SC_ENSEMBLE_PROMPT = """
Given the question described as follows: {question}
Several solutions have been generated to address the given question. They are as follows:
{solutions}

Carefully evaluate these solutions and identify the answer that appears most frequently across them. 
This consistency in answers is crucial for determining the most reliable solution.

In the "thought" field, provide a detailed explanation of your thought process. 
In the "solution_letter" field, output only the single letter ID (A, B, C, etc.) corresponding to the most consistent solution.
 Do not include any additional text or explanation in the "solution_letter" field.
"""


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

# 综合的overall_prompt，将workflow的所有步骤整合到一个prompt中
OVERALL_WORKFLOW_PROMPT = """
You are an expert mathematician with advanced problem-solving capabilities. 
You will solve a math problem using a comprehensive multi-approach workflow. Follow these steps carefully:

## STEP 1: GENERATE MULTIPLE SOLUTIONS
First, generate THREE different solutions to the problem using different approaches or perspectives:

**Solution A (Direct Mathematical Approach):**
- Read and understand the problem thoroughly
- Identify all key information, variables, and relationships
- Determine the appropriate mathematical concepts, formulas, or equations
- Solve step-by-step, showing all work clearly
- Double-check calculations and reasoning

**Solution B (Alternative Mathematical Approach):**
- Take a different mathematical perspective or method
- Use alternative formulas or calculation strategies if possible
- Show complete step-by-step reasoning
- Verify calculations independently

**Solution C (Verification-Focused Approach):**
- Solve using a third method or approach
- Focus on verification and cross-checking
- Use estimation or alternative calculation methods to validate
- Ensure logical consistency

## STEP 2: SELF-CONSISTENCY EVALUATION
Carefully evaluate the three solutions above:
- Compare the final answers from all three solutions
- Identify which answer appears most frequently (consistency check)
- Analyze the reasoning quality and mathematical rigor of each solution
- If there are discrepancies, determine which solution has the most reliable approach
- Select the most consistent and mathematically sound solution

## STEP 3: PYTHON CODE VERIFICATION
Write Python code to verify your selected solution:
- Implement the calculation steps described in your selected solution
- Create a function named `solve()` that performs the calculation and returns the result
- The function should be self-contained with all necessary inputs defined within
- Include clear comments explaining each calculation step
- Execute the logic to verify your mathematical solution

## STEP 4: FINAL ANSWER
Based on your multi-approach analysis and code verification:
- If the code output matches your mathematical solution, use that result
- If there's a discrepancy, carefully review and use the most reliable approach
- Provide your final answer as a single numerical value
- Do NOT include units, additional text, or explanations with the final answer

## FORMAT YOUR RESPONSE AS:

### Solution A:
[Your first mathematical solution here]

### Solution B:
[Your second mathematical solution here]

### Solution C:
[Your third mathematical solution here]

### Self-Consistency Evaluation:
[Compare the three solutions and select the most consistent one]

### Python Code Verification:
```python
def solve():
    # [Your verification code here]
    # [Include clear comments]
    return result

# Execute and show result
answer = solve()
print(answer)
```

### Final Answer:
[Single numerical value only]

Now solve this problem:

"""

class IO_with_optimal_prompt:
    """简单的输入输出模式，使用最优prompt"""
    def __init__(self, name: str, llm_config, dataset: str, max_retries: int = 3):
        self.name = name
        self.llm_config = llm_config
        self.dataset = dataset
        self.max_retries = max_retries
        self.llm = create_llm_instance(llm_config)
        self.custom = Custom(self.llm)
    
    async def _call_with_simple_retry(self, problem: str):
        """重试机制的内部方法"""
        for attempt in range(self.max_retries):
            try:
                if attempt > 0:
                    await asyncio.sleep(0.1)  # 重试时短暂延迟
                result = await self.custom(input=problem, instruction=OPTIMAL_MATH_SOLVE_PROMPT)
                return result['response'], self.llm.get_usage_summary()["total_cost"]
            except Exception as e:
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ['connection', 'timeout', 'network', 'rate limit']):
                    if attempt < self.max_retries - 1:
                        print(f"    ⚠️  API错误，重试中: {e}")
                        continue
                raise e
        raise Exception("所有重试都失败了")
    
    async def __call__(self, problem: str):
        """主要的调用方法"""
        print(f"  IO_with_optimal_prompt: 处理问题...")
        return await self._call_with_simple_retry(problem)


class OptimalWorkflow:
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
        """简单重试辅助函数 - 去掉延迟版本"""
        for attempt in range(self.max_retries):
            try:
                if attempt > 0:
                    await asyncio.sleep(0.1)  # 重试时只需很短延迟
                return await func(*args, **kwargs)
            except Exception as e:
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ['connection', 'timeout', 'network', 'rate limit']):
                    if attempt < self.max_retries - 1:
                        print(f"    ⚠️  API错误，重试中: {e}")
                        continue
                raise e

    async def __call__(self, problem: str):
        """最优工作流实现 - 无延迟版本"""
        solutions = []
        
        # 生成3个解决方案 - 完全并发执行
        print(f"  OptimalWorkflow: 并发生成3个解决方案...")
        tasks = []
        for i in range(3):
            task = self._call_with_simple_retry(
                self.custom, input=problem, instruction=OPTIMAL_MATH_SOLVE_PROMPT
            )
            tasks.append(task)
        
        # 完全并发执行，无延迟
        try:
            # 使用gather进行真正的并发执行
            solutions = await asyncio.gather(*tasks)
            
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
                
                # 顺序执行时也去掉延迟
        
        # # 使用自一致性集成
        # print("  OptimalWorkflow: 进行自一致性集成...")
        # valid_solutions = [s for s in solution_texts if not str(s).startswith("Error")]
        # if not valid_solutions:
        #     print("    ❌ 没有有效解决方案进行集成")
        #     return "OptimalWorkflow执行失败", 0.0
            
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

class OverallWorkflow:
    """使用综合prompt在单次调用中完成整个workflow，包含代码执行"""
    def __init__(self, name: str, llm_config, dataset: str, max_retries: int = 3):
        self.name = name
        self.llm_config = llm_config
        self.dataset = dataset
        self.max_retries = max_retries
        self.llm = create_llm_instance(llm_config)
        self.custom = Custom(self.llm)
        # 创建进程池用于代码执行
        self.process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=1)
    
    def __del__(self):
        """确保进程池在对象销毁时关闭"""
        if hasattr(self, 'process_pool'):
            self.process_pool.shutdown(wait=True)
    
    async def exec_code(self, code, timeout=30):
        """异步执行代码并在超时时返回错误"""
        loop = asyncio.get_running_loop()
        
        try:
            # 使用类级别的进程池
            future = loop.run_in_executor(self.process_pool, run_code, code)
            # 等待任务完成或超时
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            future.cancel()
            import gc
            gc.collect()
            return "Error", "Code execution timed out"
        except concurrent.futures.process.BrokenProcessPool:
            # 如果进程池损坏，重新创建
            self.process_pool.shutdown(wait=False)
            self.process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=1)
            return "Error", "Process pool broken, try again"
        except Exception as e:
            return "Error", f"Unknown error: {str(e)}"
    
    def _extract_python_code(self, response: str) -> str:
        """从响应中提取Python代码"""
        try:
            import re
            
            # 匹配 ```python 代码块
            python_blocks = re.findall(r'```python\s*\n(.*?)\n```', response, re.DOTALL)
            if python_blocks:
                return python_blocks[0].strip()
            
            # 匹配一般的 ``` 代码块
            code_blocks = re.findall(r'```\s*\n(.*?)\n```', response, re.DOTALL)
            for block in code_blocks:
                if 'def solve()' in block:
                    return block.strip()
            
            # 如果没有找到代码块，查找def solve()函数
            solve_match = re.search(r'def solve\(\):.*?(?=\n\n|\n[a-zA-Z]|\Z)', response, re.DOTALL)
            if solve_match:
                return solve_match.group(0).strip()
            
            return None
            
        except Exception as e:
            print(f"代码提取错误: {str(e)}")
            return None
    
    def _extract_final_answer(self, response: str, code_result=None) -> str:
        """从响应中提取最终答案，优先使用代码执行结果"""
        try:
            # 如果有代码执行结果，优先使用
            if code_result is not None:
                import re
                if isinstance(code_result, (int, float)):
                    return str(code_result)
                elif isinstance(code_result, str):
                    numbers = re.findall(r'\d+\.?\d*', str(code_result))
                    if numbers:
                        return numbers[-1]
            
            # 查找 "Final Answer:" 部分
            if "Final Answer:" in response:
                lines = response.split('\n')
                for line in lines:
                    if "Final Answer:" in line:
                        import red
                        numbers = re.findall(r'\d+\.?\d*', line)
                        if numbers:
                            return numbers[-1]
            
            # 如果找不到，尝试从最后几行提取数字
            lines = response.strip().split('\n')
            for line in reversed(lines[-5:]):
                import re
                numbers = re.findall(r'\d+\.?\d*', line)
                if numbers:
                    return numbers[-1]
            
            return "Could not extract answer"
            
        except Exception as e:
            return f"Extraction error: {str(e)}"
    
    async def _call_with_simple_retry(self, problem: str):
        """重试机制的内部方法，包含代码执行"""
        for attempt in range(self.max_retries):
            try:
                if attempt > 0:
                    await asyncio.sleep(0.1)  # 重试时短暂延迟
                
                # 调用大模型生成响应
                result = await self.custom(input=problem, instruction=OVERALL_WORKFLOW_PROMPT)
                response = result['response']
                
                # 提取并执行Python代码
                python_code = self._extract_python_code(response)
                code_result = None
                
                if python_code:
                    print(f"    🐍 执行Python验证代码...")
                    status, output = await self.exec_code(python_code)
                    if status == "Success":
                        code_result = output
                        print(f"    ✅ 代码执行成功: {output}")
                    else:
                        print(f"    ❌ 代码执行失败: {output}")
                else:
                    print(f"    ⚠️  未找到Python代码")
                
                # 提取最终答案（优先使用代码结果）
                final_answer = self._extract_final_answer(response, code_result)
                
                return final_answer, self.llm.get_usage_summary()["total_cost"]
                
            except Exception as e:
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ['connection', 'timeout', 'network', 'rate limit']):
                    if attempt < self.max_retries - 1:
                        print(f"    ⚠️  API错误，重试中: {e}")
                        continue
                raise e
        raise Exception("所有重试都失败了")
    
    async def __call__(self, problem: str):
        """主要的调用方法"""
        print(f"  OverallWorkflow: 使用综合prompt+代码验证处理问题...")
        return await self._call_with_simple_retry(problem)

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
                "total_samples": "困难样例 (25个错误案例)",
                "max_retries": 3,
                "optimization": "测试单prompt综合workflow vs 多步骤workflow",
                "method": "对比IO, OptimalWorkflow, OverallWorkflow"
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
    
    async def run_io_with_optimal_prompt(self, test_data: List[Dict]) -> Dict:
        """运行IO_with_optimal_prompt测试"""
        method_name = "IO_with_optimal_prompt"
        print(f"\n🚀 开始测试方法: {method_name}")
        print("=" * 60)
        
        # 创建方法实例
        method = IO_with_optimal_prompt(
            name=f"GSM8K_{method_name}", 
            llm_config=self.llm_config, 
            dataset="GSM8K",
            max_retries=3
        )
        
        results = []
        failed_samples = []
        start_time = time.time()
        
        for i, sample in enumerate(test_data):
            sample_start_time = time.time()
            try:
                print(f"🔄 [IO_with_optimal_prompt] 处理样本 {i+1}/{len(test_data)}: {sample['question'][:50]}...")
                
                # 调用方法
                prediction, cost = await method(sample['question'])
                
                # 评估结果
                expected = self.benchmark.extract_number(sample['answer'])
                predicted = self.benchmark.extract_number(prediction)
                score, _ = self.benchmark.calculate_score(expected, predicted)
                
                # 获取token使用情况
                usage_summary = method.llm.get_usage_summary()
                
                result = {
                    "sample_id": i,
                    "question": sample['question'],
                    "expected": expected,
                    "predicted": predicted,
                    "prediction_text": prediction,
                    "score": score,
                    "cost": cost,
                    "correct": score > 0.5,
                    "processing_time": time.time() - sample_start_time,
                    "input_tokens": usage_summary.get("total_input_tokens", 0),
                    "output_tokens": usage_summary.get("total_output_tokens", 0),
                    "total_tokens": usage_summary.get("total_tokens", 0)
                }
                results.append(result)
                
                print(f"✅ [IO_with_optimal_prompt] 样本 {i+1} - 得分: {score:.1f}, 成本: ${cost:.6f}, tokens: {result['total_tokens']}, 用时: {result['processing_time']:.1f}s")
                
            except Exception as e:
                processing_time = time.time() - sample_start_time
                error_msg = str(e)
                
                print(f"❌ [IO_with_optimal_prompt] 样本 {i+1} 最终失败: {error_msg}")
                
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
            
            # 每10个样本显示进度
            if (i + 1) % 10 == 0:
                successful_results = [r for r in results if 'error' not in r]
                if successful_results:
                    avg_score = sum(r['score'] for r in successful_results) / len(successful_results)
                    total_cost = sum(r['cost'] for r in results)
                    elapsed_time = time.time() - start_time
                    estimated_total_time = elapsed_time * len(test_data) / (i + 1)
                    remaining_time = estimated_total_time - elapsed_time
                    print(f"📈 [IO_with_optimal_prompt] 进度: {i+1}/{len(test_data)}, 准确率: {avg_score:.2f}, 成本: ${total_cost:.3f}")
                    print(f"⏱️  已用时: {elapsed_time/60:.1f}分钟, 预计剩余: {remaining_time/60:.1f}分钟")
        
        # 计算最终统计
        total_time = time.time() - start_time
        successful_results = [r for r in results if 'error' not in r]
        
        if successful_results:
            scores = [r['score'] for r in successful_results]
            costs = [r['cost'] for r in results]
            input_tokens = [r.get('input_tokens', 0) for r in results]
            output_tokens = [r.get('output_tokens', 0) for r in results]
            total_tokens = [r.get('total_tokens', 0) for r in results]
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
                "total_input_tokens": sum(input_tokens),
                "total_output_tokens": sum(output_tokens),
                "total_tokens_sum": sum(total_tokens),
                "avg_input_tokens": statistics.mean(input_tokens) if input_tokens else 0,
                "avg_output_tokens": statistics.mean(output_tokens) if output_tokens else 0,
                "avg_total_tokens": statistics.mean(total_tokens) if total_tokens else 0,
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
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_tokens_sum": 0,
                "avg_input_tokens": 0,
                "avg_output_tokens": 0,
                "avg_total_tokens": 0,
                "total_time_seconds": total_time,
                "total_time_minutes": total_time / 60,
                "avg_time_per_sample": total_time / len(results),
                "detailed_results": results
            }
        
        print(f"\n📊 [IO_with_optimal_prompt] 最终结果:")
        print(f"   准确率: {method_stats['accuracy']:.1%} ({method_stats['correct_samples']}/200)")
        if method_stats['failed_samples'] > 0:
            print(f"   失败样本: {method_stats['failed_samples']}")
        print(f"   总成本: ${method_stats['total_cost']:.4f}")
        print(f"   总token数: {method_stats['total_tokens_sum']:,} (输入: {method_stats['total_input_tokens']:,}, 输出: {method_stats['total_output_tokens']:,})")
        print(f"   平均token/样本: {method_stats['avg_total_tokens']:.0f} (输入: {method_stats['avg_input_tokens']:.0f}, 输出: {method_stats['avg_output_tokens']:.0f})")
        print(f"   用时: {method_stats['total_time_minutes']:.1f}分钟")
        
        return method_stats

    async def run_optimal_workflow(self, test_data: List[Dict]) -> Dict:
        """运行OptimalWorkflow测试 - 无延迟版本"""
        method_name = "OptimalWorkflow"
        print(f"\n🚀 开始测试方法: {method_name} (无延迟版本)")
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
        
        # 去掉样本间延迟 - 直接连续处理
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
                
                # 获取token使用情况
                usage_summary = method.llm.get_usage_summary()
                
                result = {
                    "sample_id": i,
                    "question": sample['question'],
                    "expected": expected,
                    "predicted": predicted,
                    "prediction_text": prediction,
                    "score": score,
                    "cost": cost,
                    "correct": score > 0.5,
                    "processing_time": time.time() - sample_start_time,
                    "input_tokens": usage_summary.get("total_input_tokens", 0),
                    "output_tokens": usage_summary.get("total_output_tokens", 0),
                    "total_tokens": usage_summary.get("total_tokens", 0)
                }
                results.append(result)
                
                print(f"✅ [OptimalWorkflow] 样本 {i+1} - 得分: {score:.1f}, 成本: ${cost:.6f}, tokens: {result['total_tokens']}, 用时: {result['processing_time']:.1f}s")
                
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
            
            # 完全去掉样本间延迟 - 直接处理下一个样本
            
            # 每10个样本显示进度
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

    async def run_overall_workflow(self, test_data: List[Dict]) -> Dict:
        """运行OverallWorkflow测试 - 单prompt综合版本"""
        method_name = "OverallWorkflow"
        print(f"\n🚀 开始测试方法: {method_name} (单prompt综合版本)")
        print("=" * 60)
        
        # 创建方法实例
        method = OverallWorkflow(
            name=f"GSM8K_{method_name}", 
            llm_config=self.llm_config, 
            dataset="GSM8K",
            max_retries=3
        )
        
        results = []
        failed_samples = []
        start_time = time.time()
        
        for i, sample in enumerate(test_data):
            sample_start_time = time.time()
            try:
                print(f"🔄 [OverallWorkflow] 处理样本 {i+1}/{len(test_data)}: {sample['question'][:50]}...")
                
                # 调用方法
                prediction, cost = await method(sample['question'])
                
                # 评估结果
                expected = self.benchmark.extract_number(sample['answer'])
                predicted = self.benchmark.extract_number(prediction)
                score, _ = self.benchmark.calculate_score(expected, predicted)
                
                # 获取token使用情况
                usage_summary = method.llm.get_usage_summary()
                
                result = {
                    "sample_id": i,
                    "question": sample['question'],
                    "expected": expected,
                    "predicted": predicted,
                    "prediction_text": prediction,
                    "score": score,
                    "cost": cost,
                    "correct": score > 0.5,
                    "processing_time": time.time() - sample_start_time,
                    "total_input_tokens": usage_summary.get("total_input_tokens", 0),
                    "total_output_tokens": usage_summary.get("total_output_tokens", 0),
                    "total_tokens": usage_summary.get("total_tokens", 0)
                }
                results.append(result)
                
                print(f"✅ [OverallWorkflow] 样本 {i+1} - 得分: {score:.1f}, 成本: ${cost:.6f}, tokens: {result['total_tokens']}, 用时: {result['processing_time']:.1f}s")
                
            except Exception as e:
                processing_time = time.time() - sample_start_time
                error_msg = str(e)
                
                print(f"❌ [OverallWorkflow] 样本 {i+1} 最终失败: {error_msg}")
                
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
                    "processing_time": processing_time,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "total_tokens": 0
                }
                results.append(result)
            
            # 每10个样本显示进度
            if (i + 1) % 10 == 0:
                successful_results = [r for r in results if 'error' not in r]
                if successful_results:
                    avg_score = sum(r['score'] for r in successful_results) / len(successful_results)
                    total_cost = sum(r['cost'] for r in results)
                    elapsed_time = time.time() - start_time
                    estimated_total_time = elapsed_time * len(test_data) / (i + 1)
                    remaining_time = estimated_total_time - elapsed_time
                    print(f"📈 [OverallWorkflow] 进度: {i+1}/{len(test_data)}, 准确率: {avg_score:.2f}, 成本: ${total_cost:.3f}")
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
                "total_input_tokens": results[-1]['total_input_tokens'],
                "total_output_tokens": results[-1]['total_output_tokens'],
                "total_tokens_sum": results[-1]['total_tokens'],
                "avg_input_tokens": results[-1]['total_input_tokens'] / len(results) if len(results) > 0 else 0,
                "avg_output_tokens": results[-1]['total_output_tokens'] / len(results) if len(results) > 0 else 0,
                "avg_total_tokens": results[-1]['total_tokens'] / len(results) if len(results) > 0 else 0,
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
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_tokens_sum": 0,
                "avg_input_tokens": 0,
                "avg_output_tokens": 0,
                "avg_total_tokens": 0,
                "total_time_seconds": total_time,
                "total_time_minutes": total_time / 60,
                "avg_time_per_sample": total_time / len(results),
                "detailed_results": results
            }
        
        # 记录失败的样本
        if failed_samples:
            self.experiment_log["failed_samples"] = failed_samples
        
        print(f"\n📊 [OverallWorkflow] 最终结果:")
        print(f"   准确率: {method_stats['accuracy']:.1%} ({method_stats['correct_samples']}/{len(test_data)})")
        if method_stats['failed_samples'] > 0:
            print(f"   失败样本: {method_stats['failed_samples']}")
        print(f"   总成本: ${method_stats['total_cost']:.4f}")
        print(f"   总token数: {method_stats['total_tokens_sum']:,} (输入: {method_stats['total_input_tokens']:,}, 输出: {method_stats['total_output_tokens']:,})")
        print(f"   平均token/样本: {method_stats['avg_total_tokens']:.0f} (输入: {method_stats['avg_input_tokens']:.0f}, 输出: {method_stats['avg_output_tokens']:.0f})")
        print(f"   用时: {method_stats['total_time_minutes']:.1f}分钟")
        
        return method_stats
    
    async def run_experiment(self, method_type: str = "io"):
        """运行实验
        
        Args:
            method_type: 方法类型，"io" 表示 IO_with_optimal_prompt, "optimal" 表示 OptimalWorkflow, "overall" 表示 OverallWorkflow
        """
        if method_type == "io":
            print("🎯 开始IO_with_optimal_prompt实验 (困难样例)")
            print("=" * 60)
            print(f"✅ 模型配置: {self.llm_config.model}")
            print(f"🚀 方法: 简单输入输出模式 + 最优prompt")
            
            # 加载测试数据
            test_data = await self.load_test_data()
            
            # 运行IO_with_optimal_prompt
            method_stats = await self.run_io_with_optimal_prompt(test_data)
            self.experiment_log["results"] = method_stats
            
            print(f"\n🎉 IO_with_optimal_prompt实验完成! 结果保存在: {self.results_dir}")
            
        elif method_type == "optimal":
            print("🎯 开始OptimalWorkflow实验 (困难样例)")
            print("=" * 60)
            print(f"✅ 模型配置: {self.llm_config.model}")
            print(f"🚀 方法: 完整OptimalWorkflow (生成+集成+验证)")
            
            # 加载测试数据
            test_data = await self.load_test_data()
            
            # 运行OptimalWorkflow
            method_stats = await self.run_optimal_workflow(test_data)
            self.experiment_log["results"] = method_stats
            
            print(f"\n🎉 OptimalWorkflow实验完成! 结果保存在: {self.results_dir}")
            
        elif method_type == "overall":
            print("🎯 开始OverallWorkflow实验 (困难样例)")
            print("=" * 60)
            print(f"✅ 模型配置: {self.llm_config.model}")
            print(f"🚀 方法: 单prompt综合workflow (一次调用完成所有步骤)")
            
            # 加载测试数据
            test_data = await self.load_test_data()
            
            # 运行OverallWorkflow
            method_stats = await self.run_overall_workflow(test_data)
            self.experiment_log["results"] = method_stats
            
            print(f"\n🎉 OverallWorkflow实验完成! 结果保存在: {self.results_dir}")
            
        else:
            raise ValueError("method_type 必须是 'io', 'optimal' 或 'overall'")
        
        # 保存结果
        await self.save_results(method_type)
    
    async def save_results(self, method_type: str):
        """保存实验结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        method_mapping = {
            "io": "IO_with_optimal_prompt",
            "optimal": "OptimalWorkflow", 
            "overall": "OverallWorkflow"
        }
        method_name = method_mapping.get(method_type, method_type)
        
        # 保存详细结果到CSV
        df = pd.DataFrame(self.experiment_log["results"]["detailed_results"])
        csv_file = self.results_dir / f"{method_name}_detailed_results_{timestamp}.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        # 保存完整实验日志
        log_file = self.results_dir / f"{method_name}_experiment_summary_{timestamp}.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(self.experiment_log, f, ensure_ascii=False, indent=2)
        
        # 如果有失败的样本，单独保存
        if self.experiment_log["failed_samples"]:
            failed_samples_file = self.results_dir / f"{method_name}_failed_samples_{timestamp}.json"
            with open(failed_samples_file, 'w', encoding='utf-8') as f:
                json.dump(self.experiment_log["failed_samples"], f, ensure_ascii=False, indent=2)
        
        print(f"💾 结果已保存到 {self.results_dir}")
        print(f"📁 详细结果: {method_name}_{timestamp}.csv")


async def main():
    """主函数"""
    import sys
    
    # 检查命令行参数
    method_type = "overall"  # 默认运行OverallWorkflow
    if len(sys.argv) > 1:
        if sys.argv[1] == "optimal":
            method_type = "optimal"
        elif sys.argv[1] == "io":
            method_type = "io"
        elif sys.argv[1] == "overall":
            method_type = "overall"
        else:
            print("用法: python single_generate_gsm8k.py [io|optimal|overall]")
            print("  io: 运行 IO_with_optimal_prompt")
            print("  optimal: 运行 OptimalWorkflow")
            print("  overall: 运行 OverallWorkflow (单prompt综合版本, 默认)")
            return
    
    method_mapping = {
        "io": "IO_with_optimal_prompt",
        "optimal": "OptimalWorkflow", 
        "overall": "OverallWorkflow"
    }
    method_name = method_mapping.get(method_type, method_type)
    print(f"🎯 GSM8K {method_name}实验 - 困难样例测试")
    print("=" * 60)
    
    # 创建实验
    experiment = OptimalWorkflowExperiment(
        data_file="z_ablation/200_gsm8k.jsonl",  # 使用困难样例数据
        results_dir="z_ablation/results",
        llm_config="meta-llama/llama-3-70b-instruct"  # 使用现有配置
    )
    
    # 运行实验
    await experiment.run_experiment(method_type)

if __name__ == "__main__":
    asyncio.run(main())