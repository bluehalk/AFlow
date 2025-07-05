 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MBPP数据集复现脚本

用于复现MBPP (Mostly Basic Python Problems) 数据集的实验结果。
该脚本使用AFlow框架中的Graph方法来解决Python编程问题。
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from benchmarks.mbpp import MBPPBenchmark
from scripts.async_llm import LLMsConfig

# 导入Graph工作流
from data.results.results.MBPP.graphs_test.round_14.graph import Workflow

def load_dataset(dataset_path: str) -> List[Dict]:
    """加载MBPP数据集"""
    data = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

class MBPPExperiment:
    """MBPP实验类"""
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        self.llm_config = self._get_llm_config()
        self.benchmark = MBPPBenchmark(
            name="MBPP",
            file_path="data/datasets",
            log_path="logs"
        )
        
    def _get_llm_config(self):
        """获取LLM配置"""
        models_config = LLMsConfig.default()
        llm_config = models_config.get(self.model_name)
        if llm_config is None:
            raise ValueError(f"Model '{self.model_name}' not found in configuration")
        return llm_config
    
    async def run_single_problem(self, problem_data: Dict) -> Tuple[str, str, str, float, float]:
        """运行单个问题"""
        # 创建工作流实例
        workflow = Workflow(
            name="MBPP_Graph_Workflow",
            llm_config=self.llm_config,
            dataset="MBPP"
        )
        
        # 使用benchmark的evaluate_problem方法
        return await self.benchmark.evaluate_problem(problem_data, workflow)
    
    async def run_experiment(self, dataset_path: str, num_samples: int = None, output_path: str = None):
        """运行完整实验"""
        print(f"🚀 开始MBPP数据集实验...")
        print(f"📊 使用模型: {self.model_name}")
        print(f"📁 数据集路径: {dataset_path}")
        
        # 加载数据集
        data = load_dataset(dataset_path)
        
        if num_samples:
            data = data[:num_samples]
            print(f"🔢 限制样本数量: {num_samples}")
        
        print(f"📋 总共问题数量: {len(data)}")
        
        # 创建输出目录
        if output_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = f"mbpp_results_{timestamp}.csv"
        
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        
        # 运行实验
        results = []
        total_cost = 0.0
        correct_count = 0
        
        print("\n" + "="*80)
        print("开始处理问题...")
        print("="*80)
        
        for i, problem in enumerate(data):
            print(f"\n📝 处理问题 {i+1}/{len(data)}: {problem.get('entry_point', 'Unknown')}")
            
            try:
                # 运行单个问题
                input_text, prediction, expected_output, score, cost = await self.run_single_problem(problem)
                
                results.append({
                    'question': input_text,
                    'prediction': prediction,
                    'expected_output': expected_output,
                    'score': score,
                    'cost': cost
                })
                
                total_cost += cost
                if score > 0:
                    correct_count += 1
                
                print(f"✅ 结果: {'通过' if score > 0 else '失败'}")
                print(f"💰 成本: ${cost:.6f}")
                print(f"📊 当前准确率: {correct_count}/{i+1} ({correct_count/(i+1)*100:.2f}%)")
                
            except Exception as e:
                print(f"❌ 错误: {str(e)}")
                results.append({
                    'question': problem.get('prompt', ''),
                    'prediction': f"Error: {str(e)}",
                    'expected_output': '',
                    'score': 0.0,
                    'cost': 0.0
                })
        
        # 保存结果
        self.save_results(results, output_path)
        
        # 打印统计信息
        self.print_statistics(results, total_cost, correct_count, len(data))
        
        return results
    
    def save_results(self, results: List[Dict], output_path: str):
        """保存结果到CSV文件"""
        import csv
        
        # 获取列名
        columns = self.benchmark.get_result_columns()
        
        # 映射结果字段到CSV列
        field_mapping = {
            'question': 'inputs',
            'prediction': 'prediction', 
            'expected_output': 'expected_output',
            'score': 'score',
            'cost': 'cost'
        }
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(columns)
            
            for result in results:
                row = []
                for col in columns:
                    # 找到对应的结果字段
                    field_name = None
                    for result_field, csv_col in field_mapping.items():
                        if csv_col == col:
                            field_name = result_field
                            break
                    
                    if field_name and field_name in result:
                        row.append(result[field_name])
                    else:
                        row.append('')
                
                writer.writerow(row)
        
        print(f"\n💾 结果已保存到: {output_path}")
    
    def print_statistics(self, results: List[Dict], total_cost: float, correct_count: int, total_count: int):
        """打印统计信息"""
        accuracy = correct_count / total_count * 100 if total_count > 0 else 0
        avg_cost = total_cost / total_count if total_count > 0 else 0
        
        print("\n" + "="*80)
        print("📊 实验统计结果")
        print("="*80)
        print(f"🎯 总准确率: {correct_count}/{total_count} ({accuracy:.2f}%)")
        print(f"💰 总成本: ${total_cost:.6f}")
        print(f"💸 平均成本: ${avg_cost:.6f}")
        print(f"🔧 使用模型: {self.model_name}")
        print("="*80)

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MBPP数据集复现实验")
    parser.add_argument(
        "--dataset", 
        type=str, 
        choices=["test", "validate", "public_test"],
        default="validate",
        help="选择数据集类型 (default: validate)"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="gpt-4o-mini",
        help="LLM模型名称 (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        help="限制样本数量 (不指定则使用全部样本)"
    )
    parser.add_argument(
        "--output", 
        type=str,
        help="输出文件路径 (default: 自动生成)"
    )
    
    args = parser.parse_args()
    
    # 确定数据集路径
    dataset_paths = {
        "test": "data/datasets/mbpp_test.jsonl",
        "validate": "data/datasets/mbpp_validate.jsonl", 
        "public_test": "data/datasets/mbpp_public_test.jsonl"
    }
    
    dataset_path = dataset_paths[args.dataset]
    
    if not os.path.exists(dataset_path):
        print(f"❌ 数据集文件不存在: {dataset_path}")
        print("请确保数据集文件存在于正确的路径。")
        return
    
    print("🎉 MBPP数据集复现实验")
    print("="*50)
    print("📖 关于MBPP数据集:")
    print("  - MBPP (Mostly Basic Python Problems)")
    print("  - 包含Python编程问题和测试用例")
    print("  - 每个问题包含: prompt, code, test, entry_point等字段")
    print("  - Graph方法包含: 代码生成 → 自一致性集成 → 测试验证 → 错误修复")
    print("="*50)
    
    # 创建实验实例并运行
    experiment = MBPPExperiment(model_name=args.model)
    
    try:
        asyncio.run(experiment.run_experiment(
            dataset_path=dataset_path,
            num_samples=args.num_samples,
            output_path=args.output
        ))
    except KeyboardInterrupt:
        print("\n⚠️  实验被用户中断")
    except Exception as e:
        print(f"\n❌ 实验运行出错: {str(e)}")
        raise

if __name__ == "__main__":
    main()