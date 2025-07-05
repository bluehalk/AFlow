 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的MBPP测试脚本

用于了解MBPP数据集的结构和Graph方法的工作原理
"""

import json
import os
import sys
from pathlib import Path

def load_dataset(dataset_path: str, num_samples: int = 5):
    """加载MBPP数据集的前几个样本"""
    data = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            if line.strip():
                data.append(json.loads(line))
    return data

def analyze_mbpp_structure():
    """分析MBPP数据集结构"""
    print("🎉 MBPP数据集结构分析")
    print("=" * 60)
    
    # 数据集路径
    datasets = {
        "validate": "data/datasets/mbpp_validate.jsonl",
        "test": "data/datasets/mbpp_test.jsonl", 
        "public_test": "data/datasets/mbpp_public_test.jsonl"
    }
    
    for name, path in datasets.items():
        if os.path.exists(path):
            print(f"\n📁 {name.upper()} 数据集 ({path}):")
            
            # 统计总数
            with open(path, 'r') as f:
                total_count = sum(1 for line in f if line.strip())
            print(f"   总样本数: {total_count}")
            
            # 查看前3个样本
            samples = load_dataset(path, 3)
            for i, sample in enumerate(samples):
                print(f"\n   📝 样本 {i+1}:")
                print(f"      - Task ID: {sample.get('task_id', 'N/A')}")
                print(f"      - Entry Point: {sample.get('entry_point', 'N/A')}")
                print(f"      - Prompt Preview: {sample.get('prompt', '')[:100]}...")
                print(f"      - Test Cases: {len(sample.get('test_list', []))} 个")
        else:
            print(f"\n❌ {name.upper()} 数据集文件不存在: {path}")

def show_graph_method_steps():
    """展示Graph方法的工作步骤"""
    print("\n🔄 Graph方法工作流程")
    print("=" * 60)
    
    steps = [
        ("1️⃣ 多解决方案生成", "生成3个不同的Python代码解决方案"),
        ("2️⃣ 自一致性集成", "从3个方案中选择最一致的解决方案"),
        ("3️⃣ 代码测试验证", "使用数据集提供的测试用例验证代码"),
        ("4️⃣ 错误修复机制", "如果测试失败，自动尝试修复代码")
    ]
    
    for step, description in steps:
        print(f"\n{step} {description}")
    
    print("\n📋 使用的操作符:")
    operators = [
        ("CustomCodeGenerate", "专门用于Python代码生成"),
        ("ScEnsemble", "自一致性集成选择最佳方案"),
        ("Test", "执行测试用例验证代码正确性"),
        ("Custom", "通用操作符，用于错误修复等")
    ]
    
    for op, desc in operators:
        print(f"   - {op}: {desc}")

def show_example_problem():
    """展示一个具体的MBPP问题示例"""
    print("\n📖 MBPP问题示例")
    print("=" * 60)
    
    # 检查数据集文件
    validate_path = "data/datasets/mbpp_validate.jsonl"
    if os.path.exists(validate_path):
        sample = load_dataset(validate_path, 1)[0]
        
        print(f"📝 问题描述:")
        print(f"   {sample.get('prompt', 'N/A')}")
        
        print(f"\n🎯 函数入口点: {sample.get('entry_point', 'N/A')}")
        
        print(f"\n✅ 标准解决方案:")
        code_lines = sample.get('code', '').split('\n')
        for line in code_lines:
            if line.strip():
                print(f"   {line}")
        
        print(f"\n🧪 测试用例:")
        for i, test in enumerate(sample.get('test_list', [])[:3]):
            print(f"   {i+1}. {test}")
        
        if len(sample.get('test_list', [])) > 3:
            print(f"   ... 还有 {len(sample.get('test_list', [])) - 3} 个测试用例")
            
    else:
        print("❌ 无法找到数据集文件，无法展示示例")

def show_reproduction_guide():
    """展示复现实验指南"""
    print("\n🚀 复现实验指南")
    print("=" * 60)
    
    print("1️⃣ 环境准备:")
    print("   - 确保已安装所有依赖包")
    print("   - 配置LLM API密钥 (在 config/config2.yaml)")
    print("   - 确保数据集文件存在于 data/datasets/ 目录")
    
    print("\n2️⃣ 运行简单测试:")
    print("   # 测试单个样本")
    print("   python scripts/interface.py")
    
    print("\n3️⃣ 运行完整实验:")
    print("   # 使用验证集")
    print("   python run.py --dataset MBPP --max_rounds 1")
    
    print("\n4️⃣ 实验参数说明:")
    params = [
        ("--dataset MBPP", "指定使用MBPP数据集"),
        ("--max_rounds 1", "只运行1轮（测试模式）"),
        ("--sample 4", "指定样本数量"),
        ("--exec_model_name", "指定执行模型 (如 gpt-4o-mini)")
    ]
    
    for param, desc in params:
        print(f"   {param}: {desc}")

def main():
    """主函数"""
    print("🎓 MBPP数据集和Graph方法介绍")
    print("=" * 80)
    
    print("\n📚 关于MBPP:")
    print("MBPP (Mostly Basic Python Problems) 是一个Python编程问题数据集")
    print("包含974个基础到中等难度的编程任务，每个问题都有完整的测试用例")
    
    # 分析数据集结构
    analyze_mbpp_structure()
    
    # 展示Graph方法
    show_graph_method_steps()
    
    # 展示问题示例
    show_example_problem()
    
    # 展示复现指南
    show_reproduction_guide()
    
    print("\n" + "=" * 80)
    print("💡 接下来可以:")
    print("1. 查看 data/results/results/MBPP/graphs_test/round_14/ 目录")
    print("2. 运行 python run.py --dataset MBPP --max_rounds 1 进行测试")
    print("3. 检查 config/config2.yaml 确保API配置正确")
    print("=" * 80)

if __name__ == "__main__":
    main()