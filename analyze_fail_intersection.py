#!/usr/bin/env python3
import json
import os

def load_jsonl(file_path):
    """加载JSON文件（可能是JSONL或JSON数组格式）"""
    questions = []
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return questions
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
            # 尝试作为JSON数组加载
            if content.startswith('['):
                data_list = json.loads(content)
                print(f"📝 检测到JSON数组格式，包含 {len(data_list)} 个条目")
            else:
                # 尝试作为JSONL格式加载
                data_list = []
                for line in content.split('\n'):
                    line = line.strip()
                    if line:
                        data_list.append(json.loads(line))
                print(f"📝 检测到JSONL格式，包含 {len(data_list)} 个条目")
            
            # 从每个条目中提取问题文本
            for data in data_list:
                # 提取问题文本，可能的字段名：question, problem, input等
                question = None
                for key in ['question', 'problem', 'input', 'text']:
                    if key in data:
                        question = data[key]
                        break
                
                if question:
                    questions.append(question)
                else:
                    # 如果没有找到标准字段，打印可用字段
                    if len(questions) == 0:  # 只打印第一个条目的字段
                        print(f"⚠️ 未找到问题字段，可用字段: {list(data.keys())}")
        
        print(f"✅ 成功加载 {len(questions)} 个失败案例从 {file_path}")
        return questions
    
    except Exception as e:
        print(f"❌ 读取文件时出错 {file_path}: {str(e)}")
        return questions

def analyze_fail_intersection():
    # 文件路径
    fail_file1 = \
    "z_ablation/results/MATH/round_5/experiments/A_SPP/20250722_174415/batch_09_score_0.639_07_22_17_56.csv"
    fail_file2 = \
    "z_ablation/results/MATH/round_5/experiments/A_SPP/20250722_180542/batch_09_score_0.500_07_22_18_22.csv"
    
    print("=== 分析失败案例交集 ===\n")
    
    # 加载失败案例
    print("📖 加载失败案例文件...")
    fail_questions1 = load_jsonl(fail_file1)
    fail_questions2 = load_jsonl(fail_file2)
    
    if not fail_questions1 or not fail_questions2:
        print("❌ 无法加载失败案例文件")
        return
    
    print(f"📊 A_Mathemation失败案例数: {len(fail_questions1)}")
    print(f"📊 A_Programmer失败案例数: {len(fail_questions2)}")
    
    # 转换为集合
    fail_set1 = set(fail_questions1)
    fail_set2 = set(fail_questions2)
    
    print(f"📊 A_Mathemation唯一失败案例数: {len(fail_set1)}")
    print(f"📊 A_Programmer唯一失败案例数: {len(fail_set2)}")
    
    # 计算失败案例的交集和并集
    fail_intersection = fail_set1.intersection(fail_set2)
    fail_union = fail_set1.union(fail_set2)
    only_fail_in_1 = fail_set1 - fail_set2
    only_fail_in_2 = fail_set2 - fail_set1
    
    print(f"\n🎯 失败案例交集数量: {len(fail_intersection)}")
    print(f"🎯 失败案例并集数量: {len(fail_union)}")
    print(f"📊 仅A_Mathemation失败: {len(only_fail_in_1)}")
    print(f"📊 仅A_Programmer失败: {len(only_fail_in_2)}")
    
    # 验证数学关系
    print(f"\n🔍 验证数学关系:")
    
    # 已知数据（从之前的分析）
    total_questions = 486
    success_intersection = 185  # 从之前分析得到
    
    print(f"总问题数: {total_questions}")
    print(f"成功交集数: {success_intersection}")
    print(f"失败交集数: {len(fail_intersection)}")
    
    # 验证：成功交集 + 失败交集 = 总数？
    sum_intersections = success_intersection + len(fail_intersection)
    print(f"\n✅ 验证: 成功交集({success_intersection}) + 失败交集({len(fail_intersection)}) = {sum_intersections}")
    print(f"🎯 总问题数: {total_questions}")
    print(f"{'✅ 验证通过！' if sum_intersections == total_questions else '❌ 验证失败！'}")
    
    # 额外验证：每个文件的成功+失败应该等于总数
    success1 = 263  # A_Mathemation成功数
    success2 = 267  # A_Programmer成功数
    
    print(f"\n🔍 额外验证:")
    print(f"A_Mathemation: 成功({success1}) + 失败({len(fail_set1)}) = {success1 + len(fail_set1)}")
    print(f"A_Programmer: 成功({success2}) + 失败({len(fail_set2)}) = {success2 + len(fail_set2)}")
    
    # 保存详细分析结果
    with open("fail_intersection_analysis.txt", "w", encoding="utf-8") as f:
        f.write("=== 失败案例交集分析 ===\n\n")
        f.write(f"失败文件1: {fail_file1}\n")
        f.write(f"失败文件2: {fail_file2}\n\n")
        f.write(f"A_Mathemation失败案例数: {len(fail_questions1)}\n")
        f.write(f"A_Programmer失败案例数: {len(fail_questions2)}\n")
        f.write(f"A_Mathemation唯一失败案例数: {len(fail_set1)}\n")
        f.write(f"A_Programmer唯一失败案例数: {len(fail_set2)}\n\n")
        f.write(f"失败案例交集数量: {len(fail_intersection)}\n")
        f.write(f"失败案例并集数量: {len(fail_union)}\n")
        f.write(f"仅A_Mathemation失败: {len(only_fail_in_1)}\n")
        f.write(f"仅A_Programmer失败: {len(only_fail_in_2)}\n\n")
        f.write(f"验证关系:\n")
        f.write(f"总问题数: {total_questions}\n")
        f.write(f"成功交集数: {success_intersection}\n")
        f.write(f"失败交集数: {len(fail_intersection)}\n")
        f.write(f"成功交集 + 失败交集 = {sum_intersections}\n")
        f.write(f"验证结果: {'通过' if sum_intersections == total_questions else '失败'}\n\n")
        
        f.write("=== 两者都失败的问题示例 ===\n")
        for i, q in enumerate(sorted(list(fail_intersection))[:10], 1):
            f.write(f"{i}. {q[:200]}{'...' if len(q) > 200 else ''}\n")
    
    print(f"\n✅ 详细分析结果已保存到: fail_intersection_analysis.txt")
    
    return {
        'fail1_count': len(fail_questions1),
        'fail2_count': len(fail_questions2),
        'fail1_unique': len(fail_set1),
        'fail2_unique': len(fail_set2),
        'fail_intersection': len(fail_intersection),
        'fail_union': len(fail_union),
        'only_fail_in_1': len(only_fail_in_1),
        'only_fail_in_2': len(only_fail_in_2),
        'verification_passed': sum_intersections == total_questions
    }

if __name__ == "__main__":
    result = analyze_fail_intersection()
    if result:
        print(f"\n🎉 分析完成！")
        print(f"📊 失败交集: {result['fail_intersection']} 个问题")
        print(f"✅ 数学验证: {'通过' if result['verification_passed'] else '失败'}") 