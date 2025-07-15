#!/usr/bin/env python3
"""
简化的CSV文件正确性交集分析器
用法: python csv_intersection_analyzer.py file1.csv file2.csv
"""

import pandas as pd
import os
import sys
from datetime import datetime

def analyze_two_csv_files(file1_path, file2_path):
    """分析两个CSV文件的正确性交集"""
    
    print(f"🔍 分析文件:")
    print(f"  文件1: {file1_path}")
    print(f"  文件2: {file2_path}")
    
    # 读取文件
    try:
        df1 = pd.read_csv(file1_path)
        df2 = pd.read_csv(file2_path)
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return
    
    # 基于问题内容合并
    merged_df = pd.merge(df1, df2, on='question', how='inner', suffixes=('_1', '_2'))
    
    if len(merged_df) == 0:
        print("❌ 两个文件没有共同问题")
        return
    
    print(f"📊 共同问题数量: {len(merged_df)}")
    
    # 判断正确性
    merged_df['correct_1'] = merged_df['score_1'] == 1.0
    merged_df['correct_2'] = merged_df['score_2'] == 1.0
    
    # 分类统计
    both_correct = merged_df[merged_df['correct_1'] & merged_df['correct_2']]
    only_1_correct = merged_df[merged_df['correct_1'] & ~merged_df['correct_2']]
    only_2_correct = merged_df[~merged_df['correct_1'] & merged_df['correct_2']]
    both_wrong = merged_df[~merged_df['correct_1'] & ~merged_df['correct_2']]
    
    # 输出统计
    print(f"\n📈 统计结果:")
    print(f"  两个都正确: {len(both_correct):3d} 个问题")
    print(f"  只有文件1正确: {len(only_1_correct):3d} 个问题")
    print(f"  只有文件2正确: {len(only_2_correct):3d} 个问题")
    print(f"  两个都错误: {len(both_wrong):3d} 个问题")
    print(f"  ─────────────────────────")
    print(f"  总计: {len(merged_df):3d} 个问题")
    
    # 计算正确率
    rate_1 = (len(both_correct) + len(only_1_correct)) / len(merged_df)
    rate_2 = (len(both_correct) + len(only_2_correct)) / len(merged_df)
    union_rate = (len(both_correct) + len(only_1_correct) + len(only_2_correct)) / len(merged_df)
    
    print(f"\n📊 正确率:")
    print(f"  文件1: {rate_1:.1%}")
    print(f"  文件2: {rate_2:.1%}")
    print(f"  并集: {union_rate:.1%}")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 选择保存的列
    save_columns = ['question', 'score_1', 'score_2']
    if 'prediction_1' in merged_df.columns:
        save_columns.extend(['prediction_1', 'prediction_2'])
    if 'expected_output_1' in merged_df.columns:
        save_columns.extend(['expected_output_1'])
    
    # 保存各类别数据
    categories = [
        (both_correct, "both_correct", "两个都正确"),
        (only_1_correct, "only_file1_correct", "只有文件1正确"),
        (only_2_correct, "only_file2_correct", "只有文件2正确"),
        (both_wrong, "both_wrong", "两个都错误")
    ]
    log_dir = "/Users/codiplay/Documents/ustc_workspace/AFlow/analysis/csv_intersection_analyzer"
    os.makedirs(log_dir, exist_ok=True)
    saved_files = []
    for data, filename, desc in categories:
        if len(data) > 0:
            output_file = f"{filename}_{timestamp}.csv"
            available_cols = [col for col in save_columns if col in data.columns]
            data[available_cols].to_csv(os.path.join(log_dir, output_file), index=False)
            saved_files.append((output_file, desc, len(data)))
            print(f"💾 {desc}: {output_file} ({len(data)} 个问题)")
    
    # 保存摘要
    summary_file = f"intersection_summary_{timestamp}.txt"
    with open(os.path.join(log_dir, summary_file), "w", encoding="utf-8") as f:
        f.write("CSV文件正确性交集分析摘要\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"文件1: {file1_path}\n")
        f.write(f"文件2: {file2_path}\n\n")
        f.write(f"共同问题数量: {len(merged_df)}\n\n")
        f.write("统计结果:\n")
        f.write(f"  两个都正确: {len(both_correct):3d} 个问题 ({len(both_correct)/len(merged_df):.1%})\n")
        f.write(f"  只有文件1正确: {len(only_1_correct):3d} 个问题 ({len(only_1_correct)/len(merged_df):.1%})\n")
        f.write(f"  只有文件2正确: {len(only_2_correct):3d} 个问题 ({len(only_2_correct)/len(merged_df):.1%})\n")
        f.write(f"  两个都错误: {len(both_wrong):3d} 个问题 ({len(both_wrong)/len(merged_df):.1%})\n\n")
        f.write("正确率:\n")
        f.write(f"  文件1: {rate_1:.1%}\n")
        f.write(f"  文件2: {rate_2:.1%}\n")
        f.write(f"  并集: {union_rate:.1%}\n\n")
        f.write("输出文件:\n")
        for filename, desc, count in saved_files:
            f.write(f"  {desc}: {filename}\n")
    
    print(f"📄 摘要保存到: {summary_file}")
    print(f"\n✅ 分析完成！")
    
    return {
        'total': len(merged_df),
        'both_correct': len(both_correct),
        'only_1_correct': len(only_1_correct),
        'only_2_correct': len(only_2_correct),
        'both_wrong': len(both_wrong)
    }

def main():
    if len(sys.argv) != 3:
        print("用法: python csv_intersection_analyzer.py <file1.csv> <file2.csv>")
        print("示例: python csv_intersection_analyzer.py batch1.csv batch2.csv")
        return
    
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    
    if not os.path.exists(file1):
        print(f"❌ 文件不存在: {file1}")
        return
    if not os.path.exists(file2):
        print(f"❌ 文件不存在: {file2}")
        return
    
    result = analyze_two_csv_files(file1, file2)
    
    if result:
        print(f"\n🎯 核心统计: {result['both_correct']} + {result['only_1_correct']} + {result['only_2_correct']} + {result['both_wrong']} = {result['total']}")

if __name__ == "__main__":
    main() 