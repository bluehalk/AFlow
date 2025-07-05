import pandas as pd
import json

# 读取CSV结果文件
results_df = pd.read_csv('/home/codiplay/AFlow/z_ablation/results/DirectIO_20250701_215636.csv')

# 找到所有错误的样例
wrong_cases = results_df[results_df['correct'] == False]

# 读取原始jsonl文件
original_problems = []
with open('z_ablation/200_gsm8k.jsonl', 'r') as f:
    for line in f:
        original_problems.append(json.loads(line))

# 创建新的tough cases列表
tough_cases = []
for _, wrong_case in wrong_cases.iterrows():
    sample_id = wrong_case['sample_id']
    # 找到对应的原始问题
    original_problem = original_problems[sample_id]
    tough_cases.append(original_problem)

# 保存为新的jsonl文件
with open('z_ablation/DirectIO_error_gsm8k.jsonl', 'w') as f:
    for case in tough_cases:
        f.write(json.dumps(case) + '\n')

print(f"总共找到 {len(tough_cases)} 个困难样例，已保存到 DirectIO_error_gsm8k.jsonl") 