import pandas as pd

df = pd.read_csv('/Users/codiplay/Documents/ustc_workspace/AFlow/z_ablation/results/MATH/round_5/experiments/A_Merge/20250714_161524/batch_00_score_0.520_07_14_16_16.csv')

# 删除包含统计项的行
df_cleaned = df[~df['question'].str.contains('Total calls|Avg calls|Batch', na=False)]

# 保存为干净的 CSV
df_cleaned.to_csv('cleaned_file.csv', index=False)
