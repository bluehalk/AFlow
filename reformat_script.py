import pandas as pd
import os
import json
from datetime import datetime

def reformat_to_jsonl(source_file_path, output_dir, batch_size=50):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 读取源CSV文件
    try:
        df = pd.read_csv(source_file_path)
    except FileNotFoundError:
        print(f"Error: Source file not found at {source_file_path}")
        return

    # 计算总批次数
    num_batches = (len(df) - 1) // batch_size + 1

    for i in range(num_batches):
        # 获取当前批次的数据
        start_index = i * batch_size
        end_index = start_index + batch_size
        batch_df = df.iloc[start_index:end_index]

        if batch_df.empty:
            continue

        # 计算批次总结信息
        avg_score = batch_df["score"].mean() if "score" in batch_df.columns else 0.0
        input_tokens = batch_df["input_tokens"].sum() if "input_tokens" in batch_df.columns else 0
        output_tokens = batch_df["output_tokens"].sum() if "output_tokens" in batch_df.columns else 0
        total_tokens = batch_df["total_tokens"].sum() if "total_tokens" in batch_df.columns else 0
        total_calls = batch_df["calls"].sum() if "calls" in batch_df.columns else 0
        avg_tokens = batch_df["total_tokens"].mean() if "total_tokens" in batch_df.columns else 0.0
        avg_calls = batch_df["calls"].mean() if "calls" in batch_df.columns else 0.0
        
        partial_summary_info = f"""Batch {i+1}/{num_batches}
Avg score: {avg_score:.3f}
Tokens: {input_tokens} + {output_tokens} = {total_tokens}
Total calls: {total_calls}
Avg tokens: {avg_tokens:.3f}
Avg calls: {avg_calls:.3f}
"""

        # 生成批次文件名
        timestamp = datetime.now().strftime("%m_%d_%H_%M")
        base_filename = f"batch_{i:02d}_score_{avg_score:.3f}_{timestamp}"
        jsonl_filename = f"{base_filename}.jsonl"
        summary_filename = f"batch_summary.txt"
        
        jsonl_output_file = os.path.join(output_dir, jsonl_filename)
        summary_output_file = os.path.join(output_dir, summary_filename)

        # 写入JSONL文件
        with open(jsonl_output_file, "w", encoding="utf-8") as f:
            for record in batch_df.to_dict('records'):
                # Rename 'question' to 'problem' to match the desired format
                if 'question' in record:
                    record['problem'] = record.pop('question')

                if 'expected_output' in record:
                    record['solution'] = record.pop('expected_output')

                f.write(json.dumps(record) + '\n')

        # 写入总结信息到txt文件
        with open(summary_output_file, "a", encoding="utf-8") as f:
            f.write(partial_summary_info + '\n')
            
        print(f"Batch {i} saved to {jsonl_output_file}")
        print(f"Summary for batch {i} saved to {summary_output_file}")



if __name__ == '__main__':
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument('source_file', type=str, help='Path to the source CSV file')
    args.add_argument('output_directory', type=str, help='Path to the output directory')
    parsed_args = args.parse_args()
    reformat_to_jsonl(parsed_args.source_file, parsed_args.output_directory)