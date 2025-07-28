import csv
import json
import os
import logging

logging.basicConfig(filename='filter_script.log', level=logging.DEBUG, filemode='w')

def create_filtered_jsonl(csv_path, jsonl_path, output_path):
    jsonl_data = {}
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    item = json.loads(line)
                    if 'problem' in item:
                        key = item['problem']
                        jsonl_data[key] = item
                        if i < 5:
                            logging.debug(f"JSONL key {i}: {repr(key)}")
                except json.JSONDecodeError:
                    logging.warning(f"Could not decode JSON from line in {jsonl_path}: {line.strip()}")
                    continue
    except FileNotFoundError:
        logging.error(f"The file {jsonl_path} was not found.")
        return

    matched_entries = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                question = row.get('question')
                if question:
                    if i < 5:
                        logging.debug(f"CSV question {i}: {repr(question)}")
                    if question in jsonl_data:
                        matched_entries.append(jsonl_data[question])
    except FileNotFoundError:
        logging.error(f"The file {csv_path} was not found.")
        return
    except Exception as e:
        logging.error(f"An error occurred while reading {csv_path}: {e}")
        return

    if not matched_entries:
        logging.warning("No matches found. Check filter_script.log for debugging info.")

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in matched_entries:
                f.write(json.dumps(entry) + '\n')
        print(f"Successfully created filtered file at: {output_path}")
        print(f"Found {len(matched_entries)} matching entries.")
    except IOError as e:
        print(f"Error writing to output file {output_path}: {e}")

if __name__ == '__main__':
    base_dir = '/Users/codiplay/Documents/ustc_workspace/AFlow'
    csv_file = os.path.join(base_dir, 'z_ablation/results/MATH/round_5/experiments/A_Consistent_Merge/20250715_131556/batch_06_score_0.200_07_15_13_33.csv')
    jsonl_file = os.path.join(base_dir, 'z_ablation/results/MATH/math_test_100.jsonl')
    output_file = os.path.join(base_dir, 'filtered_math_problems.jsonl')
    create_filtered_jsonl(csv_file, jsonl_file, output_file)