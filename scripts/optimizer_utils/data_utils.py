import datetime
import json
import os
import random

import numpy as np
import pandas as pd

from scripts.logs import logger
from scripts.utils.common import read_json_file, write_json_file


class DataUtils:
    def __init__(self, root_path: str):
        self.root_path = root_path
        self.top_scores = [] # 存储所有轮次的分数数据
    
    # 加载历史数据：从results.json读取所有轮次的性能数据
    def load_results(self, path: str) -> list: # list: 包含所有轮次结果的列表
        """
        加载历史实验结果
        Returns:
            list: 包含所有轮次结果的列表，每个元素格式：
                  {"round": 1, "score": 0.85, "avg_cost": 0.01, "total_cost": 0.10, "time": "..."}
        """
        result_path = os.path.join(path, "results.json")
        if os.path.exists(result_path):
            with open(result_path, "r") as json_file:
                try:
                    return json.load(json_file)
                except json.JSONDecodeError:
                    return []
        return []

    def get_top_rounds(self, sample: int, path=None, mode="Graph"):
        self._load_scores(path, mode)
        unique_rounds = set()
        unique_top_scores = []

        first_round = next((item for item in self.top_scores if item["round"] == 1), None)
        if first_round:
            unique_top_scores.append(first_round)
            unique_rounds.add(1)

        for item in self.top_scores:
            if item["round"] not in unique_rounds:
                unique_top_scores.append(item)
                unique_rounds.add(item["round"])

                if len(unique_top_scores) >= sample:
                    break

        return unique_top_scores

    def select_round(self, items):
        """
        使用软混合概率策略选择父节点轮次进行扩展
        """ 
        if not items:
            raise ValueError("Item list is empty.")

        sorted_items = sorted(items, key=lambda x: x["score"], reverse=True) # 降序
        scores = [item["score"] * 100 for item in sorted_items]

        probabilities = self._compute_probabilities(scores)
        logger.info(f"\nMixed probability distribution: {probabilities}")
        logger.info(f"\nSorted rounds: {sorted_items}")

        selected_index = np.random.choice(len(sorted_items), p=probabilities)
        logger.info(f"\nSelected index: {selected_index}, Selected item: {sorted_items[selected_index]}")

        return sorted_items[selected_index]

    def _compute_probabilities(self, scores, alpha=0.2, lambda_=0.3):
        scores = np.array(scores, dtype=np.float64)
        n = len(scores)

        if n == 0:
            raise ValueError("Score list is empty.")

        # 📊 组件1：均匀概率分布（探索性）
        # 每个轮次都有相等的1/n概率被选中
        uniform_prob = np.full(n, 1.0 / n, dtype=np.float64)

        # 📈 组件2：基于分数的指数加权分布（利用性）
        max_score = np.max(scores)
        shifted_scores = scores - max_score  # 数值稳定性：避免exp溢出
        exp_weights = np.exp(alpha * shifted_scores)  # 指数权重：exp(α × score)

        sum_exp_weights = np.sum(exp_weights)
        if sum_exp_weights == 0:
            raise ValueError("Sum of exponential weights is 0, cannot normalize.")

        score_prob = exp_weights / sum_exp_weights  # 归一化概率分布

        # 🎯 最终：软混合概率分布
        # λ控制探索vs利用的平衡：
        # - λ=1: 完全均匀（纯探索）
        # - λ=0: 完全基于分数（纯利用）
        # - λ=0.3: 30%探索 + 70%利用
        mixed_prob = lambda_ * uniform_prob + (1 - lambda_) * score_prob

        # 确保概率和为1
        total_prob = np.sum(mixed_prob)
        if not np.isclose(total_prob, 1.0):
            mixed_prob = mixed_prob / total_prob

        return mixed_prob

    def load_log(self, cur_round, path=None, mode: str = "Graph"):
        if mode == "Graph":
            log_dir = os.path.join(self.root_path, "workflows", f"round_{cur_round}", "log.json")
        else:
            log_dir = path

        # 检查文件是否存在
        if not os.path.exists(log_dir):
            return ""  # 如果文件不存在，返回空字符串
        logger.info(log_dir)
        data = read_json_file(log_dir, encoding="utf-8")

        if isinstance(data, dict):
            data = [data]
        elif not isinstance(data, list):
            data = list(data)

        if not data:
            return ""

        sample_size = min(3, len(data))
        random_samples = random.sample(data, sample_size)

        log = ""
        for sample in random_samples:
            log += json.dumps(sample, indent=4, ensure_ascii=False) + "\n\n"

        return log

    def get_results_file_path(self, graph_path: str) -> str:
        return os.path.join(graph_path, "results.json")

    def create_result_data(self, round: int, score: float, avg_cost: float, total_cost: float) -> dict:
        now = datetime.datetime.now()
        return {"round": round, "score": score, "avg_cost": avg_cost, "total_cost": total_cost, "time": now}

    def save_results(self, json_file_path: str, data: list):
        write_json_file(json_file_path, data, encoding="utf-8", indent=4)

    def _load_scores(self, path=None, mode="Graph"):
        if mode == "Graph":
            rounds_dir = os.path.join(self.root_path, "workflows")
        else:
            rounds_dir = path

        result_file = os.path.join(rounds_dir, "results.json")
        self.top_scores = []

        data = read_json_file(result_file, encoding="utf-8")
        df = pd.DataFrame(data)

        scores_per_round = df.groupby("round")["score"].mean().to_dict()

        for round_number, average_score in scores_per_round.items():
            self.top_scores.append({"round": round_number, "score": average_score})

        self.top_scores.sort(key=lambda x: x["score"], reverse=True)

        return self.top_scores
