import datetime
import json
import os
import random

import numpy as np
import pandas as pd

from scripts.logs import logger
from scripts.utils.common import read_json_file, write_json_file


class DataUtils:
    """
    AFlow的数据管理中心，实现蒙特卡洛树搜索的核心算法
    
    核心职责：
    1. 管理实验轮次数据（分数、成本、时间）
    2. 实现软混合概率策略进行父节点选择
    3. 提供历史经验数据支持
    """
    
    def __init__(self, root_path: str):
        self.root_path = root_path
        self.top_scores = []  # 存储所有轮次的分数数据

    def load_results(self, path: str) -> list:
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
        """
        🎯 蒙特卡洛树搜索：获取候选父节点集合
        
        策略：
        1. 必须包含round_1（确保有baseline对比）
        2. 按分数降序排列，取前sample个轮次
        3. 去重确保每个轮次只出现一次
        
        Args:
            sample: 需要返回的轮次数量（候选父节点数）
            
        Returns:
            list: 排序后的候选轮次列表 [{"round": 5, "score": 0.92}, ...]
        """
        # 加载所有轮次的分数数据
        self._load_scores(path, mode)
        unique_rounds = set()
        unique_top_scores = []

        # 🔑 策略1：必须包含第一轮作为baseline
        first_round = next((item for item in self.top_scores if item["round"] == 1), None)
        if first_round:
            unique_top_scores.append(first_round)
            unique_rounds.add(1)

        # 🔑 策略2：按分数降序添加其他轮次
        for item in self.top_scores:
            if item["round"] not in unique_rounds:
                unique_top_scores.append(item)
                unique_rounds.add(item["round"])

                if len(unique_top_scores) >= sample:
                    break

        return unique_top_scores

    def select_round(self, items):
        """
        🎲 蒙特卡洛树搜索：使用软混合概率策略选择父节点
        
        这是AFlow论文中的核心算法！
        目标：在探索(exploration)和利用(exploitation)之间找到平衡
        
        策略：
        - 高分轮次有更高被选中概率（利用已知好结果）
        - 所有轮次都有被选中的机会（探索新可能性）
        
        Args:
            items: 候选轮次列表 [{"round": 1, "score": 0.8}, {"round": 5, "score": 0.92}]
            
        Returns:
            dict: 被选中的轮次 {"round": 5, "score": 0.92}
        """ 
        if not items:
            raise ValueError("Item list is empty.")

        # 按分数降序排列（分数越高排在越前面）
        sorted_items = sorted(items, key=lambda x: x["score"], reverse=True)
        scores = [item["score"] * 100 for item in sorted_items]  # 放大分数便于计算

        # 🎯 核心：计算软混合概率分布
        probabilities = self._compute_probabilities(scores)
        logger.info(f"\n🎲 Mixed probability distribution: {probabilities}")
        logger.info(f"\n📊 Sorted rounds: {sorted_items}")

        # 🎰 根据概率分布随机选择
        selected_index = np.random.choice(len(sorted_items), p=probabilities)
        logger.info(f"\n✅ Selected index: {selected_index}, Selected item: {sorted_items[selected_index]}")

        return sorted_items[selected_index]

    def _compute_probabilities(self, scores, alpha=0.2, lambda_=0.3):
        """
        🧮 计算软混合概率分布（AFlow论文核心算法）
        
        公式：P_mixed = λ × P_uniform + (1-λ) × P_score
        
        其中：
        - P_uniform: 均匀分布（每个轮次概率相等）→ 探索性
        - P_score: 基于分数的指数分布（高分轮次概率更高）→ 利用性
        - λ: 混合参数，控制探索vs利用的平衡
        - α: 温度参数，控制分数权重的锐度
        
        Args:
            scores: 分数数组 [80, 92, 85, ...]
            alpha: 温度参数，越小分数差异的影响越大
            lambda_: 混合参数，越大越倾向于均匀探索
            
        Returns:
            numpy.array: 概率分布 [0.15, 0.45, 0.25, 0.15]
        """
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
        """
        加载指定轮次的执行日志
        
        用途：
        1. 为LLM提供历史执行的具体案例
        2. 帮助理解哪些策略有效/无效
        3. 避免重复相同的错误
        
        Args:
            cur_round: 目标轮次
            
        Returns:
            str: 格式化的日志文本，包含随机选择的3个执行样例
        """
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

        # 随机选择最多3个样例，避免日志过长
        sample_size = min(3, len(data))
        random_samples = random.sample(data, sample_size)

        log = ""
        for sample in random_samples:
            log += json.dumps(sample, indent=4, ensure_ascii=False) + "\n\n"

        return log

    def get_results_file_path(self, graph_path: str) -> str:
        """获取结果文件路径"""
        return os.path.join(graph_path, "results.json")

    def create_result_data(self, round: int, score: float, avg_cost: float, total_cost: float) -> dict:
        """创建轮次结果数据结构"""
        now = datetime.datetime.now()
        return {"round": round, "score": score, "avg_cost": avg_cost, "total_cost": total_cost, "time": now}

    def save_results(self, json_file_path: str, data: list):
        """保存实验结果到JSON文件"""
        write_json_file(json_file_path, data, encoding="utf-8", indent=4)

    def _load_scores(self, path=None, mode="Graph"):
        """
        加载所有轮次的平均分数数据
        
        处理：
        1. 读取results.json文件
        2. 按轮次分组计算平均分数
        3. 按分数降序排序
        
        Updates:
            self.top_scores: [{"round": 5, "score": 0.92}, {"round": 3, "score": 0.88}, ...]
        """
        if mode == "Graph":
            rounds_dir = os.path.join(self.root_path, "workflows")
        else:
            rounds_dir = path

        result_file = os.path.join(rounds_dir, "results.json")
        self.top_scores = []

        data = read_json_file(result_file, encoding="utf-8")
        df = pd.DataFrame(data)

        # 按轮次分组，计算每轮的平均分数
        scores_per_round = df.groupby("round")["score"].mean().to_dict()

        for round_number, average_score in scores_per_round.items():
            self.top_scores.append({"round": round_number, "score": average_score})

        # 按分数降序排序（最佳轮次在前）
        self.top_scores.sort(key=lambda x: x["score"], reverse=True)

        return self.top_scores 