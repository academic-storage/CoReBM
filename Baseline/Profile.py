# -*- coding: utf-8 -*-
"""
基于个人档案的代码审查员推荐系统复现
作者: [您的名字]
日期: [日期]

本脚本根据论文《Profile based recommendation of code reviewers》中的方法，
实现了一个自动化的代码审查员推荐系统。系统通过分析历史审查记录
构建审查员档案，并基于新提交的代码变更推荐最合适的审查员。

数据集说明：
- 每对测试数据由两个JSONL文件组成：
    - `i_test.jsonl`: 测试提交数据，共500个文件，编号从0到499。
    - `i_history.jsonl`: 对应的历史提交数据，共500个文件，编号从0到499。
- 每个测试文件包含一条测试提交数据，包含实际的审查员信息。
- 每个历史文件包含多条历史提交数据，用于构建审查员档案。

输出：
- 每个测试案例的Top1、Top3、Top5推荐结果。
- 每个测试案例的准确率（Accuracy@1, Accuracy@3, Accuracy@5）、
  平均倒数排名（MRR@1, MRR@3, MRR@5）。
- 所有测试案例的整体平均指标，打印输出并保存至`metrics.txt`文件中。

使用方法：
确保所有的`*_test.jsonl`和`*_history.jsonl`文件位于指定的数据目录下，
然后运行本脚本。推荐结果和性能指标将输出到控制台并保存至`metrics.txt`。
"""

import os
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from typing import List, Dict, Tuple

RESULT_FILE = open("profile_results.txt", "w", encoding="utf-8")

class ReviewerRecommender:
    def __init__(self, decay_by_date: bool = True, decay_by_id: bool = False,
                 f_date: float = 0.1, l_date: float = 0.0001,
                 f_id: float = 0.1, l_id: float = 0.0001):
        """
        初始化审查员推荐系统。

        :param decay_by_date: 是否基于时间对历史审查记录进行衰减。
        :param decay_by_id: 是否基于提交数对历史审查记录进行衰减。
        :param f_date: 日期衰减因子（例如，0.5表示影响力减半）。
        :param l_date: 日期半衰期，以天为单位（例如，183天）。
        :param f_id: ID衰减因子（例如，0.5表示影响力减半）。
        :param l_id: ID半衰期，以提交数为单位（例如，2500次提交）。
        """
        self.decay_by_date = decay_by_date
        self.decay_by_id = decay_by_id
        self.f_date = f_date
        self.l_date = l_date
        self.f_id = f_id
        self.l_id = l_id

    @staticmethod
    def load_jsonl(file_path: str) -> List[Dict]:
        """
        读取JSONL文件并返回JSON对象的列表。

        :param file_path: JSONL文件的路径。
        :return: JSON对象列表。
        """
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # 跳过空行
                    data.append(json.loads(line))
        return data

    @staticmethod
    def tokenize_file_path(file_path: str) -> List[str]:
        """
        将文件路径分割为各个目录和文件名的段落。

        :param file_path: 文件路径字符串。
        :return: 路径段列表。
        """
        return file_path.strip().split(os.sep)  # 使用os.sep处理不同操作系统的路径分隔符

    @staticmethod
    def parse_datetime(date_str: str) -> datetime:
        """
        解析日期字符串，处理微秒和纳秒。

        :param date_str: 日期字符串，例如 "2012-02-20 12:54:53.513000000"
        :return: datetime对象。
        """
        date_str = date_str.strip()  # 移除前后空格
        try:
            return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S.%f")
        except ValueError as e:
            # 尝试截断到微秒（保留前6位）
            if 'unconverted data remains' in str(e):
                # 找到小数点的位置
                dot_index = date_str.find('.')
                if dot_index != -1 and len(date_str) > dot_index + 7:
                    # 保留小数点后6位
                    date_str = date_str[:dot_index + 7]
                try:
                    return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S.%f")
                except ValueError:
                    # 如果仍然失败，尝试不带微秒
                    return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
            else:
                raise e

    def build_profiles(
        self,
        history_commits: List[Dict],
        test_commit_date: datetime,
        test_commit_id: int,
        test_project: str
    ) -> Tuple[Dict[int, Counter], Dict[int, datetime], Dict[int, int]]:
        """
        从历史提交记录中构建每个审查员的档案，仅包括与测试提交相同项目的历史提交。

        :param history_commits: 历史提交记录列表。
        :param test_commit_date: 测试提交的提交日期，用于计算衰减。
        :param test_commit_id: 测试提交的序号，用于基于ID的衰减。
        :param test_project: 测试提交所属的项目。
        :return:
            - 审查员ID到其路径段计数器的字典。
            - 审查员ID到其最后审查日期的字典。
            - 审查员ID到其最后审查提交ID的字典。
        """
        reviewer_profiles = defaultdict(Counter)          # 审查员ID -> 路径段计数器
        reviewer_last_review_date = dict()                # 审查员ID -> 最后审查日期
        reviewer_last_review_id = dict()                  # 审查员ID -> 最后审查提交ID

        for commit in history_commits:
            # 获取审查日期，优先使用'grant_date'，若无则使用'submit_date'
            approve_history = commit.get('approve_history', [])
            if approve_history:
                # 假设最后一个审查记录的'grant_date'是审查日期
                grant_date_str = approve_history[-1].get('grant_date')
            else:
                grant_date_str = commit.get('submit_date')

            if not grant_date_str:
                continue  # 若无日期信息，则跳过该提交

            try:
                commit_date = self.parse_datetime(grant_date_str)
            except ValueError:
                # 如果日期解析失败，则跳过该提交并记录日志
                print(f"日期解析失败: {grant_date_str}", file=RESULT_FILE)
                continue

            # 获取提交ID（使用'changeId'）
            commit_id = commit.get('changeId')
            if commit_id is None:
                continue

            # 计算时间差和提交数差
            d_days = (test_commit_date - commit_date).days
            d_ids = test_commit_id - commit_id

            # 计算衰减因子
            decay_factor = 1.0
            if self.decay_by_date:
                if d_days < 0:
                    decay_date = 1.0  # 未来的提交不衰减
                else:
                    decay_date = math.pow(self.f_date, d_days / self.l_date)
                decay_factor *= decay_date

            if self.decay_by_id:
                if d_ids < 0:
                    decay_id = 1.0  # 未来的提交不衰减
                else:
                    decay_id = math.pow(self.f_id, d_ids / self.l_id)
                decay_factor *= decay_id

            # 处理每个审查员
            for reviewer in approve_history:
                reviewer_id = reviewer.get('name')
                if reviewer_id is None:
                    continue  # 若无name，则跳过

                # 更新审查员的最后审查日期和提交ID
                if (reviewer_id not in reviewer_last_review_date) or (commit_date > reviewer_last_review_date[reviewer_id]):
                    reviewer_last_review_date[reviewer_id] = commit_date
                if (reviewer_id not in reviewer_last_review_id) or (commit_id > reviewer_last_review_id[reviewer_id]):
                    reviewer_last_review_id[reviewer_id] = commit_id

                # 处理文件路径
                for file_path in commit.get('files', []):
                    segments = self.tokenize_file_path(file_path)
                    for segment in segments:
                        reviewer_profiles[reviewer_id][segment] += decay_factor

        # 添加调试信息，打印部分审查员档案
        if reviewer_profiles:
            print(f"构建完成，审查员数量: {len(reviewer_profiles)}", file=RESULT_FILE)
            sample_reviewers = list(reviewer_profiles.keys())[:5]
            for reviewer_id in sample_reviewers:
                print(f"审查员ID: {reviewer_id}, 路径段计数: {dict(reviewer_profiles[reviewer_id])}")
        else:
            print("未构建任何审查员档案。", file=RESULT_FILE)

        return reviewer_profiles, reviewer_last_review_date, reviewer_last_review_id

    @staticmethod
    def compute_tversky_index(m_commit: Counter, profile: Counter, alpha: float = .0, beta: float = 1.0) -> float:
        """
        计算Tversky指数相似性。

        :param m_commit: 表示提交路径段的计数器。
        :param profile: 表示审查员档案的计数器。
        :param alpha: |X - Y|的权重。
        :param beta: |Y - X|的权重。
        :return: Tversky指数得分。
        """
        intersection = sum((m_commit & profile).values())
        difference_x = sum((m_commit - profile).values())
        difference_y = sum((profile - m_commit).values())
        denominator = intersection + alpha * difference_x + beta * difference_y
        if denominator == 0:
            return 0.0
        return intersection / denominator

    @staticmethod
    def compute_jaccard_index(m_commit: Counter, profile: Counter) -> float:
        """
        计算Jaccard系数相似性。

        :param m_commit: 表示提交路径段的计数器。
        :param profile: 表示审查员档案的计数器。
        :return: Jaccard系数得分。
        """
        intersection = sum((m_commit & profile).values())
        union = sum((m_commit | profile).values())
        if union == 0:
            return 0.0
        return intersection / union

    def recommend_reviewers(
        self,
        test_commit: Dict,
        reviewer_profiles: Dict[int, Counter],
        reviewer_last_review_date: Dict[int, datetime],
        reviewer_last_review_id: Dict[int, int],
        top_n: int = 5,
        similarity_metric: str = 'tversky'
    ) -> List[int]:
        """
        为测试提交推荐Top N审查员。

        :param test_commit: 测试提交的字典。
        :param reviewer_profiles: 审查员ID到其路径段计数器的字典。
        :param reviewer_last_review_date: 审查员ID到其最后审查日期的字典。
        :param reviewer_last_review_id: 审查员ID到其最后审查提交ID的字典。
        :param top_n: 推荐的审查员数量。
        :param similarity_metric: 相似性度量方法，'tversky'或'jaccard'。
        :return: 推荐的审查员ID列表。
        """
        m_commit = Counter()
        for file_path in test_commit.get('files', []):
            segments = self.tokenize_file_path(file_path)
            m_commit.update(segments)

        similarity_scores = dict()

        for reviewer_id, profile in reviewer_profiles.items():
            if similarity_metric == 'tversky':
                score = self.compute_tversky_index(m_commit, profile, alpha=0.0, beta=1.0)
            elif similarity_metric == 'jaccard':
                score = self.compute_jaccard_index(m_commit, profile)
            else:
                raise ValueError("Unsupported similarity metric.")
            similarity_scores[reviewer_id] = score

        # 如果没有审查员档案，返回空列表
        if not similarity_scores:
            print("没有找到任何匹配的审查员。", file=RESULT_FILE)
            return []

        # 根据相似性得分降序排序，若得分相同则根据最后审查日期或ID降序排序
        if self.decay_by_date and self.decay_by_id:
            # 同时基于日期和ID排序
            sorted_reviewers = sorted(
                similarity_scores.items(),
                key=lambda item: (
                    -item[1],
                    -reviewer_last_review_date.get(item[0], datetime.min).timestamp(),
                    -reviewer_last_review_id.get(item[0], 0)
                )
            )
        elif self.decay_by_date:
            # 仅基于日期排序
            sorted_reviewers = sorted(
                similarity_scores.items(),
                key=lambda item: (
                    -item[1],
                    -reviewer_last_review_date.get(item[0], datetime.min).timestamp()
                )
            )
        elif self.decay_by_id:
            # 仅基于ID排序
            sorted_reviewers = sorted(
                similarity_scores.items(),
                key=lambda item: (
                    -item[1],
                    -reviewer_last_review_id.get(item[0], 0)
                )
            )
        else:
            # 仅基于相似性得分排序
            sorted_reviewers = sorted(
                similarity_scores.items(),
                key=lambda item: -item[1]
            )

        # 获取推荐的审查员ID
        top_reviewers = [reviewer_id for reviewer_id, _ in sorted_reviewers[:top_n]]

        # 如果推荐的审查员少于top_n，从历史中推荐其他审查员（确保推荐列表长度固定）
        if len(top_reviewers) < top_n:
            # 获取所有审查员，按相似性得分降序排序
            all_sorted_reviewers = sorted(similarity_scores.items(), key=lambda item: -item[1])
            additional_reviewers = [reviewer_id for reviewer_id, _ in all_sorted_reviewers if reviewer_id not in top_reviewers]
            for reviewer_id in additional_reviewers:
                if len(top_reviewers) >= top_n:
                    break
                top_reviewers.append(reviewer_id)

        # 确保推荐列表长度为top_n或历史审查员数量
        top_reviewers = top_reviewers[:top_n]

        return top_reviewers

    def process_pair(
        self,
        test_file: str,
        history_file: str,
        similarity_metric: str = 'tversky'
    ) -> Tuple[int, List[int], List[int]]:
        """
        处理单个测试/历史文件对，推荐审查员并返回相关信息。

        :param test_file: 测试JSONL文件的路径。
        :param history_file: 历史JSONL文件的路径。
        :param similarity_metric: 相似性度量方法，'tversky'或'jaccard'。
        :return:
            - 测试ID。
            - 推荐的审查员ID列表（Top5）。
            - 实际的审查员ID列表。
        """
        # 从文件名中提取测试ID，假设格式为'0_test.jsonl'
        test_id = int(os.path.basename(test_file).split('_')[0])

        # 加载测试提交
        test_commits = self.load_jsonl(test_file)
        if not test_commits:
            print(f"在{test_file}中未找到数据。", file=RESULT_FILE)
            return test_id, [], []
        test_commit = test_commits[0]

        # 加载历史提交
        history_commits = self.load_jsonl(history_file)

        # 解析测试提交的提交日期
        submit_date_str = test_commit.get('submit_date')
        if not submit_date_str:
            print(f"在{test_file}中未找到submit_date。", file=RESULT_FILE)
            return test_id, [], []
        try:
            test_commit_date = self.parse_datetime(submit_date_str)
        except ValueError:
            print(f"测试提交的日期解析失败: {submit_date_str}，Test ID: {test_id}", file=RESULT_FILE)
            return test_id, [], []

        # 获取测试提交的ID（使用'changeId'）
        test_commit_id = test_commit.get('changeId')
        if test_commit_id is None:
            print(f"在{test_file}中未找到changeId。", file=RESULT_FILE)
            return test_id, [], []

        # 获取测试提交的项目
        test_project = test_commit.get('project')
        if not test_project:
            print(f"在{test_file}中未找到project。", file=RESULT_FILE)
            return test_id, [], []

        # 构建审查员档案，仅基于与测试提交相同项目的历史提交
        reviewer_profiles, reviewer_last_review_date, reviewer_last_review_id = self.build_profiles(
            history_commits,
            test_commit_date,
            test_commit_id,
            test_project
        )

        # 推荐审查员
        top_reviewers = self.recommend_reviewers(
            test_commit,
            reviewer_profiles,
            reviewer_last_review_date,
            reviewer_last_review_id,
            top_n=7,
            similarity_metric=similarity_metric
        )

        # 获取实际的审查员ID
        actual_reviewers = [reviewer.get('name') for reviewer in test_commit.get('approve_history', []) if reviewer.get('name') is not None]

        # 如果没有推荐审查员，尝试从历史审查员中推荐
        if not top_reviewers and reviewer_profiles:
            # 推荐历史上最活跃的审查员
            sorted_reviewers = sorted(
                reviewer_profiles.items(),
                key=lambda item: len(item[1]),
                reverse=True
            )
            top_reviewers = [reviewer_id for reviewer_id, _ in sorted_reviewers[:5]]

        return test_id, top_reviewers, actual_reviewers

def calculate_accuracy_at_n(recommended: List[int], actual: List[int], top_n: int) -> float:
    """
    计算Accuracy@N。

    :param recommended: 推荐的审查员ID列表。
    :param actual: 实际的审查员ID列表。
    :param top_n: N的值（1、3、5）。
    :return: Accuracy@N得分。
    """
    recommended_at_n = recommended[:top_n]
    actual_set = set(actual)
    recommended_set = set(recommended_at_n)
    intersection = recommended_set.intersection(actual_set)
    # 如果至少有一个实际审查员在推荐的前N名中，则视为正确
    return 1.0 if len(intersection) > 0 else 0.0

def calculate_mrr_at_n(recommended: List[int], actual: List[int], top_n: int) -> float:
    """
    计算MRR@N。

    :param recommended: 推荐的审查员ID列表。
    :param actual: 实际的审查员ID列表。
    :param top_n: N的值（1、3、5）。
    :return: MRR@N得分。
    """
    recommended_at_n = recommended[:top_n]
    for idx, reviewer_id in enumerate(recommended_at_n, start=1):
        if reviewer_id in actual:
            return 1.0 / idx
    return 0.0

def main():
    """
    主函数，处理所有测试/历史文件对，生成推荐结果和性能指标。
    """
    import argparse

    parser = argparse.ArgumentParser(description="基于个人档案的代码审查员推荐系统")
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/',
        help='包含测试和历史JSONL文件的目录。'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='profile_metrics.txt',
        help='输出性能指标的文本文件路径。'
    )
    parser.add_argument(
        '--similarity_metric',
        type=str,
        choices=['tversky', 'jaccard'],
        default='tversky',
        help='用于推荐的相似性度量方法。'
    )
    parser.add_argument(
        '--decay_by_date',
        action='store_true',
        help='基于时间对历史审查记录进行衰减。'
    )
    parser.add_argument(
        '--no_decay_by_date',
        dest='decay_by_date',
        action='store_false',
        help='不基于时间对历史审查记录进行衰减。'
    )
    parser.add_argument(
        '--decay_by_id',
        action='store_true',
        help='基于提交数对历史审查记录进行衰减。'
    )
    parser.add_argument(
        '--no_decay_by_id',
        dest='decay_by_id',
        action='store_false',
        help='不基于提交数对历史审查记录进行衰减。'
    )
    parser.set_defaults(decay_by_date=True, decay_by_id=True)
    args = parser.parse_args()

    data_dir = args.data_dir
    output_file = args.output_file
    similarity_metric = args.similarity_metric
    decay_by_date = args.decay_by_date
    decay_by_id = args.decay_by_id

    # 初始化推荐系统
    recommender = ReviewerRecommender(
        decay_by_date=decay_by_date,
        decay_by_id=decay_by_id
    )

    # 初始化性能指标
    metrics = {
        'accuracy@1': [],
        'accuracy@3': [],
        'accuracy@5': [],
        'mrr@1': [],
        'mrr@3': [],
        'mrr@5': []
    }

    # 打开输出文件
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # 写入标题
        outfile.write("TestID,Accuracy@1,Accuracy@3,Accuracy@5,MRR@1,MRR@3,MRR@5\n")

        # 处理测试/历史文件
        for i in range(1590):
            test_filename = f"{i}_test.jsonl"
            history_filename = f"{i}_history.jsonl"
            test_file = os.path.join(data_dir, test_filename)
            history_file = os.path.join(data_dir, history_filename)

            # 检查文件是否存在
            if not os.path.exists(test_file):
                print(f"测试文件 {test_file} 不存在，跳过。", file=RESULT_FILE)
                continue
            if not os.path.exists(history_file):
                print(f"历史文件 {history_file} 不存在，跳过。", file=RESULT_FILE)
                continue

            # 处理文件对
            test_id, top_reviewers, actual_reviewers = recommender.process_pair(
                test_file,
                history_file,
                similarity_metric=similarity_metric
            )

            # 计算指标
            accuracy_at_1 = calculate_accuracy_at_n(top_reviewers, actual_reviewers, 1)
            accuracy_at_3 = calculate_accuracy_at_n(top_reviewers, actual_reviewers, 3)
            accuracy_at_5 = calculate_accuracy_at_n(top_reviewers, actual_reviewers, 5)

            mrr_at_1 = calculate_mrr_at_n(top_reviewers, actual_reviewers, 1)
            mrr_at_3 = calculate_mrr_at_n(top_reviewers, actual_reviewers, 3)
            mrr_at_5 = calculate_mrr_at_n(top_reviewers, actual_reviewers, 5)

            # 存储指标
            metrics['accuracy@1'].append(accuracy_at_1)
            metrics['accuracy@3'].append(accuracy_at_3)
            metrics['accuracy@5'].append(accuracy_at_5)
            metrics['mrr@1'].append(mrr_at_1)
            metrics['mrr@3'].append(mrr_at_3)
            metrics['mrr@5'].append(mrr_at_5)

            # 打印单个测试案例的结果
            print(f"Test ID: {test_id}", file=RESULT_FILE)
            print(f"推荐审查员: {top_reviewers}", file=RESULT_FILE)
            print(f"实际审查员: {actual_reviewers}", file=RESULT_FILE)
            print(f"Accuracy@1: {accuracy_at_1:.4f}, Accuracy@3: {accuracy_at_3:.4f}, Accuracy@5: {accuracy_at_5:.4f}", file=RESULT_FILE)
            print(f"MRR@1: {mrr_at_1:.4f}, MRR@3: {mrr_at_3:.4f}, MRR@5: {mrr_at_5:.4f}", file=RESULT_FILE)
            print("-" * 50, file=RESULT_FILE)

            # 写入输出文件
            outfile.write(f"{test_id},{accuracy_at_1:.4f},{accuracy_at_3:.4f},{accuracy_at_5:.4f},"
                          f"{mrr_at_1:.4f},{mrr_at_3:.4f},{mrr_at_5:.4f}\n")

            # 每处理50对文件，输出进度
            if (i+1) % 50 == 0:
                print(f"已处理 {i+1}/500 对文件。", file=RESULT_FILE)

    # 计算整体平均指标
    avg_accuracy_at_1 = sum(metrics['accuracy@1']) / len(metrics['accuracy@1']) if metrics['accuracy@1'] else 0.0
    avg_accuracy_at_3 = sum(metrics['accuracy@3']) / len(metrics['accuracy@3']) if metrics['accuracy@3'] else 0.0
    avg_accuracy_at_5 = sum(metrics['accuracy@5']) / len(metrics['accuracy@5']) if metrics['accuracy@5'] else 0.0

    avg_mrr_at_1 = sum(metrics['mrr@1']) / len(metrics['mrr@1']) if metrics['mrr@1'] else 0.0
    avg_mrr_at_3 = sum(metrics['mrr@3']) / len(metrics['mrr@3']) if metrics['mrr@3'] else 0.0
    avg_mrr_at_5 = sum(metrics['mrr@5']) / len(metrics['mrr@5']) if metrics['mrr@5'] else 0.0

    # 打印整体平均指标
    print("整体平均指标：", file=RESULT_FILE)
    print(f"Average Accuracy@1: {avg_accuracy_at_1:.4f}", file=RESULT_FILE)
    print(f"Average Accuracy@3: {avg_accuracy_at_3:.4f}", file=RESULT_FILE)
    print(f"Average Accuracy@5: {avg_accuracy_at_5:.4f}", file=RESULT_FILE)
    print(f"Average MRR@1: {avg_mrr_at_1:.4f}", file=RESULT_FILE)
    print(f"Average MRR@3: {avg_mrr_at_3:.4f}", file=RESULT_FILE)
    print(f"Average MRR@5: {avg_mrr_at_5:.4f}", file=RESULT_FILE)
    print("整体平均指标：")
    print(f"Average Accuracy@1: {avg_accuracy_at_1:.4f}")
    print(f"Average Accuracy@3: {avg_accuracy_at_3:.4f}")
    print(f"Average Accuracy@5: {avg_accuracy_at_5:.4f}")
    print(f"Average MRR@1: {avg_mrr_at_1:.4f}")
    print(f"Average MRR@3: {avg_mrr_at_3:.4f}")
    print(f"Average MRR@5: {avg_mrr_at_5:.4f}")

    # 写入整体平均指标到输出文件
    with open(output_file, 'a', encoding='utf-8') as outfile:
        outfile.write("\n整体平均指标:\n")
        outfile.write(f"Average Accuracy@1: {avg_accuracy_at_1:.4f}\n")
        outfile.write(f"Average Accuracy@3: {avg_accuracy_at_3:.4f}\n")
        outfile.write(f"Average Accuracy@5: {avg_accuracy_at_5:.4f}\n")
        outfile.write(f"Average MRR@1: {avg_mrr_at_1:.4f}\n")
        outfile.write(f"Average MRR@3: {avg_mrr_at_3:.4f}\n")
        outfile.write(f"Average MRR@5: {avg_mrr_at_5:.4f}\n")

    print(f"所有推荐结果和指标已保存至 {output_file}。", file=RESULT_FILE)

if __name__ == "__main__":
    main()
