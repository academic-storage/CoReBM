import json
import random
import re
from collections import defaultdict
from datetime import datetime, timedelta
import os

# 加载数据函数
def load_data_from_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():
                data.append(json.loads(line))
    print(f"Loaded {len(data)} records from {file_path}")
    return data

# 解析日期字符串，转换成 datetime 对象
def parse_date(date_string):
    if not date_string:
        return None
    date_string = re.sub(r"(\.\d{6})\d+", r"\1", date_string)
    date_string = re.sub(r"\s+000$", "", date_string)
    try:
        return datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        try:
            return datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            print(f"无法解析的日期格式: {date_string}")
            return None

# 构建审查者专业知识模型（RE）
def build_expertise_model(history_data):
    print("Building expertise model...")
    reviewer_file_comments = defaultdict(lambda: {'frequency': 0, 'recency': None})
    for record in history_data:
        for reviewer in record.get('approve_history', []):
            user_id = reviewer['name']
            date = parse_date(reviewer.get('grant_date', ''))
            for file_path in record.get('files', []):
                key = (user_id, file_path)
                reviewer_file_comments[key]['frequency'] += 1
                if (reviewer_file_comments[key]['recency'] is None) or (
                        date and date > reviewer_file_comments[key]['recency']):
                    reviewer_file_comments[key]['recency'] = date
    print(f"Expertise model built with {len(reviewer_file_comments)} entries.")
    return reviewer_file_comments

# 计算审查者的专业知识得分 E(ri, Fc)
def calculate_reviewer_expertise(reviewer_ids, target_files, expertise_model, current_date):
    expertise_scores = {}
    for reviewer_id in reviewer_ids:
        E_total = 0
        for file_path in target_files:
            key = (reviewer_id, file_path)
            freq = expertise_model.get(key, {}).get('frequency', 0)
            recency_date = expertise_model.get(key, {}).get('recency', None)
            if recency_date and current_date:
                days_since_last = (current_date - recency_date).days
                total_days = (current_date - datetime(2000, 1, 1)).days
                recency_score = 1 - (days_since_last / total_days)
                recency_score = max(recency_score, 0)
            else:
                recency_score = 0
            Exp = freq * recency_score
            E_total += Exp
        expertise_scores[reviewer_id] = E_total
    return expertise_scores

# 构建审查者协作模型（RC）
def build_collaboration_graph(history_data):
    print("Building collaboration graph...")
    collaboration_graph = defaultdict(lambda: defaultdict(int))
    for record in history_data:
        reviewers = [reviewer['name'] for reviewer in record.get('approve_history', [])]
        owner = record.get('owner', {})
        developer_id = owner.get('accountId', None)
        if developer_id is not None:
            developer_id = int(developer_id)
            reviewers.append(developer_id)  # 在协作图中包括开发者
        for i in range(len(reviewers)):
            for j in range(i + 1, len(reviewers)):
                reviewer_i = reviewers[i]
                reviewer_j = reviewers[j]
                collaboration_graph[reviewer_i][reviewer_j] += 1
                collaboration_graph[reviewer_j][reviewer_i] += 1
    print(f"Collaboration graph built with {len(collaboration_graph)} reviewers.")
    return collaboration_graph

# 计算审查者的协作得分 RC(d, Rc)
def calculate_collaboration_score(developer_id, reviewer_ids, collaboration_graph):
    sub_graph_nodes = set(reviewer_ids)
    if developer_id is not None:
        sub_graph_nodes.add(developer_id)
    sub_graph_edges = 0
    edge_weights = 0
    node_list = list(sub_graph_nodes)
    for i in range(len(node_list)):
        for j in range(i + 1, len(node_list)):
            reviewer_i = node_list[i]
            reviewer_j = node_list[j]
            weight = collaboration_graph[reviewer_i].get(reviewer_j, 0)
            if weight > 0:
                sub_graph_edges += 1
                edge_weights += weight
    max_edges = len(sub_graph_nodes) * (len(sub_graph_nodes) - 1) / 2
    connectivity = sub_graph_edges / max_edges if max_edges > 0 else 0
    RC = connectivity * edge_weights
    return RC

# 构建审查者工作负载模型（RW）
def build_reviewer_workload(history_data):
    print("Building reviewer workload model...")
    reviewer_workload = defaultdict(int)
    for record in history_data:
        if record.get('status', '') != 'M':  # 非已合并或关闭的变更
            for reviewer in record.get('approve_history', []):
                user_id = reviewer['name']
                reviewer_workload[user_id] += 1
    print(f"Reviewer workload model built with {len(reviewer_workload)} reviewers.")
    return reviewer_workload

# 计算审查者的工作负载 RW(r)
def calculate_reviewer_workload(reviewer_ids, reviewer_workload):
    workloads = {}
    for reviewer_id in reviewer_ids:
        workloads[reviewer_id] = reviewer_workload.get(reviewer_id, 0)
    return workloads

# 初始化种群
def initialize_population(reviewer_ids, pop_size, min_size=1, max_size=5):
    print("Initializing population...")
    population = []
    reviewer_list = list(reviewer_ids)
    for _ in range(pop_size):
        if len(reviewer_list) >= min_size:
            individual = random.sample(reviewer_list, random.randint(min_size, min(max_size, len(reviewer_list))))
        else:
            individual = reviewer_list.copy()
        population.append(individual)
    print(f"Population initialized with {len(population)} individuals.")
    return population

# 交叉操作
def crossover(parent1, parent2, reviewer_ids, max_size=5):
    child = list(set(parent1 + parent2))
    if len(child) > max_size:
        child = random.sample(child, max_size)
    return child

# 变异操作
def mutation(individual, reviewer_ids, mutation_rate=0.5, min_size=1, max_size=5):
    if random.random() < mutation_rate:
        if len(individual) > min_size:
            removed = random.choice(individual)
            individual.remove(removed)
            print(f"Mutation: Removed reviewer {removed}")
    if random.random() < mutation_rate and len(individual) < max_size:
        candidate_ids = reviewer_ids - set(individual)
        if candidate_ids:
            added = random.choice(list(candidate_ids))
            individual.append(added)
            print(f"Mutation: Added reviewer {added}")
    return individual

# 适应度函数
def fitness_function(individual, developer_id, target_files, expertise_model, collaboration_graph, reviewer_workload,
                     current_date, alpha=0.76, beta=0.24):
    # 计算 REC
    expertise_scores = calculate_reviewer_expertise(individual, target_files, expertise_model, current_date)
    RE = sum(expertise_scores.values()) / len(individual) if len(individual) > 0 else 0
    RC = calculate_collaboration_score(developer_id, individual, collaboration_graph)
    REC = alpha * RE + beta * RC
    # 计算工作负载
    workloads = calculate_reviewer_workload(individual, reviewer_workload)
    RW = sum(workloads.values()) / len(individual) if len(individual) > 0 else 0
    return REC, RW

# 选择操作（基于适应度排序）
def selection(population, fitnesses):
    # 根据适应度值对种群进行排序，选择前半部分
    sorted_population = [x for _, x in
                         sorted(zip(fitnesses, population), key=lambda pair: (pair[0][0], -pair[0][1]), reverse=True)]
    selected = sorted_population[:len(population) // 2]
    print(f"Selection: Selected {len(selected)} individuals.")
    return selected

# 遗传算法主函数
def genetic_algorithm(record, history_data, generations=86, pop_size=70):
    print(f"Starting genetic algorithm for change {record.get('changeId', '')}...")
    current_date = parse_date(record.get('submit_date', ''))
    target_files = record.get('files', [])
    owner = record.get('owner', {})
    developer_id = owner.get('accountId', None)
    if developer_id is not None:
        developer_id = int(developer_id)
    else:
        print(f"记录 {record.get('changeId', '')} 缺少开发者信息，无法计算协作得分。")
        developer_id = None  # 或者设置一个默认值

    # 构建所有可能的审查者集合，只包含审查者，不包括开发者
    all_reviewer_ids = set()
    for r in history_data:
        for reviewer in r.get('approve_history', []):
            all_reviewer_ids.add(reviewer['name'])
        # 不再将所有者（开发者）ID 添加到 all_reviewer_ids 中

    # 构建潜在的评审人列表，确保不包含当前开发者
    reviewer_ids = all_reviewer_ids
    if developer_id is not None and developer_id in reviewer_ids:
        reviewer_ids.remove(developer_id)

    # 检查是否有可用的评审者
    if len(reviewer_ids) == 0:
        print("没有可用的审查者")
        return []

    expertise_model = build_expertise_model(history_data)
    collaboration_graph = build_collaboration_graph(history_data)
    reviewer_workload = build_reviewer_workload(history_data)

    population = initialize_population(reviewer_ids, pop_size)

    for gen in range(generations):
        print(f"Generation {gen + 1}/{generations}")
        fitnesses = []
        for idx, individual in enumerate(population):
            REC, RW = fitness_function(
                individual,
                developer_id,
                target_files,
                expertise_model,
                collaboration_graph,
                reviewer_workload,
                current_date
            )
            fitnesses.append((REC, RW))
            print(f"Individual {idx + 1}: REC={REC:.4f}, RW={RW:.4f}, Reviewers={individual}")
        # 选择操作
        selected_population = selection(population, fitnesses)

        # 生成新种群
        new_population = selected_population.copy()
        while len(new_population) < pop_size:
            parents = random.sample(selected_population, 2)
            child = crossover(parents[0], parents[1], reviewer_ids)
            child = mutation(child, reviewer_ids)
            new_population.append(child)
            print(f"Created new individual: {child}")

        population = new_population

    # 最终的 Pareto 前沿解
    print("Evaluating final population for Pareto front...")
    final_fitnesses = []
    for idx, individual in enumerate(population):
        REC, RW = fitness_function(
            individual,
            developer_id,
            target_files,
            expertise_model,
            collaboration_graph,
            reviewer_workload,
            current_date
        )
        final_fitnesses.append((REC, RW))
        print(f"Final Individual {idx + 1}: REC={REC:.4f}, RW={RW:.4f}, Reviewers={individual}")

    pareto_front = []
    for i in range(len(population)):
        dominated = False
        for j in range(len(population)):
            if i != j:
                if (final_fitnesses[j][0] >= final_fitnesses[i][0] and final_fitnesses[j][1] <= final_fitnesses[i][1]) and \
                        (final_fitnesses[j][0] > final_fitnesses[i][0] or final_fitnesses[j][1] < final_fitnesses[i][1]):
                    dominated = True
                    break
        if not dominated:
            pareto_front.append(population[i])
    print(f"Pareto front contains {len(pareto_front)} individuals.")

    # 统计审查者出现的次数
    reviewer_counts = defaultdict(int)
    for individual in pareto_front:
        for reviewer_id in individual:
            reviewer_counts[reviewer_id] += 1

    # 按出现次数排序，得到最终的推荐列表
    sorted_reviewers = sorted(reviewer_counts.items(), key=lambda x: x[1], reverse=True)
    recommended_reviewers = [reviewer_id for reviewer_id, count in sorted_reviewers]

    print(f"Recommended reviewers: {recommended_reviewers}")
    return recommended_reviewers

# 评估模型
def evaluate_model(test_data, history_data, file_index, total_accs, total_mrrs, total_cases):
    for idx in range(len(test_data)):
        record = test_data[idx]
        print(f"\nProcessing change {record.get('changeId', '')} (File Index {file_index}, Test Index {idx})...")
        recommended_reviewers = genetic_algorithm(record, history_data)

        actual_reviewers = [reviewer['name'] for reviewer in record.get('approve_history', [])]
        print(f"Actual reviewers: {actual_reviewers}")

        accs = {}
        mrrs = {}
        for k in [1, 3, 5]:
            top_k_reviewers = recommended_reviewers[:k]
            hit = any(reviewer in top_k_reviewers for reviewer in actual_reviewers)
            acc = 1 if hit else 0
            accs[k] = acc
            ranks = [top_k_reviewers.index(reviewer) + 1 for reviewer in actual_reviewers if
                     reviewer in top_k_reviewers]
            mrr = 1 / min(ranks) if ranks else 0
            mrrs[k] = mrr
            print(f"ACC@{k}: {'Hit' if hit else 'Miss'}, MRR@{k}: {mrr}")

            # 累计 ACC 和 MRR
            total_accs[k] += acc
            total_mrrs[k] += mrr

        total_cases += 1

        # 将结果写入到列表中
        output_line = f"{file_index}-{idx}\t{recommended_reviewers}\t{actual_reviewers}\t{accs[1]}\t{accs[3]}\t{accs[5]}\t{mrrs[1]:.4f}\t{mrrs[3]:.4f}\t{mrrs[5]:.4f}"
        with open('whoreview_evaluation_results.txt', 'a', encoding='utf-8') as f:
            f.write(output_line + '\n')

    return total_cases

def main():
    # 初始化累计的 ACC 和 MRR
    total_accs = {1: 0, 3: 0, 5: 0}
    total_mrrs = {1: 0.0, 3: 0.0, 5: 0.0}
    total_cases = 0

    # 如果结果文件已存在，删除它
    if os.path.exists('whoreview_evaluation_results.txt'):
        os.remove('whoreview_evaluation_results.txt')

    # 循环处理文件编号
    for file_index in range(0, 1590):
        test_file_path = f"data/{file_index}_test.jsonl"
        history_file_path = f"data/{file_index}_history.jsonl"

        # 检查文件是否存在
        if not os.path.exists(test_file_path) or not os.path.exists(history_file_path):
            print(f"Files for index {file_index} do not exist. Skipping.")
            continue

        # 加载测试数据和历史数据
        test_data = load_data_from_jsonl(test_file_path)
        history_data = load_data_from_jsonl(history_file_path)

        # 评估模型
        total_cases = evaluate_model(test_data, history_data, file_index, total_accs, total_mrrs, total_cases)

    # 计算平均 ACC 和 MRR
    avg_accs = {k: total_accs[k] / total_cases if total_cases > 0 else 0 for k in total_accs}
    avg_mrrs = {k: total_mrrs[k] / total_cases if total_cases > 0 else 0 for k in total_mrrs}

    # 输出汇总结果
    summary_line = f"Total Cases: {total_cases}\n" \
                   f"ACC@1: {avg_accs[1]:.4f}, MRR@1: {avg_mrrs[1]:.4f}\n" \
                   f"ACC@3: {avg_accs[3]:.4f}, MRR@3: {avg_mrrs[3]:.4f}\n" \
                   f"ACC@5: {avg_accs[5]:.4f}, MRR@5: {avg_mrrs[5]:.4f}\n"

    print("\nSummary of ACC and MRR:")
    print(summary_line)

    # 将汇总结果写入到文件
    with open('whoreview_evaluation_summary.txt', 'w', encoding='utf-8') as f:
        f.write(summary_line)

if __name__ == "__main__":
    main()
