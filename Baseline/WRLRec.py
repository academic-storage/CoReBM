import json
import os
import math
import random
import concurrent.futures
from collections import defaultdict
from deap import base, creator, tools, algorithms

# 设置随机种子以确保结果可重复
random.seed(42)

# 数据集文件夹路径
DATA_FOLDER_DIR = 'data'

# 定义适应度和个体
# 多目标优化：最大化CPR（权重为1.0），最小化SRW（权重为-1.0）
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)


# 工具函数：提取模块目录
def get_modules(files):
    modules = set()
    for file in files:
        module = os.path.dirname(file)
        modules.add(module)
    return modules


# 计算代码所有权（CO）
def compute_CO(reviewer, test_modules, history_patches):
    co_sum = 0
    for m in test_modules:
        total_reviews_in_m = sum(1 for patch in history_patches if m in patch['files'])
        if total_reviews_in_m == 0:
            continue
        authored_reviews = sum(1 for patch in history_patches if
                               m in patch['files'] and patch['owner']['accountId'] == reviewer['accountId'])
        co_sum += authored_reviews / total_reviews_in_m
    if len(test_modules) == 0:
        return 0
    return co_sum / len(test_modules)


# 计算评审经验（RE）
def compute_RE(reviewer, test_modules, history_patches):
    re_sum = 0
    for m in test_modules:
        total_reviews_in_m = sum(1 for patch in history_patches if m in patch['files'])
        if total_reviews_in_m == 0:
            continue
        reviewed_reviews = sum(1 for patch in history_patches if m in patch['files'] and any(
            r['name'] == reviewer['name'] for r in patch['approve_history']))
        re_sum += reviewed_reviews / total_reviews_in_m
    if len(test_modules) == 0:
        return 0
    return re_sum / len(test_modules)


# 计算与补丁作者的熟悉度（FPA）
def compute_FPA(reviewer, test_owner_id, history_patches):
    count = 0
    for patch in history_patches:
        if patch['owner']['accountId'] == test_owner_id and any(
                r['name'] == reviewer['name'] for r in patch['approve_history']):
            count += 1
    return count


# 计算评审参与率（RPR）
def compute_RPR(reviewer, history_patches):
    total_requests = len(history_patches)
    participated = sum(
        1 for patch in history_patches if any(r['name'] == reviewer['name'] for r in patch['approve_history']))
    if total_requests == 0:
        return 0
    return participated / total_requests


# 计算评审工作负载偏斜度（SRW）使用Shannon熵
def compute_SRW(selected_reviewers, reviewers_metrics):
    rr_values = [reviewers_metrics[r]['RR'] for r in selected_reviewers]
    rr_total = sum(rr_values)
    if rr_total == 0:
        return 0
    entropy = 0
    for rr in rr_values:
        h_r = rr / rr_total if rr_total > 0 else 0
        entropy += h_r * math.log2(h_r) if h_r > 0 else 0
    normalized_entropy = entropy / math.log2(len(selected_reviewers)) if len(selected_reviewers) > 1 else 0
    return normalized_entropy


# 加载JSON Lines文件，指定编码为utf-8
def load_jsonl(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data


# 计算MRR
def compute_mrr(recommended, actual, k):
    for rank, reviewer in enumerate(recommended[:k], start=1):
        if reviewer['name'] in actual:
            return 1.0 / rank
    return 0.0


# 自定义交叉函数，确保交叉后个体唯一性
def cxTwoPointUnique(ind1, ind2):
    tools.cxTwoPoint(ind1, ind2)
    # 修复重复
    set1 = set(ind1)
    set2 = set(ind2)
    if len(set1) < len(ind1):
        missing = list(set(ind2) - set1)
        for i, gene in enumerate(ind1):
            if ind1.count(gene) > 1:
                if missing:
                    ind1[i] = missing.pop()
                else:
                    break
    if len(set2) < len(ind2):
        missing = list(set(ind1) - set2)
        for i, gene in enumerate(ind2):
            if ind2.count(gene) > 1:
                if missing:
                    ind2[i] = missing.pop()
                else:
                    break
    return ind1, ind2


# 自定义变异操作，确保唯一性
def unique_mutate(individual, candidate_size, indpb=0.1):
    if candidate_size > len(individual):
        # 当候选数量大于个体长度时，进行基因替换，确保唯一性
        for i in range(len(individual)):
            if random.random() < indpb:
                available_genes = list(set(range(candidate_size)) - set(individual))
                if available_genes:
                    new_gene = random.choice(available_genes)
                    individual[i] = new_gene
    else:
        # 当候选数量等于个体长度时，进行基因交换，保持唯一性
        if random.random() < indpb:
            a, b = random.sample(range(len(individual)), 2)
            individual[a], individual[b] = individual[b], individual[a]
    return (individual,)


# 定义单个测试案例的处理函数
def process_case(i):
    print(f"开始处理测试案例 {i}")
    test_file = os.path.join(DATA_FOLDER_DIR, f"{i}_test.jsonl")
    history_file = os.path.join(DATA_FOLDER_DIR, f"{i}_history.jsonl")

    if not os.path.exists(test_file) or not os.path.exists(history_file):
        print(f"缺少文件：{i}_test.jsonl 或 {i}_history.jsonl")
        return (i, 0, 0, 0, 0.0, 0.0, 0.0, [])

    # 加载测试数据和历史数据
    try:
        print(f"加载测试数据和历史数据 for Test Case {i}")
        test_data = load_jsonl(test_file)[0]
        history_data = load_jsonl(history_file)
        print(f"成功加载数据 for Test Case {i}")
    except UnicodeDecodeError as e:
        print(f"文件编码错误在 {i}_history.jsonl 或 {i}_test.jsonl: {e}")
        return (i, 0, 0, 0, 0.0, 0.0, 0.0, [])
    except json.JSONDecodeError as e:
        print(f"JSON解析错误在 {i}_history.jsonl 或 {i}_test.jsonl: {e}")
        return (i, 0, 0, 0, 0.0, 0.0, 0.0, [])

    # 提取测试补丁信息
    test_files = test_data.get('files', [])
    test_modules = get_modules(test_files)
    test_owner = test_data.get('owner', {})
    test_owner_id = test_owner.get('accountId', None)

    if test_owner_id is None:
        print(f"测试案例 {i} 缺少补丁作者信息")
        return (i, 0, 0, 0, 0.0, 0.0, 0.0, [])

    # 提取历史补丁
    history_patches = history_data  # 列表形式

    # 确定候选评审者：至少审阅过一个历史补丁
    print(f"确定候选评审者 for Test Case {i}")
    candidate_reviewers = {}
    for patch in history_patches:
        for reviewer in patch.get('approve_history', []):
            user_id = reviewer.get('name', None)
            if user_id is None:
                continue
            if user_id not in candidate_reviewers:
                candidate_reviewers[user_id] = {
                    'name': reviewer['name'],
                    'email': reviewer.get('email', ''),
                    'name': reviewer.get('name', ''),
                    'accountId': reviewer['name']  # 假设name即accountId
                }
    # 移除补丁作者（如果在候选者中）
    if test_owner_id in candidate_reviewers:
        del candidate_reviewers[test_owner_id]

    candidate_list = list(candidate_reviewers.keys())
    candidate_size = len(candidate_list)
    print(f"测试案例 {i} 的候选评审者数量: {candidate_size}")

    if candidate_size == 0:
        print(f"测试案例 {i} 没有可用的候选评审者")
        return (i, 0, 0, 0, 0.0, 0.0, 0.0, [])

    # 计算每个候选评审者的指标
    print(f"计算每个候选评审者的指标 for Test Case {i}")
    reviewers_metrics = {}
    for user_id in candidate_list:
        reviewer = candidate_reviewers[user_id]
        co = compute_CO(reviewer, test_modules, history_patches)
        re = compute_RE(reviewer, test_modules, history_patches)
        fpa = compute_FPA(reviewer, test_owner_id, history_patches)
        rpr = compute_RPR(reviewer, history_patches)
        rr = 2 - rpr  # RR' = 2 - RPR
        reviewers_metrics[user_id] = {
            'CO': co,
            'RE': re,
            'FPA': fpa,
            'RPR': rpr,
            'RR': rr
        }

    if candidate_size < 5:
        print(f"测试案例 {i} 的候选评审者少于5，直接推荐所有候选评审者")
        selected_reviewers_ids = candidate_list  # 推荐所有候选者
        # 计算CPR
        reviewer_cpr = {
            r_id: 0.25 * reviewers_metrics[r_id]['CO'] +
                  0.25 * reviewers_metrics[r_id]['RE'] +
                  0.25 * reviewers_metrics[r_id]['FPA'] +
                  0.25 * reviewers_metrics[r_id]['RPR']
            for r_id in selected_reviewers_ids
        }
        # 按照CPR贡献排序
        selected_reviewers_sorted = sorted(selected_reviewers_ids, key=lambda r: reviewer_cpr[r], reverse=True)

        # 获取实际评审者
        actual_reviewers = set(r['name'] for r in test_data.get('approve_history', []))

        # 计算评价指标
        acc1 = 1 if any(r_id in actual_reviewers for r_id in selected_reviewers_sorted[:1]) else 0
        acc3 = 1 if any(r_id in actual_reviewers for r_id in selected_reviewers_sorted[:3]) else 0
        acc5 = 1 if any(r_id in actual_reviewers for r_id in selected_reviewers_sorted[:5]) else 0

        mrr1 = compute_mrr(recommended=[candidate_reviewers[r_id] for r_id in selected_reviewers_sorted],
                           actual=actual_reviewers, k=1)
        mrr3 = compute_mrr(recommended=[candidate_reviewers[r_id] for r_id in selected_reviewers_sorted],
                           actual=actual_reviewers, k=3)
        mrr5 = compute_mrr(recommended=[candidate_reviewers[r_id] for r_id in selected_reviewers_sorted],
                           actual=actual_reviewers, k=5)

        print(f"完成处理测试案例 {i}")
        # 返回推荐评审者列表
        return (i, acc1, acc3, acc5, mrr1, mrr3, mrr5, selected_reviewers_sorted)

    # 如果候选评审者数量 >=5，则使用遗传算法进行优化推荐
    # DEAP工具箱设置
    print(f"设置DEAP工具箱 for Test Case {i}")
    toolbox = base.Toolbox()
    # 个体生成函数：随机选择5个唯一的评审者索引
    toolbox.register("attr_reviewer", random.sample, range(candidate_size), 5)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_reviewer)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # 评估函数
    def evaluate(individual):
        selected_reviewers = [candidate_list[idx] for idx in individual]
        # 计算CPR
        cpr = 0
        for r_id in selected_reviewers:
            metrics = reviewers_metrics[r_id]
            cpr += 0.25 * metrics['CO'] + 0.25 * metrics['RE'] + 0.25 * metrics['FPA'] + 0.25 * metrics['RPR']
        # 计算SRW
        srw = compute_SRW(selected_reviewers, reviewers_metrics)
        return (cpr, srw)

    toolbox.register("evaluate", evaluate)
    # 使用自定义的cxTwoPointUnique
    toolbox.register("mate", cxTwoPointUnique)
    # 自定义变异操作，确保唯一性
    toolbox.register("mutate", unique_mutate, candidate_size=candidate_size, indpb=0.1)
    toolbox.register("select", tools.selNSGA2)

    # 初始化种群
    print(f"初始化种群 for Test Case {i}")
    pop = toolbox.population(n=100)

    # 运行NSGA-II算法
    print(f"运行遗传算法 for Test Case {i}")
    NGEN = 10  # 调试时减少代数，加快速度
    MU = 50  # 调整种群大小
    LAMBDA = 50
    CXPB = 0.9
    MUTPB = 0.1

    try:
        algorithms.eaMuPlusLambda(pop, toolbox, mu=MU, lambda_=LAMBDA, cxpb=CXPB, mutpb=MUTPB, ngen=NGEN, verbose=False)
        print(f"遗传算法完成 for Test Case {i}")
    except Exception as e:
        print(f"遗传算法运行时出错在测试案例 {i}: {e}")
        return (i, 0, 0, 0, 0.0, 0.0, 0.0, [])

    # 提取Pareto前沿
    print(f"提取Pareto前沿 for Test Case {i}")
    pareto_front = tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]

    if not pareto_front:
        print(f"测试案例 {i} 无Pareto前沿解")
        return (i, 0, 0, 0, 0.0, 0.0, 0.0, [])

    # 确定膝点
    print(f"确定膝点 for Test Case {i}")
    cpr_max = max(ind.fitness.values[0] for ind in pareto_front)
    srw_min = min(ind.fitness.values[1] for ind in pareto_front)

    best_ind = None
    best_dist = float('inf')
    for ind in pareto_front:
        cpr, srw = ind.fitness.values
        dist = math.sqrt((cpr_max - cpr) ** 2 + (srw_min - srw) ** 2)
        if dist < best_dist:
            best_dist = dist
            best_ind = ind
    if best_ind is None:
        print(f"测试案例 {i} 无法确定膝点解")
        return (i, 0, 0, 0, 0.0, 0.0, 0.0, [])

    # 获取推荐的评审者ID
    selected_reviewers_ids = [candidate_list[idx] for idx in best_ind]
    # 计算每个评审者的单独CPR贡献，以便排序
    reviewer_cpr = {
        r_id: 0.25 * reviewers_metrics[r_id]['CO'] +
              0.25 * reviewers_metrics[r_id]['RE'] +
              0.25 * reviewers_metrics[r_id]['FPA'] +
              0.25 * reviewers_metrics[r_id]['RPR']
        for r_id in selected_reviewers_ids
    }
    # 按照CPR贡献排序
    selected_reviewers_sorted = sorted(selected_reviewers_ids, key=lambda r: reviewer_cpr[r], reverse=True)
    recommended_reviewers = [candidate_reviewers[r_id] for r_id in selected_reviewers_sorted]

    # 获取实际评审者
    actual_reviewers = set(r['name'] for r in test_data.get('approve_history', []))

    # 计算评价指标
    acc1 = 1 if any(r['name'] in actual_reviewers for r in recommended_reviewers[:1]) else 0
    acc3 = 1 if any(r['name'] in actual_reviewers for r in recommended_reviewers[:3]) else 0
    acc5 = 1 if any(r['name'] in actual_reviewers for r in recommended_reviewers[:5]) else 0

    mrr1 = compute_mrr(recommended_reviewers, actual_reviewers, 1)
    mrr3 = compute_mrr(recommended_reviewers, actual_reviewers, 3)
    mrr5 = compute_mrr(recommended_reviewers, actual_reviewers, 5)

    print(f"完成处理测试案例 {i}")
    # 返回推荐评审者列表
    return (i, acc1, acc3, acc5, mrr1, mrr3, mrr5, selected_reviewers_sorted)


# 主函数
def main():
    # 打开结果文件
    with open('wrlrec_results.txt', 'w', encoding='utf-8') as result_file:
        # 写入标题
        result_file.write("Test Case, Accuracy@1, Accuracy@3, Accuracy@5, MRR@1, MRR@3, MRR@5, Recommended Reviewers\n")

        # 初始化汇总指标
        total_acc1 = 0
        total_acc3 = 0
        total_acc5 = 0
        total_mrr1 = 0
        total_mrr3 = 0
        total_mrr5 = 0
        total_cases = 0

        # 使用多进程加速处理
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:  # 限制最大进程数，防止资源耗尽
            # 提交所有任务
            futures = {executor.submit(process_case, i): i for i in range(1590)}  # 调试时先处理500个

            for future in concurrent.futures.as_completed(futures):
                i = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    print(f"测试案例 {i} 生成异常: {exc}")
                    continue
                if result is None:
                    continue
                i, acc1, acc3, acc5, mrr1, mrr3, mrr5, recommended_reviewers_sorted = result

                # 获取推荐评审者的字符串表示（例如，使用 name 逗号分隔）
                recommended_reviewers_str = ';'.join(map(str, recommended_reviewers_sorted))

                # 写入每个测试案例的指标和推荐评审者列表
                result_file.write(
                    f"{i}\t{acc1}\t{acc3}\t{acc5}\t{mrr1:.4f}\t{mrr3:.4f}\t{mrr5:.4f}\t{recommended_reviewers_str}\n")

                # 更新汇总指标
                total_acc1 += acc1
                total_acc3 += acc3
                total_acc5 += acc5
                total_mrr1 += mrr1
                total_mrr3 += mrr3
                total_mrr5 += mrr5
                total_cases += 1

                # 输出推荐结果（包括推荐评审者列表）
                print(
                    f"测试案例 {i} 的评价指标: Accuracy@1={acc1}, Accuracy@3={acc3}, Accuracy@5={acc5}, MRR@1={mrr1:.4f}, MRR@3={mrr3:.4f}, MRR@5={mrr5:.4f}")
                print(f"推荐的评审者列表: {recommended_reviewers_sorted}")

    # 计算并写入汇总指标
    if total_cases > 0:
        avg_acc1 = total_acc1 / total_cases
        avg_acc3 = total_acc3 / total_cases
        avg_acc5 = total_acc5 / total_cases
        avg_mrr1 = total_mrr1 / total_cases
        avg_mrr3 = total_mrr3 / total_cases
        avg_mrr5 = total_mrr5 / total_cases

        with open('wrlrec_results.txt', 'a', encoding='utf-8') as result_file_append:
            result_file_append.write("\n汇总结果:\n")
            result_file_append.write(f"总测试案例数: {total_cases}\n")
            result_file_append.write(f"Accuracy@1: {avg_acc1:.4f}\n")
            result_file_append.write(f"Accuracy@3: {avg_acc3:.4f}\n")
            result_file_append.write(f"Accuracy@5: {avg_acc5:.4f}\n")
            result_file_append.write(f"MRR@1: {avg_mrr1:.4f}\n")
            result_file_append.write(f"MRR@3: {avg_mrr3:.4f}\n")
            result_file_append.write(f"MRR@5: {avg_mrr5:.4f}\n")

        # 在控制台输出汇总结果
        print("\n汇总结果:")
        print(f"总测试案例数: {total_cases}")
        print(f"Accuracy@1: {avg_acc1:.4f}")
        print(f"Accuracy@3: {avg_acc3:.4f}")
        print(f"Accuracy@5: {avg_acc5:.4f}")
        print(f"MRR@1: {avg_mrr1:.4f}")
        print(f"MRR@3: {avg_mrr3:.4f}")
        print(f"MRR@5: {avg_mrr5:.4f}")
    else:
        with open('wrlrec_results.txt', 'a', encoding='utf-8') as result_file_append:
            result_file_append.write("没有有效的测试案例。\n")
        print("没有有效的测试案例。")


if __name__ == "__main__":
    main()
