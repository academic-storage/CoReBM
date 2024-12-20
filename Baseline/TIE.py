import json
import os
import glob
import datetime
import math
from collections import defaultdict, Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import itertools

# 确保下载了NLTK所需资源
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# 设置数据集目录路径
DATASET_DIR = 'data/'

# 设置过程输出文件
RESULT_FILE = open("tie_results.txt", "w", encoding="utf-8")

# 设置超参数搜索空间
M_VALUES = [50]  # 时间窗口天数
LAMBDA_VALUES = [0.1]  # 组合权重

# 初始化词干提取器和停用词
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# 在预处理函数中，指定语言参数
def preprocess_text(text):
    """
    预处理文本：分词、去停用词、词干化。
    """
    text = re.sub(r'[^a-zA-Z\s]', ' ', text).lower()  # 移除非字母字符并转换为小写
    tokens = nltk.word_tokenize(text, language='english')  # 使用nltk的word_tokenize
    filtered_tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]  # 词干提取
    return filtered_tokens

def read_jsonl_file(filepath):
    """
    读取JSONL文件，返回数据列表。
    """
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
    except Exception as e:
        print(f"Error reading file {filepath}: {e}", file=RESULT_FILE)
    return data

def build_text_mining_model(history_data):
    """
    构建文本挖掘模型，使用朴素贝叶斯多项式分类器。
    """
    term_freqs = defaultdict(Counter)  # 构建词汇表，计算每个审查者的词频
    doc_counts = defaultdict(int)
    total_docs = 0

    print("Building text mining model...", file=RESULT_FILE)
    for idx, entry in enumerate(history_data, start=1):
        reviewers = [approver['name'] for approver in entry.get('approve_history', [])]
        text = entry.get('subject', '') + ' ' + entry.get('message', '')
        tokens = preprocess_text(text)
        token_counts = Counter(tokens)

        for reviewer in reviewers:
            term_freqs[reviewer].update(token_counts)
            doc_counts[reviewer] += 1
        total_docs += 1

        if idx % 1000 == 0:
            print(f"Processed {idx} entries for text mining model.", file=RESULT_FILE)

    print(f"Total documents processed for text mining model: {total_docs}", file=RESULT_FILE)

    # 计算先验概率和条件概率
    priors = {}
    cond_probs = {}
    vocab = set()
    for reviewer in term_freqs:
        priors[reviewer] = doc_counts[reviewer] / total_docs
        cond_probs[reviewer] = {}
        total_terms = sum(term_freqs[reviewer].values())
        for term in term_freqs[reviewer]:
            cond_probs[reviewer][term] = (term_freqs[reviewer][term] + 1) / (total_terms + len(term_freqs[reviewer]))
            vocab.add(term)

    print("Text mining model built successfully.", file=RESULT_FILE)
    return priors, cond_probs, vocab

def predict_text_mining(priors, cond_probs, vocab, test_entry):
    """
    使用文本挖掘模型预测审查者。
    """
    text = test_entry.get('subject', '') + ' ' + test_entry.get('message', '')
    tokens = preprocess_text(text)
    token_counts = Counter(tokens)

    scores = {}
    for reviewer in priors:
        score = math.log(priors[reviewer])
        for term in token_counts:
            if term in cond_probs[reviewer]:
                score += token_counts[term] * math.log(cond_probs[reviewer][term])
            else:
                score += token_counts[term] * math.log(
                    1 / (sum(cond_probs[reviewer].values()) + len(vocab)))  # 拉普拉斯平滑处理未见过的词
        scores[reviewer] = score

    return scores

def parse_datetime(date_str):
    """
    解析日期字符串，处理纳秒级别，转换为微秒。
    """
    try:
        if '.' in date_str:
            main_part, nanoseconds = date_str.split('.')
            microseconds = nanoseconds[:6]  # 取前6位，转换为微秒
            date_str = f"{main_part}.{microseconds}"
        return datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S.%f')
    except ValueError as ve:
        print(f"Error parsing date string '{date_str}': {ve}", file=RESULT_FILE)
        return None

def build_similarity_model(history_data, test_entry, M):
    """
    构建相似度模型，比较过去M天内的文件路径。
    """
    test_date = parse_datetime(test_entry['submit_date'])
    if not test_date:
        print(f"Invalid test_date in entry: {test_entry.get('changeId', 'Unknown')}", file=RESULT_FILE)
        return defaultdict(float)
    past_date = test_date - datetime.timedelta(days=M)

    test_paths = test_entry.get('files', [])
    test_path_tokens = [set(path.split('/')) for path in test_paths]

    reviewer_scores = defaultdict(float)

    print(f"Building similarity model for Change ID: {test_entry['changeId']} with M={M}", file=RESULT_FILE)

    for idx, entry in enumerate(history_data, start=1):
        entry_date = parse_datetime(entry['submit_date'])
        if not entry_date:
            print(f"Invalid entry_date in history entry: {entry.get('changeId', 'Unknown')}", file=RESULT_FILE)
            continue
        if not (past_date <= entry_date <= test_date):
            continue  # 跳过时间窗口外的条目

        entry_paths = entry.get('files', [])
        entry_path_tokens = [set(path.split('/')) for path in entry_paths]

        # 计算文件路径之间的相似度
        similarities = []
        for test_tokens in test_path_tokens:
            for entry_tokens in entry_path_tokens:
                common_tokens = test_tokens & entry_tokens
                max_len = max(len(test_tokens), len(entry_tokens))
                if max_len > 0:
                    similarity = len(common_tokens) / max_len
                    similarities.append(similarity)

        if similarities:
            avg_similarity = sum(similarities) / len(similarities)
            reviewers = [approver['name'] for approver in entry.get('approve_history', [])]
            for reviewer in reviewers:
                reviewer_scores[reviewer] += avg_similarity

        if idx % 1000 == 0:
            print(f"Processed {idx} history entries for similarity model.", file=RESULT_FILE)

    print(f"Similarity model built for Change ID: {test_entry['changeId']}", file=RESULT_FILE)
    return reviewer_scores

def normalize_scores(scores):
    """
    归一化得分，使其总和为1。
    """
    total_score = sum(scores.values())
    if total_score == 0:
        return scores
    normalized_scores = {k: v / total_score for k, v in scores.items()}
    return normalized_scores

def combine_scores(text_scores, path_scores, lambda_value):
    """
    使用TIEComposer组合文本挖掘模型和相似度模型的得分。
    """
    all_reviewers = set(text_scores.keys()) | set(path_scores.keys())
    combined_scores = {}
    for reviewer in all_reviewers:
        text_score = text_scores.get(reviewer, 0)
        path_score = path_scores.get(reviewer, 0)
        combined_score = lambda_value * text_score + (1 - lambda_value) * path_score
        combined_scores[reviewer] = combined_score
    return combined_scores

def recommend_reviewers(combined_scores, top_k=5):
    """
    根据组合得分推荐前K名审查者。
    """
    sorted_reviewers = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)
    recommended_reviewers = [email for email, score in sorted_reviewers[:top_k]]
    return recommended_reviewers

def compute_top_k_accuracy(recommended_reviewers, actual_reviewers_emails, k=5):
    """
    计算单个测试条目的Top-K准确率。
    """
    top_k_recommendations = recommended_reviewers[:k]
    for reviewer in top_k_recommendations:
        if reviewer in actual_reviewers_emails:
            return 1  # 在Top K中正确推荐
    return 0  # Top K中没有正确推荐

def compute_mrr_at_k(recommended_reviewers, actual_reviewers_emails, k=5):
    """
    计算单个测试条目的MRR@K值。
    """
    top_k_recommendations = recommended_reviewers[:k]
    for idx, reviewer in enumerate(top_k_recommendations, start=1):
        if reviewer in actual_reviewers_emails:
            return 1 / idx
    return 0

def evaluate_model(M, LAMBDA, test_files):
    """
    评估模型在给定超参数M和LAMBDA下的性能。
    """
    test_results = []

    print(f"\nEvaluating model with M={M} and LAMBDA={LAMBDA}", file=RESULT_FILE)
    for idx, test_file in enumerate(test_files, start=1):
        print(f"\nProcessing test file {idx}/{len(test_files)}: {test_file}", file=RESULT_FILE)
        test_data = read_jsonl_file(test_file)
        if not test_data:
            print(f"No data found in {test_file}. Skipping.", file=RESULT_FILE)
            continue  # 跳过空文件
        test_entry = test_data[0]

        # 对应的历史文件
        history_file = test_file.replace('_test.jsonl', '_history.jsonl')
        history_data = read_jsonl_file(history_file)
        if not history_data:
            print(f"No history data found for {test_file}. Skipping.", file=RESULT_FILE)
            continue  # 如果没有历史数据，跳过

        # 使用历史数据构建模型
        priors, cond_probs, vocab = build_text_mining_model(history_data)

        # 预测文本模型得分
        text_scores = predict_text_mining(priors, cond_probs, vocab, test_entry)
        text_scores = normalize_scores(text_scores)
        print("Text mining scores predicted and normalized.", file=RESULT_FILE)

        # 构建相似度模型得分
        path_scores = build_similarity_model(history_data, test_entry, M)
        path_scores = normalize_scores(path_scores)
        print("Similarity scores calculated and normalized.", file=RESULT_FILE)

        # 使用TIEComposer组合得分
        combined_scores = combine_scores(text_scores, path_scores, LAMBDA)
        print("Combined scores computed using TIEComposer.", file=RESULT_FILE)

        # 推荐审查者
        recommended_reviewers = recommend_reviewers(combined_scores, top_k=7)
        print(f"Recommended Reviewers: {recommended_reviewers}", file=RESULT_FILE)

        # 获取实际审查者的邮箱
        actual_reviewers = [approver['name'] for approver in test_entry.get('approve_history', [])]
        print(f"Actual Reviewers: {actual_reviewers}", file=RESULT_FILE)

        # 计算指标
        acc_top1 = compute_top_k_accuracy(recommended_reviewers, actual_reviewers, k=1)
        acc_top3 = compute_top_k_accuracy(recommended_reviewers, actual_reviewers, k=3)
        acc_top5 = compute_top_k_accuracy(recommended_reviewers, actual_reviewers, k=5)
        mrr_top1 = compute_mrr_at_k(recommended_reviewers, actual_reviewers, k=1)
        mrr_top3 = compute_mrr_at_k(recommended_reviewers, actual_reviewers, k=3)
        mrr_top5 = compute_mrr_at_k(recommended_reviewers, actual_reviewers, k=5)

        print(f"Top-1 Accuracy: {acc_top1}", file=RESULT_FILE)
        print(f"Top-3 Accuracy: {acc_top3}", file=RESULT_FILE)
        print(f"Top-5 Accuracy: {acc_top5}", file=RESULT_FILE)
        print(f"MRR@1: {mrr_top1:.4f}", file=RESULT_FILE)
        print(f"MRR@3: {mrr_top3:.4f}", file=RESULT_FILE)
        print(f"MRR@5: {mrr_top5:.4f}", file=RESULT_FILE)

        # 保存测试结果
        test_results.append({
            'change_id': test_entry.get('changeId', 'N/A'),
            'acc_top1': acc_top1,
            'acc_top3': acc_top3,
            'acc_top5': acc_top5,
            'mrr_top1': mrr_top1,
            'mrr_top3': mrr_top3,
            'mrr_top5': mrr_top5
        })

    # 计算总体指标
    overall_metrics = compute_overall_metrics(test_results)
    return overall_metrics, test_results

def compute_overall_metrics(test_results):
    """
    计算总体的ACC、MRR@1、MRR@3、MRR@5、Top-1、Top-3、Top-5指标。
    """
    total_tests = len(test_results)
    if total_tests == 0:
        print("没有测试结果可用于计算指标。", file=RESULT_FILE)
        return None

    # 计算ACC
    acc_top1 = sum([1 if result['acc_top1'] else 0 for result in test_results]) / total_tests
    acc_top3 = sum([1 if result['acc_top3'] else 0 for result in test_results]) / total_tests
    acc_top5 = sum([1 if result['acc_top5'] else 0 for result in test_results]) / total_tests

    # 计算MRR
    mrr_top1 = sum([result['mrr_top1'] for result in test_results]) / total_tests
    mrr_top3 = sum([result['mrr_top3'] for result in test_results]) / total_tests
    mrr_top5 = sum([result['mrr_top5'] for result in test_results]) / total_tests

    # 创建指标字典
    metrics = {
        'acc_top1': acc_top1,
        'acc_top3': acc_top3,
        'acc_top5': acc_top5,
        'mrr_top1': mrr_top1,
        'mrr_top3': mrr_top3,
        'mrr_top5': mrr_top5
    }

    return metrics

def hyperparameter_optimization(test_files, M_values, LAMBDA_values, output_summary='hyperparameter_optimization_results.txt'):
    """
    自动化超参数优化，遍历所有参数组合，评估模型性能，并记录结果。
    """
    best_metrics = None
    best_params = {}
    optimization_results = []

    print("\nStarting Hyperparameter Optimization...\n", file=RESULT_FILE)

    with open(output_summary, 'w', encoding='utf-8') as summary_file:
        summary_file.write("Hyperparameter Optimization Results\n")
        summary_file.write("===================================\n\n")

        # 遍历所有超参数组合
        for M, LAMBDA in itertools.product(M_VALUES, LAMBDA_VALUES):
            print(f"Evaluating combination: M={M}, LAMBDA={LAMBDA}", file=RESULT_FILE)
            summary_file.write(f"Evaluating combination: M={M}, LAMBDA={LAMBDA}\n")

            # 评估模型
            overall_metrics, test_results = evaluate_model(M, LAMBDA, test_files)

            if not overall_metrics:
                print("No metrics computed for this combination. Skipping.", file=RESULT_FILE)
                summary_file.write("No metrics computed for this combination. Skipping.\n---\n\n")
                continue

            # 记录结果
            optimization_results.append({
                'M': M,
                'LAMBDA': LAMBDA,
                **overall_metrics
            })

            # 写入总结文件
            summary_file.write(f"Overall Metrics for M={M}, LAMBDA={LAMBDA}:\n")
            summary_file.write(f"  Top-1 Accuracy: {overall_metrics['acc_top1']:.4f}\n")
            summary_file.write(f"  Top-3 Accuracy: {overall_metrics['acc_top3']:.4f}\n")
            summary_file.write(f"  Top-5 Accuracy: {overall_metrics['acc_top5']:.4f}\n")
            summary_file.write(f"  MRR@1: {overall_metrics['mrr_top1']:.4f}\n")
            summary_file.write(f"  MRR@3: {overall_metrics['mrr_top3']:.4f}\n")
            summary_file.write(f"  MRR@5: {overall_metrics['mrr_top5']:.4f}\n")
            summary_file.write('---\n\n')

            # 打印和写入最佳参数
            if best_metrics is None or overall_metrics['mrr_top3'] > best_metrics['mrr_top3']:
                best_metrics = overall_metrics
                best_params = {'M': M, 'LAMBDA': LAMBDA}

        # 总结最佳参数
        if best_metrics:
            summary_file.write("Best Hyperparameter Combination:\n")
            summary_file.write(f"  M={best_params['M']}, LAMBDA={best_params['LAMBDA']}\n")
            summary_file.write(f"  Top-1 Accuracy: {best_metrics['acc_top1']:.4f}\n")
            summary_file.write(f"  Top-3 Accuracy: {best_metrics['acc_top3']:.4f}\n")
            summary_file.write(f"  Top-5 Accuracy: {best_metrics['acc_top5']:.4f}\n")
            summary_file.write(f"  MRR@1: {best_metrics['mrr_top1']:.4f}\n")
            summary_file.write(f"  MRR@3: {best_metrics['mrr_top3']:.4f}\n")
            summary_file.write(f"  MRR@5: {best_metrics['mrr_top5']:.4f}\n")
            summary_file.write("===================================\n")

            print("\nBest Hyperparameter Combination:", file=RESULT_FILE)
            print(f"  M={best_params['M']}, LAMBDA={best_params['LAMBDA']}", file=RESULT_FILE)
            print(f"  Top-1 Accuracy: {best_metrics['acc_top1']:.4f}", file=RESULT_FILE)
            print(f"  Top-3 Accuracy: {best_metrics['acc_top3']:.4f}", file=RESULT_FILE)
            print(f"  Top-5 Accuracy: {best_metrics['acc_top5']:.4f}", file=RESULT_FILE)
            print(f"  MRR@1: {best_metrics['mrr_top1']:.4f}", file=RESULT_FILE)
            print(f"  MRR@3: {best_metrics['mrr_top3']:.4f}", file=RESULT_FILE)
            print(f"  MRR@5: {best_metrics['mrr_top5']:.4f}", file=RESULT_FILE)

        else:
            summary_file.write("No valid hyperparameter combinations were evaluated.\n")
            print("No valid hyperparameter combinations were evaluated.", file=RESULT_FILE)

def main():
    # 获取所有测试文件列表
    test_files = sorted(glob.glob(os.path.join(DATASET_DIR, '*_test.jsonl')))
    total_test_files = len(test_files)
    print(f"Total test files found: {total_test_files}", file=RESULT_FILE)

    # 定义超参数搜索空间
    M_values = M_VALUES
    LAMBDA_values = LAMBDA_VALUES

    # 执行超参数优化
    hyperparameter_optimization(test_files, M_values, LAMBDA_values, "tie_hyperparameter_optimization_results.txt")

if __name__ == "__main__":
    main()
