import json
import pprint

def path2List(fileString):
    return fileString.split("/")

def LCP(f1, f2):
    f1 = path2List(f1)
    f2 = path2List(f2)
    common_path = 0
    min_length = min(len(f1), len(f2))
    for i in range(min_length):
        if f1[i] == f2[i]:
            common_path += 1
        else:
            break
    return common_path


def LCSuff(f1, f2):
    f1 = path2List(f1)
    f2 = path2List(f2)
    common_path = 0
    r = range(min(len(f1), len(f2)))
    rr = list(r)[::-1]
    for i in rr:
        if f1[i] == f2[i]:
            common_path += 1
        else:
            break
    return common_path


def LCSubstr(f1, f2):
    f1 = path2List(f1)
    f2 = path2List(f2)
    common_path = 0
    if len(set(f1) & set(f2)) > 0:
        mat = [[0 for x in range(len(f2) + 1)] for x in range(len(f1) + 1)]
        for i in range(len(f1) + 1):
            for j in range(len(f2) + 1):
                if i == 0 or j == 0:
                    mat[i][j] = 0
                elif f1[i - 1] == f2[j - 1]:
                    mat[i][j] = mat[i - 1][j - 1] + 1
                    common_path = max(common_path, mat[i][j])
                else:
                    mat[i][j] = 0
    return common_path


def LCSubseq(f1, f2):
    f1 = path2List(f1)
    f2 = path2List(f2)
    if len(set(f1) & set(f2)) > 0:
        L = [[0 for x in range(len(f2) + 1)] for x in range(len(f1) + 1)]
        for i in range(len(f1) + 1):
            for j in range(len(f2) + 1):
                if i == 0 or j == 0:
                    L[i][j] = 0
                elif f1[i - 1] == f2[j - 1]:
                    L[i][j] = L[i - 1][j - 1] + 1
                else:
                    L[i][j] = max(L[i - 1][j], L[i][j - 1])
        common_path = L[len(f1)][len(f2)]
    else:
        common_path = 0
    return common_path


def percentage(f1, f2):
    number = len(path2List(f1))
    s1 = LCP(f1, f2)
    s2 = LCSuff(f1, f2)
    s3 = LCSubstr(f1, f2)
    s4 = LCSubseq(f1, f2)
    return (s1 + s2 + s3 + s4) / (number * 4)


def read_data(fname) -> list[dict]:
    data = []
    with open(fname, 'r', encoding="utf-8") as file:
        for line in file:
            line_json = json.loads(line)
            for file in line_json["files"]:
                d = {"file_path": file, "reviewer": line_json['approve_history'][0]["name"]}
                data.append(d)
    return data


def get_gt_pro(file_path):
    with open(file_path, 'r', encoding="utf-8") as file:
        json_data = json.load(file)
        gt = json_data["approve_history"][0]["name"]
        profile_parent = json_data['project_parent']
    return gt, profile_parent


def find_best_match(test_item, history_data):
    """
    计算分数

    :param test_item:
    :param history_data:
    :return: 返回数量等于history_file的数量
    """
    similarity_scores = []

    for history_item in history_data:
        similarity = percentage(test_item['file_path'], history_item['file_path'])
        if similarity >= 0:
            similarity_scores.append((history_item['reviewer'], similarity))


    return similarity_scores


def match_test_to_history(history_data_file, test_data_file):
    history_data = read_data(history_data_file)
    test_data = read_data(test_data_file)
    ground_truth, profile_parent = get_gt_pro(test_data_file)

    result = {}
    for test_item in test_data:
        reviewers_score = find_best_match(test_item, history_data)

        for item in reviewers_score:
            key = item[0]  # 第一个元素（字符串）
            value = item[1]  # 第二个元素（数字）

            # 如果字符串已经在字典中，则累加其值，否则添加新键
            if key in result:
                result[key] += value
            else:
                result[key] = value

    sorted_reviewers = [k for k, v in sorted(result.items(), key=lambda item: item[1], reverse=True)]
    print(f"推荐列表: {sorted_reviewers}\n正确的推荐人ID: {ground_truth}")
    if ground_truth in sorted_reviewers:
        rank = sorted_reviewers.index(ground_truth) + 1
    else:
        rank = 0

    return sorted_reviewers, rank, profile_parent


if __name__ == '__main__':
    dir_data = f"../Implementation/data"
    list_reviewer_all = []
    for index in range(1590):
        print(f"index: {index}")
        recommended_reviewers, rank, project_parent = match_test_to_history(f'{dir_data}/{index}_history.jsonl', f'{dir_data}/{index}_test.jsonl')
        list_reviewer_all.append(recommended_reviewers)
