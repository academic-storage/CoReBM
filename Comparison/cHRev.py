import json
import math
import pprint
import pandas as pd
from enum import Enum
from collections import defaultdict

from datetime import datetime

class Xfactor:
    def __init__(self, cf, wf, tf, cf_, wf_, tf_):
        self.cf = cf    # 审阅者r对文件f所提供的审阅意见的数量
        self.wf = wf    # 审阅者r为文件f提供评审意见的工作日数
        self.tf = tf    # 审阅者r为文件f审阅的最近一个工作日
        self.cf_ = cf_  # 文件f所有的审阅数
        self.wf_ = wf_  # 文件f提交审查意见的总工作日数
        self.tf_ = tf_  # 文件f提交审查意见的最近一个工作日

    def xfactor(self):
        pfactor = self.cf / self.cf_ + self.wf / self.wf_

        tf_date = datetime.strptime(self.tf, "%Y-%m-%d")
        tf__date = datetime.strptime(self.tf_, "%Y-%m-%d")
        date_difference = (tf_date - tf__date).days

        if self.tf == self.tf_:
            return pfactor + 1
        else:
            return pfactor + 1 / date_difference

def process_all_data(input_file_path, output_file_path):
    extracted_data = []

    with open(input_file_path, 'r', encoding="utf-8") as infile:
        for line in infile:
            data = json.loads(line)

            files_list = data.get('files', [])
            submit_date = data.get('submit_date', '')
            close_date = data.get('close_date', '')

            approve_history_list = []

            approve_history = data.get('approve_history', [])
            for approve in approve_history:
                approve_info = {
                    'userId': approve.get('name', ''),
                    'grant_date': approve.get('grant_date', '')
                }
                approve_history_list.append(approve_info)

            extracted_info = {
                'files': files_list,
                'approve_history': approve_history_list,
                'submit_date': submit_date,
                'close_date': close_date
            }

            extracted_data.append(extracted_info)

    results = []

    for item in extracted_data:
        files = item.get("files", [])
        approve_history = item.get("approve_history", [])
        submit_date = item.get("submit_date")
        close_date = item.get("close_date")

        allhistory = len(approve_history)

        if approve_history:
            last = max([datetime.strptime(history["grant_date"][:26], "%Y-%m-%d %H:%M:%S.%f") for history in
                        approve_history]).strftime("%Y-%m-%d")
        else:
            last = None

        diff_dates = set([datetime.strptime(history["grant_date"][:26], "%Y-%m-%d %H:%M:%S.%f").date() for history in
                          approve_history])
        diffday = len(diff_dates)

        user_history_count = defaultdict(int)
        for history in approve_history:
            user_history_count[history["userId"]] += 1

        user_dates = defaultdict(set)
        for history in approve_history:
            user_dates[history["userId"]].add(
                datetime.strptime(history["grant_date"][:26], "%Y-%m-%d %H:%M:%S.%f").date())

        user_diffday_count = {user_id: len(dates) for user_id, dates in user_dates.items()}

        user_last_grant_date = {}
        for user_id, dates in user_dates.items():
            user_last_grant_date[user_id] = max(dates).strftime("%Y-%m-%d")

        result = {
            "files": files,
            "cf2": allhistory,
            "tf2": last,
            "wf2": diffday,
            "user_cf": dict(user_history_count),
            "user_wf": user_diffday_count,
            "user_tf": user_last_grant_date
        }

        results.append(result)

    final_results = []

    for item in results:
        files = item.get("files", [])
        cf2 = item.get("cf2")
        tf2 = item.get("tf2")
        wf2 = item.get("wf2")
        user_cf = item.get("user_cf", {})
        user_tf = item.get("user_tf", {})
        user_wf = item.get("user_wf", {})

        for user in user_cf.keys():
            for file in files:
                current_cf2 = cf2
                current_tf2 = tf2
                current_wf2 = wf2
                current_user_cf = user_cf[user]
                current_user_tf = user_tf[user]
                current_user_wf = user_wf[user]

                result = Xfactor(current_user_cf, current_user_wf, current_user_tf, current_cf2, current_wf2,
                                         current_tf2).xfactor()

                final_results.append({
                    "file": file,
                    "user": user,
                    "result": result
                })

    with open(output_file_path, 'w', encoding="utf-8") as outfile:
        json.dump(final_results, outfile, indent=4, ensure_ascii=False)

    # print("Processing complete")

def read_user_from_json(file_path):
    '''
    获取用户信息
    :param file_path:
    :return: 用户姓名, 父项目
    '''
    # 打开并读取JSON文件
    with open(file_path, 'r', encoding="utf-8") as file:
        data = json.load(file)

    # 从JSON数据中提取user name
    approve_history = data.get('approve_history', [])
    project_parent = data.get('project_parent', "NO_PROJECT_PARENT")
    if approve_history:
        # 假设JSON文件中只有一条数据，我们只需要第一条数据的userId
        user_name = approve_history[0].get('name')
        return user_name, project_parent
    else:
        # 如果approve_history为空，返回None或抛出异常
        return None, project_parent


def calculate_user_scores(input_file_path, n, test):
    '''
    计算用户得分，并排序
    :param input_file_path:
    :param n:
    :param test:
    :return: 评审人姓名列表, 真值的位次
    '''
    with open(input_file_path, 'r', encoding="utf-8") as infile:
        data = json.load(infile)

    user_scores = defaultdict(float)

    for item in data:
        user = item['user']
        result = float(item['result'])
        user_scores[user] += result

    sorted_user_scores = sorted(user_scores.items(), key=lambda x: x[1], reverse=True)

    top_n_users = sorted_user_scores[:n]
    list = []
    top_m = 0
    for rank, (user, allresult) in enumerate(top_n_users, start=1):
        # print(n, ":", user)
        list.append(user)
        if user == test:
            # print("--------------------  top-m = ", n - 1)
            top_m = rank
    return list, top_m

if __name__ == '__main__':
    list_reviewer_all = []
    for i in range(1590):
        test_path = 'data/' + str(i) + '_test.jsonl'
        history_path = 'data/' + str(i) + '_history.jsonl'
        test_name, project_parent = read_user_from_json(test_path)
        output_file_path = "data/chis/0.json"
        process_all_data(history_path, output_file_path)

        input_file_path = "data/chis/0.json"
        top_n = 7
        list_reviewer, rank = calculate_user_scores(input_file_path, top_n, test_name)

        list_reviewer_all.append(list_reviewer)

