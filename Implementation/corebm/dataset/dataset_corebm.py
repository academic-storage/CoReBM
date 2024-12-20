import os
import random
import subprocess
import pandas as pd
import numpy as np
import json
import time
import re
from loguru import logger
from pathlib import Path
from langchain.prompts import PromptTemplate


def to_timestamp(s: str) -> int:
    timestamp = s
    timestamp = int(time.mktime(time.strptime(timestamp, "%Y-%m-%d %H:%M:%S")))
    return timestamp


def read_data(dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    user_profile = pd.read_csv(os.path.join(dir, 'user_info', 'user_info.csv')).set_index('id')
    folder = Path(os.path.join(dir))
    dfs = []

    for file_path in folder.glob("*.jsonl"):
        lines = []
        # read jsonl
        with file_path.open('r', encoding='utf-8') as file:
            for line in file:
                line_dic = json.loads(line)
                line_dic['submit_date'] = line_dic['submit_date'].split('.')[0]
                line_dic['submit_time'] = to_timestamp(line_dic['submit_date'])
                line_dic['code_diff'] = line_dic['code_diff']['add'].replace("<!-One File End-!>", '')[:200] + "\n" + line_dic['code_diff']['delete'].replace("<!-One File End-!>", '')[:200]
                line_dic['code_diff'] = line_dic['code_diff'].replace('\n', '  ')
                lines.append(line_dic)

        df = pd.DataFrame(lines)
        dfs.append(df)

    data_df = pd.concat(dfs)
    data_df = data_df.sort_values(by=['submit_time'], kind='mergesort').reset_index(drop=True)
    data_df['PR_id'] = pd.Series(range(1, len(data_df) + 1))
    data_df = data_df[['PR_id', 'changeId', 'approve_history', 'submit_date', 'submit_time', 'files', 'project_parent', 'project', 'subject', 'owner', 'code_diff']]

    return data_df, user_profile


def reindex(out_df: pd.DataFrame = None) -> dict:
    # reindex (start from 1)
    all_name = pd.concat([out_df['reviewer_name'], out_df['owner_name']]).drop_duplicates()

    names = sorted(all_name.unique())
    r2name = dict(zip(names, range(1, len(names) + 1)))
    return r2name


def process_interaction_data(data_df: pd.DataFrame, n_neg_reviewer: int, user_profile_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out_df = pd.DataFrame(data=None, columns=['PR_id', 'changeId', 'submit_date', 'submit_time', 'grant_date', 'grant_time', 'reviewer_id', 'reviewer_name', 'duration', 'files', 'project_parent', 'project', 'subject', 'owner_name', 'owner_id', 'code_diff'])
    index_out = 0
    for _, row in data_df.iterrows():
        for reviewer in row['approve_history']:
            if int(reviewer['grant_date'].split('-')[0]) == 1900:
                continue
            if reviewer['name'].isspace() is True or row['owner']['name'] is None or row['owner']['name'].isspace() is True:
                continue
            grant_date = reviewer['grant_date'].split('.')[0]
            out_df.loc[index_out] = [row['PR_id'], row['changeId'], row['submit_date'], row['submit_time'], grant_date, to_timestamp(grant_date), int(reviewer['accountId']), reviewer['name'], to_timestamp(grant_date) - row['submit_time'], row['files'], row['project_parent'], row['project'], row['subject'], row['owner']['name'], row['owner']['accountId'], row['code_diff']]
            index_out += 1

    out_df['files'] = out_df['files'].astype(str)
    # temporary storage out_df
    # out_df.to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'corebm', 'temp', 'out.csv'))

    # read temporary storage
    # out_df = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'corebm', 'temp', 'out.csv'))[['PR_id', 'changeId', 'submit_date', 'submit_time', 'grant_date', 'grant_time', 'reviewer_id', 'reviewer_name', 'duration', 'files', 'project_parent', 'project', 'subject', 'owner_id', 'owner_name', 'code_diff']]
    # out_df = out_df.groupby('reviewer_name').tail(20).copy()

    # reindex (start from 1)
    r2name = reindex(out_df)

    out_df['reviewer_id'] = out_df['reviewer_name'].apply(lambda x: r2name[x])
    out_df['owner_id'] = out_df['owner_name'].apply(lambda x: r2name[x])

    out_df['reviewer_profile'] = out_df['reviewer_id'].apply(lambda x: user_profile_df.loc[x]['profile'])
    out_df['owner_profile'] = out_df['owner_id'].apply(lambda x: user_profile_df.loc[x]['profile'])

    pr_reviewer_set = dict()
    for pr_id, seq_df in out_df.groupby('PR_id'):
        pr_reviewer_set[pr_id] = set(seq_df['reviewer_id'].values.tolist())


    def negative_sample(df):
        reviewer_id_list = list(set(out_df['reviewer_id'].tolist()))
        random_rng = np.random.default_rng(seed=41)
        neg_reviewer = random_rng.choice(reviewer_id_list, (len(df), n_neg_reviewer), replace=True)
        for i, pr_row in df.iterrows():
            reviewer = pr_reviewer_set[pr_row['PR_id']]
            for j in range(len(neg_reviewer[i])):
                # while user_profile_df.loc[neg_reviewer[i][j]]["project"] != pr_row['project_parent'] or neg_reviewer[i][j] in reviewer or neg_reviewer[i][j] in neg_reviewer[i][:j]:
                while neg_reviewer[i][j] in reviewer or neg_reviewer[i][j] in neg_reviewer[i][:j]:
                    neg_reviewer[i][j] = random_rng.choice(reviewer_id_list)
            assert len(set(neg_reviewer[i])) == len(neg_reviewer[i])  # duplicate
        df['n_reviewer_id'] = neg_reviewer.tolist()
        return df

    def generate_dev_test(data_df):
        result_dfs = []
        for idx in range(2):
            result_df = data_df.groupby('reviewer_name').tail(1).copy()
            data_df = data_df.drop(result_df.index)
            result_dfs.append(result_df)
        return result_dfs, data_df

    out_df = negative_sample(out_df)

    leave_df = out_df.groupby('reviewer_name').head(1)
    data_df = out_df.drop(leave_df.index)   # drop 1 pr

    [test_df, dev_df], data_df = generate_dev_test(data_df)
    train_df = pd.concat([leave_df, data_df]).sort_index()
    # train_df + test_df + dev_df = out_df
    return train_df, dev_df, test_df, out_df


def process_reviewer_data(out_df: pd.DataFrame, user_profile_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    reviewer_df = out_df[['reviewer_id', 'reviewer_name', 'project_parent']].drop_duplicates()

    df1 = out_df[['reviewer_id', 'reviewer_name']].rename(columns={'reviewer_id': 'id', 'reviewer_name': 'name'})
    df2 = out_df[['owner_id', 'owner_name']].rename(columns={'owner_id': 'id', 'owner_name': 'name'})
    all_user_df = pd.concat([df1, df2]).drop_duplicates()
    all_user_df = all_user_df.set_index('id').sort_index()

    reviewer_df['reviewer_profile'] = reviewer_df['reviewer_id'].apply(lambda x: user_profile_df.loc[x]['profile'])
    reviewer_df = reviewer_df.set_index('reviewer_id').sort_index()

    return reviewer_df, all_user_df

def process_pr_data(out_df: pd.DataFrame, user_profile_df: pd.DataFrame) -> pd.DataFrame:
    pr_df = out_df[['PR_id', 'files', 'project', 'subject', 'code_diff', 'owner_id']].drop_duplicates().set_index('PR_id')

    pr_df['owner_profile'] = pr_df['owner_id'].apply(lambda x: user_profile_df.loc[x]['profile'])

    for col in pr_df.columns.to_list():
        try:
            pr_df[col] = pr_df[col].apply(lambda x: 'None' if pd.isna(x) else x)
        except:
            continue

    input_variables = pr_df.columns.to_list()
    template = PromptTemplate(
        template='PR Subject: {subject}, Project: {project}, Files: {files}, PR Contributor Info: [{owner_profile}], Changed code: {code_diff}',
        input_variables=input_variables,
    )
    pr_df['PR_info'] = pr_df[input_variables].apply(lambda x: template.format(**x), axis=1)

    return pr_df



def corebm_process_data(dir: str, n_neg_pr: int = 9):
    """Process the amazon raw data and output the processed data to `dir`.

    Args:
        `dir (str)`: the directory of the dataset, e.g. `'data/Beauty'`. The raw dataset will be downloaded into `'{dir}/raw_data'` if not exists. We supppose the base name of dir is the category name.
    """
    raw_dir = os.path.join(dir, 'raw_data')
    data_df, user_profile_df = read_data(raw_dir)

    # train_df + test_df + dev_df = out_df
    train_df, dev_df, test_df, out_df = process_interaction_data(data_df, n_neg_pr, user_profile_df)
    logger.info(f'Number of interactions: {out_df.shape[0]}')

    pr_df = process_pr_data(out_df, user_profile_df)
    logger.info(f"Number of pullrequest: {out_df['PR_id'].nunique()}")

    reviewer_df, all_user_df = process_reviewer_data(out_df, user_profile_df)
    logger.info(f"Number of reviewers: {out_df['reviewer_id'].nunique()}")

    dfs = [train_df, dev_df, test_df]
    logger.info(f'Completed information, adding candidate info.')

    for df in dfs:
        df['PR_info'] = df['PR_id'].apply(lambda x: pr_df.loc[x]['PR_info'])
        df['candidate_reviewer_id'] = df.apply(lambda x: [x['reviewer_id']]+x['n_reviewer_id'], axis = 1)
        def shuffle_list(x):
            np.random.default_rng(seed=41).shuffle(x)
            return x
        df['candidate_reviewer_id'] = df['candidate_reviewer_id'].apply(lambda x: shuffle_list(x))  # shuffle candidates id
        # replace empty string with 'None'
        for col in df.columns.to_list():
            try:
                df[col] = df[col].apply(lambda x: 'None' if x == '' else x)
            except:
                continue

    train_df = dfs[0]
    dev_df = dfs[1]
    test_df = dfs[2]
    all_df = pd.concat([train_df, dev_df, test_df])
    all_df = all_df.sort_values(by=['submit_time'], kind='mergesort')
    all_df = all_df.reset_index(drop=True)
    logger.info('Output data')
    test_df = test_df.sample(frac=1, random_state=41)
    dev_df = dev_df.sample(frac=1, random_state=41)
    pr_df.to_csv(os.path.join(dir, 'pullrequest.csv'))
    reviewer_df.to_csv(os.path.join(dir, 'reviewer.csv'))
    # all_user_df.to_csv(os.path.join(dir, 'all_user.csv'))
    train_df.to_csv(os.path.join(dir, 'train.csv'), index=False)
    dev_df.to_csv(os.path.join(dir, 'dev.csv'), index=False)
    test_df.to_csv(os.path.join(dir, 'test.csv'), index=False)
    all_df.to_csv(os.path.join(dir, 'all.csv'), index=False)


if __name__ == "__main__":
    corebm_process_data(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'corebm'), 6)
