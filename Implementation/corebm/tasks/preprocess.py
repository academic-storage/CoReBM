from argparse import ArgumentParser

from corebm.tasks.base import Task
from corebm.dataset import corebm_process_data

class PreprocessTask(Task):
    @staticmethod
    def parse_task_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument('--data_dir', type=str, required=True, help='input file')
        parser.add_argument('--dataset', type=str, required=True, choices=['corebm'], help='output file')
        parser.add_argument('--n_neg_items', type=int, default=7, help='numbers of negative items')
        return parser

    def run(self, data_dir: str, dataset: dir, n_neg_items: int):
        if dataset == 'corebm':
            corebm_process_data(data_dir, n_neg_items)
        else:
            raise NotImplementedError

if __name__ == '__main__':
    PreprocessTask().launch()