import argparse
import math
import os.path as osp
from collections import defaultdict

import torch
import tqdm
import json
import pandas as pd
import numpy as np

from train_lmpnn import newlstr2name

parser = argparse.ArgumentParser()
parser.add_argument("--data_folder", type=str, default='data/FB15k-237-EFO1')
parser.add_argument("--output_folder", type=str, default='data/FB15k-237-EFO1')

formula_correspondence = {
    'r1(s1,f)': 'r1(s1,f1)',
    '(r1(s1,e1))&(r2(e1,f))': '(r1(s1,e1))&(r2(e1,f1))',
    'r1(s1,e1)&r2(e1,f)': '(r1(s1,e1))&(r2(e1,f1))',
    '((r1(s1,e1))&(r2(e1,e2)))&(r3(e2,f))': '(r1(s1,e1))&((r2(e1,e2))&(r3(e2,f1)))',
    'r1(s1,e1)&r2(e1,e2)&r3(e2,f)': '(r1(s1,e1))&((r2(e1,e2))&(r3(e2,f1)))',
    '(r1(s1,f))&(r2(s2,f))': '(r1(s1,f1))&(r2(s2,f1))',
    'r1(s1,f)&r2(s2,f)': '(r1(s1,f1))&(r2(s2,f1))',
    '((r1(s1,f))&(r2(s2,f)))&(r3(s3,f))': '(r1(s1,f1))&((r2(s2,f1))&(r3(s3,f1)))',
    'r1(s1,f)&r2(s2,f)&r3(s3,f)': '(r1(s1,f1))&((r2(s2,f1))&(r3(s3,f1)))',
    '((r1(s1,e1))&(r2(s2,e1)))&(r3(e1,f))': '(r1(s1,e1))&((r2(s2,e1))&(r3(e1,f1)))',
    'r1(s1,e1)&r2(s2,e1)&r3(e1,f)': '(r1(s1,e1))&((r2(s2,e1))&(r3(e1,f1)))',
    '((r1(s1,e1))&(r2(e1,f)))&(r3(s2,f))': '(r1(s1,e1))&((r2(e1,f1))&(r3(s2,f1)))',
    'r1(s1,e1)&r2(e1,f)&r3(s2,f)': '(r1(s1,e1))&((r2(e1,f1))&(r3(s2,f1)))',
    '(r1(s1,f))&(!(r2(s2,f)))': '(r1(s1,f1))&(!(r2(s2,f1)))',
    'r1(s1,f)&!r2(s2,f)': '(r1(s1,f1))&(!(r2(s2,f1)))',
    '((r1(s1,f))&(r2(s2,f)))&(!(r3(s3,f)))': '((r1(s1,f1))&(r2(s2,f1)))&(!(r3(s3,f1)))',
    'r1(s1,f)&r2(s2,f)&!r3(s3,f)': '((r1(s1,f1))&(r2(s2,f1)))&(!(r3(s3,f1)))',
    '((r1(s1,e1))&(!(r2(s2,e1))))&(r3(e1,f))': '((r1(s1,e1))&(!(r2(s2,e1))))&(r3(e1,f1))',
    'r1(s1,e1)&!r2(s2,e1)&r3(e1,f)': '((r1(s1,e1))&(!(r2(s2,e1))))&(r3(e1,f1))',
    '((r1(s1,e1))&(r2(e1,f)))&(!(r3(s2,f)))': '((r1(s1,e1))&(r2(e1,f1)))&(!(r3(s2,f1)))',
    'r1(s1,e1)&r2(e1,f)&!r3(s2,f)': '((r1(s1,e1))&(r2(e1,f1)))&(!(r3(s2,f1)))',
    '((r1(s1,e1))&(!(r2(e1,f))))&(r3(s2,f))': '((r1(s1,e1))&(!(r2(e1,f1))))&(r3(s2,f1))',
    'r1(s1,e1)&!r2(e1,f)&r3(s2,f)': '((r1(s1,e1))&(!(r2(e1,f1))))&(r3(s2,f1))',
    '(r1(s1,f))|(r2(s2,f))': '(r1(s1,f1))|(r2(s2,f1))',
    'r1(s1,f)|r2(s2,f)': '(r1(s1,f1))|(r2(s2,f1))',
    '((r1(s1,e1))&(r3(e1,f)))|((r2(s2,e1))&(r3(e1,f)))': '((r1(s1,e1))&(r3(e1,f1)))|((r2(s2,e1))&(r3(e1,f1)))',
    '(r1(s1,e1)|r2(s2,e1))&r3(e1,f)': '((r1(s1,e1))&(r3(e1,f1)))|((r2(s2,e1))&(r3(e1,f1)))',
    'r1(s1,e1)|r2(s2,e1))&r3(e1,f)': '((r1(s1,e1))&(r3(e1,f1)))|((r2(s2,e1))&(r3(e1,f1)))',
    '!(!r1(s1,f)&!r2(s2,f))': '(r1(s1,f1))|(r2(s2,f1))',
    '!(!r1(s1,e1)|r2(s2,e1))&r3(e1,f)': '((r1(s1,e1))&(r3(e1,f1)))|((r2(s2,e1))&(r3(e1,f1)))'
}

def convert_real_efo1_query(query_name, old_qa_dict):
    if query_name == '2m':
        return old_qa_dict
    elif query_name == '2nm':
        return old_qa_dict
    elif query_name == '3mp':
        new_qa_dict = {'s1': old_qa_dict['s1'], 'r1': old_qa_dict['r1'], 'r2': old_qa_dict['r2'],
                       'r3': old_qa_dict['r4'], 'r4': old_qa_dict['r3']}
        return new_qa_dict
    elif query_name == '3pm':
        return old_qa_dict
    elif query_name == 'im':
        return old_qa_dict
    elif query_name == '2il':
        return old_qa_dict
    elif query_name == '3il':
        return old_qa_dict
    elif query_name == '3c':
        new_qa_dict = {'s1': old_qa_dict['s1'], 's2': old_qa_dict['s2'], 'r1': old_qa_dict['r1'], 'r2': old_qa_dict['r3'],
                       'r3': old_qa_dict['r5'], 'r4': old_qa_dict['r2'], 'r5': old_qa_dict['r4']}
        return new_qa_dict
    elif query_name == '3cm':
        new_qa_dict = {'s1': old_qa_dict['s1'], 's2': old_qa_dict['s2'], 'r1': old_qa_dict['r1'], 'r2': old_qa_dict['r3'],
                       'r3': old_qa_dict['r2'], 'r4': old_qa_dict['r6'], 'r5': old_qa_dict['r5'], 'r6': old_qa_dict['r4']}
        return new_qa_dict
    elif query_name == 'pni':
        return old_qa_dict
    else:
        return old_qa_dict


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    abstract_query_data = pd.read_csv("data/DNF_evaluate_EFO1.csv")
    for mode in ['valid']:
        old_data_file = osp.join(args.data_folder, f'{mode}-qaa.json')
        new_data_file = osp.join(args.output_folder, f'{mode}_qaa.json')
        with open(old_data_file, 'rt') as f:
            old_data = json.load(f)
        new_data = defaultdict(list)
        for formula in old_data:
            if formula in formula_correspondence:
                new_formula = formula_correspondence[formula]
                if old_data[formula]:
                    for query in old_data[formula]:
                        qa_dict, easy_ans_dict, hard_ans_dict = query
                        new_easy_ans_dict = {'f1': [[easy_ans] for easy_ans in easy_ans_dict['f']]}
                        if mode in ['train']:
                            new_hard_ans_dict = []
                        else:
                            new_hard_ans_dict = {'f1': [[hard_ans] for hard_ans in hard_ans_dict['f']]}
                        new_data[new_formula].append([qa_dict, new_easy_ans_dict, new_hard_ans_dict])
            elif formula in newlstr2name:
                formula_name = newlstr2name[formula]
                new_formula = abstract_query_data['formula'][
                    abstract_query_data['Name'].index[abstract_query_data['Name'] == formula_name][0]]
                for query in old_data[formula]:
                    qa_dict, easy_ans_dict, hard_ans_dict = query
                    new_easy_ans_dict = {'f1': [[easy_ans] for easy_ans in easy_ans_dict['f']]}
                    if mode in ['train']:
                        new_hard_ans_dict = []
                    else:
                        new_hard_ans_dict = {'f1': [[hard_ans] for hard_ans in hard_ans_dict['f']]}
                    new_qa_dict = convert_real_efo1_query(formula_name, qa_dict)
                    new_data[new_formula].append([new_qa_dict, new_easy_ans_dict, new_hard_ans_dict])
            else:
                raise NotImplementedError

        with open(new_data_file, 'wt') as f:
            json.dump(new_data, f)

