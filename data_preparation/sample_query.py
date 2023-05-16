import argparse
import json
import logging
import os
import os.path as osp
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch import nn

from src.language.tnorm import GodelTNorm, ProductTNorm, Tnorm
from src.structure import get_nbp_class
from src.structure.knowledge_graph import KnowledgeGraph
from src.structure.knowledge_graph_index import KGIndex
from src.utils.data_util import RaggedBatch
from lifted_embedding_estimation_with_truth_value import name2lstr, newlstr2name, index2newlstr
from src.language.grammar import parse_lstr_to_disjunctive_formula
from src.language.fof import Disjunction, ConjunctiveFormula, DisjunctiveFormula
from src.utils.data import (QueryAnsweringMixDataLoader, QueryAnsweringSeqDataLoader,
                            QueryAnsweringSeqDataLoader_v2,
                            TrainRandomSentencePairDataLoader)

train_queries = list(name2lstr.values())
query_2in = 'r1(s1,f)&!r2(s2,f)'
query_2i = 'r1(s1,f)&r2(s2,f)'
parser = argparse.ArgumentParser()
#parser.add_argument("--output_name", type=str, default='new-qaa')
parser.add_argument("--output_folder", type=str, default='data/FB15k-237-EFO1-10000')
parser.add_argument("--data_folder", type=str, default='data/FB15k-237-betae')
parser.add_argument("--sample_num", type=int, default=10000)
parser.add_argument('--mode', choices=['train', 'valid', 'test'], default='test')
parser.add_argument("--meaningful_negation", type=bool, default=True)
parser.add_argument("--sample_formula_list", type=list, default=[8])


lstr_3c = '((((r1(s1,e1))&(r2(e1,f)))&(r3(s2,e2)))&(r4(e2,f)))&(r5(e1,e2))'
lstr_3pnc = '((((r1(s1,e1))&(r2(e1,f)))&(r3(s2,e2)))&(r4(e2,f)))&(!(r5(e1,e2)))'
lstr_mi = '(((r1(s1,e1))&(r2(e1,f)))&(r3(e1,f)))&(r4(s2,f))'
lstr_2an = '(r1(e1,f))&(!(r2(s1,f)))'
lstr_3pcp = '(((((r1(s1,e1))&(r2(e1,e3)))&(r3(s2,e2)))&(r4(e2,e3)))&(r5(e1,e2)))&(r6(e3,f))'


def double_checking_answer(given_lstr, fof_qa_dict, kg: KnowledgeGraph):
    if kg is None:
        return None
    if given_lstr == lstr_3c:
        e1_candidate = kg.hr2t[(fof_qa_dict['s1'], fof_qa_dict['r1'])]
        e2_candidate = kg.hr2t[(fof_qa_dict['s2'], fof_qa_dict['r3'])]
        all_ans = set()
        for e1_c in e1_candidate:
            f_candidate_set = kg.hr2t[(e1_c, fof_qa_dict['r2'])]
            e2_c_set = kg.hr2t[(e1_c, fof_qa_dict['r5'])].intersection(e2_candidate)
            if e2_c_set:
                f_e2_candidate_set = set.union(*[kg.hr2t[(e2_c, fof_qa_dict['r4'])] for e2_c in e2_c_set])
            else:
                f_e2_candidate_set = {}
            f_final_candidate = f_candidate_set.intersection(f_e2_candidate_set)
            all_ans.update(f_final_candidate)
        return all_ans
    elif given_lstr == lstr_3pcp:
        e1_candidate = kg.hr2t[(fof_qa_dict['s1'], fof_qa_dict['r1'])]
        e2_candidate = kg.hr2t[(fof_qa_dict['s2'], fof_qa_dict['r3'])]
        e3_candidate = set()
        for e1_c in e1_candidate:
            f_candidate_set = kg.hr2t[(e1_c, fof_qa_dict['r2'])]
            e2_c_set = kg.hr2t[(e1_c, fof_qa_dict['r5'])].intersection(e2_candidate)
            if e2_c_set:
                f_e2_candidate_set = set.union(*[kg.hr2t[(e2_c, fof_qa_dict['r4'])] for e2_c in e2_c_set])
            else:
                f_e2_candidate_set = {}
            f_final_candidate = f_candidate_set.intersection(f_e2_candidate_set)
            e3_candidate.update(f_final_candidate)
        all_ans = set()
        for e3_c in e3_candidate:
            all_ans.update(kg.hr2t[(e3_c, fof_qa_dict['r6'])])
        return all_ans
    elif given_lstr == lstr_3pnc:
        e1_candidate = kg.hr2t[(fof_qa_dict['s1'], fof_qa_dict['r1'])]
        e2_candidate = kg.hr2t[(fof_qa_dict['s2'], fof_qa_dict['r3'])]
        all_ans = set()
        for e1_c in e1_candidate:
            f_candidate_set = kg.hr2t[(e1_c, fof_qa_dict['r2'])]
            e2_c_set = e2_candidate - kg.hr2t[(e1_c, fof_qa_dict['r5'])]
            if e2_c_set:
                f_e2_candidate_set = set.union(*[kg.hr2t[(e2_c, fof_qa_dict['r4'])] for e2_c in e2_c_set])
            else:
                f_e2_candidate_set = {}
            f_final_candidate = f_candidate_set.intersection(f_e2_candidate_set)
            all_ans.update(f_final_candidate)
        return all_ans
    elif given_lstr == lstr_mi:
        e1_candidate = kg.hr2t[(fof_qa_dict['s1'], fof_qa_dict['r1'])]
        f_candidate = kg.hr2t[(fof_qa_dict['s2'], fof_qa_dict['r4'])]
        if e1_candidate:
            f_candidate2 = set.union(
                *[kg.hr2t[(e1_c, fof_qa_dict['r2'])].intersection(kg.hr2t[(e1_c, fof_qa_dict['r3'])])
                  for e1_c in e1_candidate])
        else:
            f_candidate2 = {}
        return f_candidate.intersection(f_candidate2)
    elif given_lstr == lstr_2an:
        f_candidate = kg.r2t[fof_qa_dict['r1']]
        f_candidate = f_candidate - kg.hr2t[(fof_qa_dict['s1'], fof_qa_dict['r2'])]
        return f_candidate
    else:
        return None


def sample_one_formula_query(given_lstr, easy_kg: KnowledgeGraph, hard_kg: KnowledgeGraph, sample_num, sample_mode,
                             meaningful_negation, double_checking, existing_all_qa_dict=None):
    print(f'sampling query of {given_lstr}')
    fof = parse_lstr_to_disjunctive_formula(given_lstr)
    all_qa_dict = existing_all_qa_dict if existing_all_qa_dict else set()
    all_query_list = []
    now_index = -1
    with tqdm.tqdm(total=sample_num) as pbar:
        while pbar.n < sample_num:
            qa_dict = fof.sample_query(hard_kg, meaningful_negation)
            if qa_dict and str(qa_dict) not in all_qa_dict:  # We notice sampling may fail and return None
                all_qa_dict.add(str(qa_dict))  # remember it to avoid repeat
                fof.append_qa_instances(qa_dict)
                now_index += 1
                if sample_mode == 'train':
                    hard_answer = fof.deterministic_query(now_index, hard_kg)
                    easy_answer = set()
                else:
                    hard_answer = fof.deterministic_query(now_index, hard_kg)
                    easy_answer = fof.deterministic_query(now_index, easy_kg)
                if double_checking:
                    check_easy_ans, check_hard_ans = double_checking_answer(given_lstr, qa_dict, easy_kg), \
                        double_checking_answer(given_lstr, qa_dict, hard_kg)
                else:
                    check_easy_ans, check_hard_ans = None, None
                if check_hard_ans is not None:
                    assert hard_answer == check_hard_ans
                if check_easy_ans is not None:
                    assert easy_answer == check_easy_ans
                if hard_answer - easy_answer:
                    if sample_mode == 'train':
                        new_query = [qa_dict, {'f': list(hard_answer)}, []]
                    else:
                        new_query = [qa_dict, {'f': list(easy_answer)}, {'f': list(hard_answer - easy_answer)}]
                    all_query_list.append(new_query)
                    pbar.update(1)
    return all_query_list


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    kgidx = KGIndex.load(osp.join(args.data_folder, 'kgindex.json'))
    train_kg = KnowledgeGraph.create(
        triple_files=osp.join(args.data_folder, 'train_kg.tsv'),
        kgindex=kgidx)
    valid_kg = KnowledgeGraph.create(
        triple_files=osp.join(args.data_folder, 'valid_kg.tsv'),
        kgindex=kgidx)
    test_kg = KnowledgeGraph.create(
        triple_files=osp.join(args.data_folder, 'test_kg.tsv'),
        kgindex=kgidx)
    """
    for lstr in DNF_lstr2name:
        test_sample_query(lstr, train_kg)
    """

    for index, lstr_index in enumerate(args.sample_formula_list):
        lstr = index2newlstr[lstr_index]
        now_data = {lstr: []}
        output_file_name = osp.join(args.output_folder, f'{args.mode}_{lstr_index}_real_EFO1_qaa.json')
        if os.path.exists(output_file_name):
            with open(output_file_name, 'rt') as f:
                old_data = json.load(f)
        else:
            old_data = {}
        useful_num = 0
        all_qa_dict = set()
        if lstr in old_data:
            for i in range(len(old_data[lstr])):
                if old_data[lstr][i][2]['f'] and str(old_data[lstr][i][0]) not in all_qa_dict:
                    now_data[lstr].append(old_data[lstr][i])
                    useful_num += 1
                all_qa_dict.add(str(old_data[lstr][i][0]))
        if useful_num == args.sample_num:
            continue
        else:
            if args.mode == 'train':
                all_query = sample_one_formula_query(lstr, None, train_kg, args.sample_num - useful_num, args.mode,
                                                     args.meaningful_negation, True, all_qa_dict)
            elif args.mode == 'valid':
                all_query = sample_one_formula_query(lstr, train_kg, valid_kg, args.sample_num - useful_num, args.mode,
                                                     args.meaningful_negation, True, all_qa_dict)
            elif args.mode == 'test':
                all_query = sample_one_formula_query(lstr, valid_kg, test_kg, args.sample_num - useful_num, args.mode,
                                                     args.meaningful_negation, False, all_qa_dict)
            else:
                raise NotImplementedError
            now_data[lstr].extend(all_query)
        with open(output_file_name, 'wt') as f:
            json.dump(now_data, f)

