import argparse
import json
import logging
import os
import os.path as osp
import random
from collections import defaultdict
from typing import List
import copy

import numpy as np
import scipy.sparse
import torch
import torch.nn.functional as F
import tqdm
import pickle
from torch import nn
from scipy.sparse import csc_matrix, diags, issparse

from src.language.tnorm import GodelTNorm, ProductTNorm, Tnorm
from src.language.fof import ConjunctiveFormula, DisjunctiveFormula
from src.structure import get_nbp_class
from src.structure.knowledge_graph import KnowledgeGraph, kg_remove_node
from src.structure.knowledge_graph_index import KGIndex
from src.structure.neural_binary_predicate import NeuralBinaryPredicate
from src.utils.data import QueryAnsweringSeqDataLoader_v2
from src.utils.class_util import Writer
from src.utils.data_util import RaggedBatch
from lifted_embedding_estimation_with_truth_value import compute_evaluation_scores


torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser()
parser.add_argument("--pkl", type=str, default='sparse/scipy_0.01_0.01.pickle')
parser.add_argument("--batch_size", type=int, default=3)
parser.add_argument("--data_folder", type=str, default='data')


def solve_conjunctive(positive_graph: KnowledgeGraph, negative_graph: KnowledgeGraph, relation_matrix,
                      now_candidate_set: dict, conjunctive_tnorm, existential_tnorm, now_variable, max_enumeration=None):
    n_entity = relation_matrix[0].shape[0]
    if not positive_graph.triples and not negative_graph.triples:
        return now_candidate_set[now_variable]
    if len(now_candidate_set) == 1:
        return now_candidate_set
    now_leaf_node, adjacency_node, being_asked_variable = \
        find_leaf_node(positive_graph, negative_graph, now_candidate_set, now_variable)
    if now_leaf_node:  # If there exists leaf node in the query graph, always possible to shrink into a sub_problem.
        adjacency_node_list = [adjacency_node]
        if being_asked_variable:
            next_variable = adjacency_node
            sub_pos_g, sub_neg_g = kg_remove_node(positive_graph, now_leaf_node), \
                                   kg_remove_node(negative_graph, now_leaf_node)
            sub_ans = solve_conjunctive(sub_pos_g, sub_neg_g, relation_matrix, now_candidate_set,
                                        conjunctive_tnorm, existential_tnorm, next_variable)
            final_ans = extend_ans(now_leaf_node, adjacency_node, positive_graph, negative_graph, relation_matrix,
                                   now_candidate_set[now_leaf_node], sub_ans, conjunctive_tnorm, existential_tnorm)
            return final_ans
        else:
            answer = cut_node_sub_problem(now_leaf_node, adjacency_node_list, positive_graph, negative_graph,
                                          relation_matrix, now_candidate_set, conjunctive_tnorm, existential_tnorm,
                                          now_variable)
            return answer
    else:
        to_enumerate_node, adjacency_node_list = find_enumerate_node(positive_graph, negative_graph, now_candidate_set,
                                                                     now_variable)
        if max_enumeration:
            easy_candidate = np.count_nonzero(now_candidate_set[to_enumerate_node] == 1)
            max_enumeration_here = max_enumeration + easy_candidate
            to_enumerate_candidates = np.argsort(now_candidate_set[to_enumerate_node][::-1])[max_enumeration_here]
        else:
            to_enumerate_candidates = now_candidate_set[to_enumerate_node].nonzero()[0]
        all_enumerate_ans = np.zeros((len(to_enumerate_candidates), n_entity))
        for i, enumerate_candidate in enumerate(to_enumerate_candidates):
            single_candidate = np.zeros_like(now_candidate_set[to_enumerate_node])
            candidate_truth_value = now_candidate_set[to_enumerate_node][enumerate_candidate]
            single_candidate[enumerate_candidate] = 1
            now_candidate_set[to_enumerate_node] = single_candidate
            answer = cut_node_sub_problem(to_enumerate_node, adjacency_node_list, positive_graph, negative_graph,
                                          relation_matrix, now_candidate_set, conjunctive_tnorm, existential_tnorm,
                                          now_variable)
            if conjunctive_tnorm == 'product':
                enumerate_ans = candidate_truth_value * answer
            elif conjunctive_tnorm == 'Godel':
                enumerate_ans = np.minimum(candidate_truth_value, answer)
            else:
                raise NotImplementedError
            all_enumerate_ans[i] = enumerate_ans
        if existential_tnorm == 'Godel':
            final_ans = np.amax(all_enumerate_ans, axis=-2)
        else:
            raise NotImplementedError
        return final_ans


def find_leaf_node(sub_graph: KnowledgeGraph, neg_sub_graph: KnowledgeGraph, now_candidate, now_variable):
    """
    Find a leaf node with least possible candidate. The now-asking variable is first.
    """
    return_candidate = [None, None, 0]
    for node in now_candidate:
        adjacency_node_set = set.union(
            *[sub_graph.h2t[node], sub_graph.t2h[node], neg_sub_graph.h2t[node],
              neg_sub_graph.t2h[node]])
        if len(adjacency_node_set) == 1:
            if node == now_variable:
                return node, list(adjacency_node_set)[0], True
            candidate_num = np.count_nonzero(now_candidate[node])
            if not return_candidate[0] or candidate_num < return_candidate[2]:
                return_candidate = [node, list(adjacency_node_set)[0], candidate_num]
    return return_candidate[0], return_candidate[1], False


def find_enumerate_node(sub_graph: KnowledgeGraph, neg_sub_graph: KnowledgeGraph, now_candidate, now_variable):
    return_candidate = [None, 100, 100000]
    for node in now_candidate:
        if node == now_variable:
            continue
        adjacency_node_list = list(set.union(*[sub_graph.h2t[node], sub_graph.t2h[node], neg_sub_graph.h2t[node],
                                               neg_sub_graph.t2h[node]]))
        adjacency_node_num = len(adjacency_node_list)
        candidate_num = np.count_nonzero(now_candidate[node])
        if not return_candidate[0] or adjacency_node_num < len(return_candidate[1]) or \
                (adjacency_node_num == len(return_candidate[1]) and candidate_num < return_candidate[2]):
            return_candidate = node, adjacency_node_list, candidate_num
    return return_candidate[0], return_candidate[1]


def cut_node_sub_problem(to_cut_node, adjacency_node_list, sub_graph: KnowledgeGraph, neg_sub_graph: KnowledgeGraph,
                         r_matrix_list, now_candidate_set, conj_tnorm, exist_tnorm, now_variable):
    new_candidate_set = copy.deepcopy(now_candidate_set)
    for adjacency_node in adjacency_node_list:
        adj_candidate_vec = existential_update(to_cut_node, adjacency_node, sub_graph, neg_sub_graph, r_matrix_list,
                                               new_candidate_set[to_cut_node], new_candidate_set[adjacency_node],
                                               conj_tnorm, exist_tnorm)
        new_candidate_set[adjacency_node] = adj_candidate_vec
    new_sub_graph, new_sub_neg_graph = kg_remove_node(sub_graph, to_cut_node), \
                                       kg_remove_node(neg_sub_graph, to_cut_node)
    cut_node_candidate_set = new_candidate_set.pop(to_cut_node)
    sub_answer = solve_conjunctive(new_sub_graph, new_sub_neg_graph, r_matrix_list, new_candidate_set, conj_tnorm,
                                   exist_tnorm, now_variable)
    return sub_answer


def existential_update(leaf_node, adjacency_node, sub_graph: KnowledgeGraph, neg_sub_graph: KnowledgeGraph,
                       r_matrix_list, leaf_candidates, adj_candidates, conj_tnorm, exist_tnorm) -> dict:
    all_prob_matrix = construct_matrix_list(leaf_node, adjacency_node, sub_graph, neg_sub_graph, r_matrix_list,
                                            conj_tnorm)
    if conj_tnorm == 'product':
        transit_matrix = np.multiply(all_prob_matrix, np.expand_dims(leaf_candidates, axis=-1))
        transit_matrix = np.multiply(transit_matrix, np.expand_dims(adj_candidates, axis=-2))
    elif conj_tnorm == 'Godel':
        transit_matrix = np.maximum(all_prob_matrix, np.expand_dims(leaf_candidates, axis=-1))
        transit_matrix = np.maximum(transit_matrix, np.expand_dims(adj_candidates, axis=-2))
    else:
        raise NotImplementedError
    if exist_tnorm == 'Godel':
        prob_vec = np.asarray(np.amax(transit_matrix, axis=-2)).squeeze()
    else:
        raise NotImplementedError
    return prob_vec


def extend_ans(ans_node, sub_ans_node, sub_graph: KnowledgeGraph, neg_sub_graph: KnowledgeGraph, relation_matrix,
               leaf_candidate, sub_ans, conj_tnorm, exist_tnorm):
    all_prob_matrix = construct_matrix_list(sub_ans_node, ans_node, sub_graph, neg_sub_graph, relation_matrix,
                                            conj_tnorm)
    all_prob_matrix = np.multiply(all_prob_matrix, np.expand_dims(sub_ans, axis=-1))
    if exist_tnorm == 'Godel':
        prob_vec = np.asarray((np.amax(all_prob_matrix, axis=-2))).squeeze()  # prob*vec is 1*n  matrix
    else:
        raise NotImplementedError
    if conj_tnorm == 'product':
        final_ans = leaf_candidate * prob_vec
    elif conj_tnorm == 'Godel':
        final_ans = np.minimum(leaf_candidate, prob_vec)
    else:
        raise NotImplementedError
    return final_ans


def construct_matrix_list(head_node, tail_node, sub_graph, neg_sub_graph, relation_matrix_list, conj_tnorm):
    node_pair, reverse_node_pair = (head_node, tail_node), (tail_node, head_node)
    h2t_relation, t2h_relation = sub_graph.ht2r[node_pair], sub_graph.ht2r[reverse_node_pair]
    h2t_negation, t2h_negation = neg_sub_graph.ht2r[node_pair], neg_sub_graph.ht2r[reverse_node_pair]
    transit_matrix_list = []
    for r in h2t_relation:
        transit_matrix_list.append(relation_matrix_list[r])
    for r in t2h_relation:
        transit_matrix_list.append(relation_matrix_list[r].transpose())
    for r in h2t_negation:
        transit_matrix_list.append(1 - relation_matrix_list[r].toarray())
    for r in t2h_negation:
        transit_matrix_list.append(1 - relation_matrix_list[r].transpose().toarray())
    if conj_tnorm == 'product':
        all_prob_matrix = transit_matrix_list[0]
        for i in range(1, len(transit_matrix_list)):
            all_prob_matrix = all_prob_matrix.multiply(transit_matrix_list[i])
    elif conj_tnorm == 'Godel':
        all_prob_matrix = transit_matrix_list[0]
        for i in range(1, len(transit_matrix_list)):
            all_prob_matrix = all_prob_matrix.minimum(transit_matrix_list[i])
    else:
        raise NotImplementedError
    if issparse(all_prob_matrix):  # n*n sparse matrix or dense matrix (when only one negation edges)
        return all_prob_matrix.toarray()
    else:
        return all_prob_matrix


def solve_EFO1(DNF_formula:DisjunctiveFormula, relation_matrix, conjunctive_tnorm, existential_tnorm, index):
    sub_ans_list = []
    n_entity = relation_matrix[0].shape[0]
    for sub_formula in DNF_formula.formula_list:
        all_candidates = {}
        for term_name in sub_formula.term_dict:
            if sub_formula.has_term_grounded_entity_id_list(term_name):
                all_candidates[term_name] = np.zeros(n_entity)
                all_candidates[term_name][sub_formula.term_grounded_entity_id_dict[term_name][index]] = 1
            else:
                all_candidates[term_name] = np.ones(n_entity)
        sub_graph_edge, sub_graph_negation_edge = [], []
        for pred in sub_formula.predicate_dict.values():
            pred_triples = (pred.head.name, sub_formula.pred_grounded_relation_id_dict[pred.name][index],
                            pred.tail.name)
            if pred.skolem_negation:
                sub_graph_negation_edge.append(pred_triples)
            else:
                sub_graph_edge.append(pred_triples)
        sub_kg_index = KGIndex()
        sub_kg_index.map_entity_name_to_id = {term: 0 for term in sub_formula.term_dict}
        sub_kg = KnowledgeGraph(sub_graph_edge, sub_kg_index)
        neg_kg = KnowledgeGraph(sub_graph_negation_edge, sub_kg_index)
        sub_kg_index.map_relation_name_to_id = {predicate: 0 for predicate in sub_formula.predicate_dict}
        sub_ans = solve_conjunctive(sub_kg, neg_kg, relation_matrix,
                                    all_candidates, conjunctive_tnorm, existential_tnorm, 'f')
        sub_ans_list.append(sub_ans)
    if len(sub_ans_list) == 1:
        return sub_ans_list[0]
    else:
        if conjunctive_tnorm == 'product':
            not_ans = 1 - sub_ans_list[0]
            for i in range(1, len(sub_ans_list)):
                not_ans = not_ans * (1 - sub_ans_list[i])
            return 1 - not_ans
        if conjunctive_tnorm == 'Godel':
            final_ans = sub_ans_list[0]
            for i in range(1, len(sub_ans_list)):
                final_ans = np.maximum(final_ans, sub_ans_list[i])
        else:
            raise NotImplementedError


def compute_single_evaluation(fof, batch_ans, n_entity):
    k = 'f'
    metrics = defaultdict(float)
    for i, single_ans in enumerate(batch_ans):
        argsort = torch.argsort(torch.tensor(single_ans), descending=True)
        ranking = argsort.clone().to(torch.float)
        ranking = ranking.scatter_(0, argsort, torch.arange(n_entity).to(torch.float))
        hard_ans = fof.hard_answer_list[i][k]
        easy_ans = fof.easy_answer_list[i][k]
        num_hard = len(hard_ans)
        num_easy = len(easy_ans)
        cur_ranking = ranking[list(easy_ans) + list(hard_ans)]
        cur_ranking, indices = torch.sort(cur_ranking)
        masks = indices >= num_easy
        answer_list = torch.arange(num_hard + num_easy).to(torch.float)
        cur_ranking = cur_ranking - answer_list + 1
        # filtered setting: +1 for start at 0, -answer_list for ignore other answers
        cur_ranking = cur_ranking[masks]
        # only take indices that belong to the hard answers
        mrr = torch.mean(1. / cur_ranking).item()
        h1 = torch.mean((cur_ranking <= 1).to(torch.float)).item()
        h3 = torch.mean((cur_ranking <= 3).to(torch.float)).item()
        h10 = torch.mean(
            (cur_ranking <= 10).to(torch.float)).item()
        metrics['mrr'] += mrr
        metrics['hit1'] += h1
        metrics['hit3'] += h3
        metrics['hit10'] += h10
    metrics['num_queries'] += len(batch_ans)
    return metrics


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    with open(f'{args.pkl}', 'rb') as data:
        relation_matrix_list = pickle.load(data)
    train_dataloader = QueryAnsweringSeqDataLoader_v2(
        osp.join(args.data_folder, 'test_8_real_EFO1_qaa.json'),
        # size_limit=args.batch_size * 1,
        target_lstr=None,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0)
    writer = Writer(case_name=args.pkl, config=args, log_path='results/log')
    fof_list = train_dataloader.get_fof_list_no_shuffle()
    t = tqdm.tqdm(enumerate(fof_list), total=len(fof_list))
    all_metrics = defaultdict(dict)
    for ifof, fof in t:
        batch_ans_list, metric = [], {}
        for query_index in range(len(fof.easy_answer_list)):
            ans = solve_EFO1(fof, relation_matrix_list, 'product', 'Godel', query_index)
            batch_ans_list.append(ans)
        batch_score = compute_single_evaluation(fof, batch_ans_list, 14505)
        for metric in batch_score:
            if metric not in all_metrics[fof.lstr]:
                all_metrics[fof.lstr][metric] = 0
            all_metrics[fof.lstr][metric] += batch_score[metric]
    for full_formula in all_metrics.keys():
        for log_metric in all_metrics[full_formula].keys():
            if log_metric != 'num_queries':
                all_metrics[full_formula][log_metric] /= all_metrics[full_formula]['num_queries']
    print(all_metrics)
    writer.save_pickle(all_metrics, f"all_metrics.pickle")
