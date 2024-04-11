import copy
import os.path

from math import ceil
import torch
from torch import nn

from src.structure.knowledge_graph import KnowledgeGraph, kg_remove_node
from src.structure.knowledge_graph_index import KGIndex
from Symbloic_FIT import find_leaf_node, find_enumerate_node, solve_EFO1_new
from data_preparation.compute_score import compute_batch_score_complex


class Kge_finetune(nn.Module):
    def __init__(self, observed_kg, ent_emb, rel_emb, relation_matrix_list, freeze_ent,
                 score_function, head_batch, sparse,  threshold,
                 epsilon, device):
        super(Kge_finetune, self).__init__()
        self.observed_kg = observed_kg
        self.freeze_ent = freeze_ent
        self.ent_emb = nn.Embedding.from_pretrained(ent_emb, freeze=freeze_ent)
        self.rel_emb = nn.Embedding.from_pretrained(rel_emb, freeze=False)
        self.rank = rel_emb.shape[1] // 2
        self.score_function = score_function
        self.head_batch = head_batch
        self.sparse = sparse
        self.threshold, self.epsilon = threshold, epsilon
        self.device = device
        self.n_entity, self.ent_dim = ent_emb.shape[0], ent_emb.shape[1]
        self.pretrained_rel_matrix_list = relation_matrix_list

    def forward(self, rel_id, head_ent_vec, train_mask=False):
        """
            rel_id: number
            head_ent_vec: (ent_dim)
            train_mask: bool, to determine whether we set the probabilities of training edges to 1
        """
        nonzero_index = torch.nonzero(head_ent_vec)  # (nonzero_num, 1)
        nonzero_head_num = nonzero_index.shape[0]
        nonzero_head_matrix = torch.zeros((nonzero_head_num, self.n_entity)).to(self.device)
        if not nonzero_head_num:
            return nonzero_head_matrix
        batch_head_list = torch.chunk(nonzero_index.squeeze(-1), nonzero_head_num)
        for head_id, head_batch_tensor in enumerate(batch_head_list):
            batch_head_emb = self.ent_emb(head_batch_tensor)
            tail_emb = self.ent_emb.weight.unsqueeze(0)
            batch_head_emb = batch_head_emb.unsqueeze(-2)
            this_rel_emb = self.rel_emb(torch.tensor(rel_id, dtype=torch.int, device=self.device))\
                .unsqueeze(0).unsqueeze(0)
            batch_score = self.score_function(this_rel_emb, batch_head_emb, tail_emb, self.rank)
            batch_score = batch_score.squeeze()
            batch_prob = torch.softmax(batch_score, dim=-1)
            observed_tail_set = self.observed_kg.hr2t[(int(head_batch_tensor.data), rel_id)]
            observed_t_num = len(observed_tail_set)
            scaling = observed_t_num/torch.sum(batch_prob[list(observed_tail_set)]) if observed_t_num else 1
            del tail_emb, batch_score, batch_head_emb, this_rel_emb
            scaled_tail_prob = batch_prob * scaling
            if self.sparse:
                sparse_topk_batch_prob = torch.where(
                    scaled_tail_prob > self.threshold, scaled_tail_prob, torch.zeros_like(scaled_tail_prob).to(self.device))
            else:
                sparse_topk_batch_prob = scaled_tail_prob
            if train_mask:
                clamp_batch_prob = torch.clamp(sparse_topk_batch_prob, 0, 1 - self.epsilon)
            else:
                clamp_batch_prob = torch.clamp(sparse_topk_batch_prob, 0, 1)
            if train_mask:
                clamp_batch_prob[list(observed_tail_set)] = 1
                # pretrain_rel_matrix = self.pretrained_rel_matrix_list[rel_id].to_dense()
                # assert torch.all(pretrain_rel_matrix[head_batch_tensor] == clamp_batch_prob)
            nonzero_head_matrix[head_id] = clamp_batch_prob
        return nonzero_head_matrix

    def create_whole_matrix(self, rel_id):
        batch_head = self.head_batch
        rel_matrix = torch.zeros((self.n_entity, self.n_entity)).to(self.device)
        head_total_batch = ceil(self.n_entity / batch_head)
        batch_head_list = torch.chunk(torch.arange(self.n_entity).to(self.device), head_total_batch)
        for head_batch_tensor in batch_head_list:
            if head_batch_tensor.ndim == 1:
                head_batch_tensor.unsqueeze_(-1)  # (batch_size, 1)
            batch_head_emb = self.ent_emb(head_batch_tensor)
            tail_emb = self.ent_emb.weight.unsqueeze(0)
            # batch_head_emb = batch_head_emb.unsqueeze(-2)
            this_rel_emb = self.rel_emb(torch.tensor(rel_id, dtype=torch.int, device=self.device)) \
                .unsqueeze(0).unsqueeze(0)
            batch_score = self.score_function(this_rel_emb, batch_head_emb, tail_emb, self.rank)
            batch_score = batch_score.squeeze(1)
            batch_prob = torch.softmax(batch_score, dim=-1)
            del tail_emb, batch_head_emb, this_rel_emb
            for batch_index in range(batch_prob.shape[0]):
                head_id = int(head_batch_tensor[batch_index].data)
                observed_tail_set = self.observed_kg.hr2t[(head_id, rel_id)]
                observed_t_num = len(observed_tail_set)
                scaling = observed_t_num / torch.sum(batch_prob[batch_index, list(observed_tail_set)]) if observed_t_num else 1
                scaled_tail_prob = batch_prob[batch_index] * scaling
                sparse_topk_batch_prob = torch.where(
                        scaled_tail_prob > self.threshold, scaled_tail_prob,
                        torch.zeros_like(scaled_tail_prob).to(self.device))
                clamp_batch_prob = torch.clamp(sparse_topk_batch_prob, 0, 1-self.epsilon)
                clamp_batch_prob[list(observed_tail_set)] = 1
                rel_matrix[head_id] = clamp_batch_prob
            del batch_prob, batch_score
        sparse_matrix = rel_matrix.to('cpu').to_sparse()
        return sparse_matrix


class FIT_finetune(nn.Module):
    """
    Fuzzy Inference with Neural Truth value, we finetune the KGE model to answer complex queries
    """
    name = "FIT"

    def __init__(self, n_entity, n_relation, freeze_ent, negative_sample_size, train_kg: KnowledgeGraph, kge_path: str,
                 kge: str, matrix_path: str,
                 c_norm, e_norm, max_enumeration,
                 head_batch, sparse, threshold, epsilon, device):
        super(FIT_finetune, self).__init__()
        self.kg = train_kg
        self.kge_path = kge_path
        self.n_entity, self.n_relation = n_entity, n_relation
        self.freeze_ent = freeze_ent
        self.negative_sample_size = negative_sample_size
        self.c_norm, self.e_norm = c_norm, e_norm
        self.max_enumeration = max_enumeration
        self.kge = kge
        self.matrix_path = matrix_path
        self.head_batch = head_batch
        self.sparse = sparse
        self.threshold, self.epsilon = threshold, epsilon
        self.device = device
        if self.kge == 'complex':
            self.score_function = compute_batch_score_complex
        kge_ckpt = torch.load(kge_path)
        # relation_matrix_list = torch.load(self.matrix_path)
        # for i in range(len(relation_matrix_list)):
        #     relation_matrix_list[i] = relation_matrix_list[i].to(self.device)
        ent_emb = kge_ckpt['_entity_embedding.weight'].to(device)
        rel_emb = kge_ckpt['_relation_embedding.weight'].to(device)
        self.kge_matrix = Kge_finetune(self.kg, ent_emb, rel_emb, None, self.freeze_ent,
                                    self.score_function, self.head_batch,  self.sparse,
                                    self.threshold, self.epsilon, self.device)
        if self.matrix_path and os.path.exists(self.matrix_path):
            self.stored_matrix_list = torch.load(self.matrix_path)
            self.exist_matrix = True
        else:
            self.stored_matrix_list = []
            self.exist_matrix = False

    def solve_fof(self, fof, mode):
        batch_ans_list = []
        if 'e' not in fof.formula:
            for query_index in range(len(fof.pred_grounded_relation_id_dict['r1'])):
                ans = neural_solve_EFO1_X(fof, self.kge_matrix, self.c_norm, self.e_norm, query_index, self.device,
                                          self.max_enumeration, mode)
                batch_ans_list.append(ans)
        else:
            for query_index in range(len(fof.pred_grounded_relation_id_dict['r1'])):
                ans = solve_EFO1_new(fof, self.stored_matrix_list, self.c_norm, self.e_norm, query_index, self.device,
                                          self.max_enumeration)
                batch_ans_list.append(ans)

        batch_ans_tensor = torch.stack(batch_ans_list, dim=0)
        return batch_ans_tensor

    def criterion(self, pred_ans, answer_set):
        ans_vec_list = []
        base_num = 4
        subsampling_weight = torch.zeros(len(answer_set)).to(self.device)
        for i in range(len(answer_set)):
            ans_vec = torch.zeros_like(pred_ans[i], device=self.device)
            ans_vec.scatter_(0, torch.tensor(answer_set[i], device=self.device),
                             torch.ones(len(answer_set[i]), device=self.device))
            ans_vec_list.append(ans_vec)
            subsampling_weight[i] = len(answer_set[i]) + base_num
        subsampling_weight = torch.sqrt(1 / subsampling_weight)
        ans_vec_tensor = torch.stack(ans_vec_list, dim=0)

        return pred_ans, ans_vec_tensor, subsampling_weight

    def compute_all_entity_logit(self, pred_emb, union=False):
        return pred_emb

    def construct_all_matrices(self):
        with torch.no_grad():
            if not self.matrix_path:
                return None
            else:
                if self.exist_matrix:
                    pass
                else:
                    self.stored_matrix_list = []
            for rel_id in range(self.n_relation):
                if rel_id < len(self.stored_matrix_list):
                    continue
                rel_matrix = self.kge_matrix.create_whole_matrix(rel_id)
                self.stored_matrix_list.append(rel_matrix)
                print('finish rel {}'.format(rel_id))
                if rel_id % 5 == 0 or rel_id == self.n_relation - 1:
                    torch.save(self.stored_matrix_list, self.matrix_path)
            for rel in range(len(self.stored_matrix_list)):
                if self.n_entity > 60000:
                    torch.set_default_dtype(torch.float16)
                    self.stored_matrix_list[rel] = self.stored_matrix_list[rel].to(dtype=torch.float16).to(self.device)
                else:
                    self.stored_matrix_list[rel] = self.stored_matrix_list[rel].to(self.device)


def neural_solve_EFO1_X(DNF_formula, kgematrix: Kge_finetune, conjunctive_tnorm, existential_tnorm, index, device,
                      max_enumeration, mode):
    torch.cuda.empty_cache()
    sub_ans_list = []
    n_entity = kgematrix.n_entity
    for sub_formula in DNF_formula.formula_list:
        all_candidates = {}
        for term_name in sub_formula.term_dict:
            if sub_formula.has_term_grounded_entity_id_list(term_name):
                all_candidates[term_name] = torch.zeros(n_entity).to(device)
                all_candidates[term_name][sub_formula.term_grounded_entity_id_dict[term_name][index]] = 1
            else:
                all_candidates[term_name] = torch.ones(n_entity).to(device)
        sub_graph_edge, sub_graph_negation_edge = [], []
        for pred in sub_formula.predicate_dict.values():
            pred_triples = (pred.head.name, sub_formula.pred_grounded_relation_id_dict[pred.name][index],
                            pred.tail.name)
            if pred.negated:
                sub_graph_negation_edge.append(pred_triples)
            else:
                sub_graph_edge.append(pred_triples)
        sub_kg_index = KGIndex()
        sub_kg_index.map_entity_name_to_id = {term: 0 for term in sub_formula.term_dict}
        sub_kg = KnowledgeGraph(sub_graph_edge, sub_kg_index)
        neg_kg = KnowledgeGraph(sub_graph_negation_edge, sub_kg_index)
        sub_kg_index.map_relation_name_to_id = {predicate: 0 for predicate in sub_formula.predicate_dict}
        sub_ans = neural_solve_conjunctive(sub_kg, neg_kg, kgematrix,
                                           all_candidates, conjunctive_tnorm, existential_tnorm, 'f1', device,
                                           max_enumeration, mode)
        sub_ans_list.append(sub_ans)
    if len(sub_ans_list) == 1:
        return sub_ans_list[0]
    else:
        if conjunctive_tnorm == 'product':
            not_ans = 1 - sub_ans_list[0]
            for i in range(1, len(sub_ans_list)):
                not_ans = not_ans * (1 - sub_ans_list[i])
            return 1 - not_ans
        elif conjunctive_tnorm == 'Godel':
            final_ans = sub_ans_list[0]
            for i in range(1, len(sub_ans_list)):
                final_ans = torch.maximum(final_ans, sub_ans_list[i])
            return final_ans
        else:
            raise NotImplementedError


def construct_matrix_dynamic(head_node, tail_node, sub_graph, neg_sub_graph, kge_matrix: Kge_finetune, conj_tnorm,
                             head_ent_vec, mode='train'):
    """
    Use head node to update tail node
    """
    node_pair, reverse_node_pair = (head_node, tail_node), (tail_node, head_node)
    h2t_relation, t2h_relation = sub_graph.ht2r[node_pair], sub_graph.ht2r[reverse_node_pair]
    h2t_negation, t2h_negation = neg_sub_graph.ht2r[node_pair], neg_sub_graph.ht2r[reverse_node_pair]
    transit_matrix_list = []
    for r in h2t_relation:
        transit_matrix_list.append(kge_matrix.forward(r, head_ent_vec,
                                                      train_mask=False if mode == 'train' else True))
    for r in t2h_relation:
        transit_matrix_list.append(kge_matrix.forward(r, head_ent_vec, train_mask=False if mode == 'train' else True).transpose(-2, -1))
    for r in h2t_negation:
        transit_matrix_list.append(1 - kge_matrix.forward(r, head_ent_vec,
                                                          train_mask=False if mode == 'train' else True))
    for r in t2h_negation:
        transit_matrix_list.append(1 - kge_matrix.forward(r, head_ent_vec,
                                                          train_mask=False if mode == 'train' else True).transpose(-2, -1))
    if conj_tnorm == 'product':
        all_prob_matrix = transit_matrix_list[0]
        for i in range(1, len(transit_matrix_list)):
            if all_prob_matrix.is_sparse and not transit_matrix_list[i].is_sparse:
                all_prob_matrix = all_prob_matrix.to_dense().multiply(transit_matrix_list[i])
            else:
                all_prob_matrix = all_prob_matrix.multiply(transit_matrix_list[i])
    elif conj_tnorm == 'Godel':
        all_prob_matrix = transit_matrix_list[0].to_dense() \
            if transit_matrix_list[0].is_sparse else transit_matrix_list[0]
        for i in range(1, len(transit_matrix_list)):
            if transit_matrix_list[i].is_sparse:
                all_prob_matrix = torch.minimum(all_prob_matrix, transit_matrix_list[i].to_dense())
            else:
                all_prob_matrix = torch.minimum(all_prob_matrix, transit_matrix_list[i])
    else:
        raise NotImplementedError

    if all_prob_matrix.is_sparse:  # n*n sparse matrix or dense matrix (when only one negation edges)
        return all_prob_matrix.to_dense()
    else:
        return all_prob_matrix


def neural_extend_ans(ans_node, sub_ans_node, sub_graph: KnowledgeGraph, neg_sub_graph: KnowledgeGraph,
                      kge_matrix: Kge_finetune, leaf_candidate, sub_ans, conj_tnorm, exist_tnorm, mode):
    all_prob_matrix = construct_matrix_dynamic(sub_ans_node, ans_node, sub_graph, neg_sub_graph, kge_matrix,
                                               conj_tnorm, sub_ans, mode=mode)
    if conj_tnorm == 'product':
        all_prob_matrix.mul_(sub_ans.unsqueeze(-1))
    elif conj_tnorm == 'Godel':
        all_prob_matrix = torch.minimum(all_prob_matrix, sub_ans.unsqueeze(-1))
    else:
        raise NotImplementedError
    if exist_tnorm == 'Godel':
        prob_vec = (torch.amax(all_prob_matrix, dim=-2)).squeeze()  # prob*vec is 1*n  matrix
        del all_prob_matrix
    elif exist_tnorm == 'product':
        prob_vec = 1 - torch.prod(1 - all_prob_matrix, dim=-2)
    else:
        raise NotImplementedError
    if conj_tnorm == 'product':
        prob_vec = leaf_candidate * prob_vec
    elif conj_tnorm == 'Godel':
        prob_vec = torch.minimum(leaf_candidate, prob_vec)
    else:
        raise NotImplementedError
    return prob_vec


def neural_existential_update(leaf_node, adjacency_node, sub_graph: KnowledgeGraph, neg_sub_graph: KnowledgeGraph,
                              kge_matrix, leaf_candidates, adj_candidates, conj_tnorm, exist_tnorm, mode) -> dict:
    nonzero_head_matrix = construct_matrix_dynamic(leaf_node, adjacency_node, sub_graph, neg_sub_graph, kge_matrix,
                                               conj_tnorm, leaf_candidates, mode=mode)
    leaf_nonzero_index = torch.nonzero(leaf_candidates)[0]
    if conj_tnorm == 'product':
        # assert torch.count_nonzero(leaf_candidates) == 1
        nonzero_head_matrix.mul_(leaf_candidates[leaf_nonzero_index].unsqueeze(-1))
        nonzero_head_matrix.mul_(adj_candidates.unsqueeze(-2))
    elif conj_tnorm == 'Godel':
        nonzero_head_matrix = torch.minimum(nonzero_head_matrix, leaf_candidates[leaf_nonzero_index].unsqueeze(-1))
        nonzero_head_matrix = torch.minimum(nonzero_head_matrix, adj_candidates.unsqueeze(-2))
    else:
        raise NotImplementedError
    if exist_tnorm == 'Godel':
        prob_vec = torch.amax(nonzero_head_matrix, dim=-2).squeeze()
    else:
        raise NotImplementedError
    return prob_vec


def neural_cut_node_sub_problem(to_cut_node, adjacency_node_list, sub_graph: KnowledgeGraph,
                                neg_sub_graph: KnowledgeGraph,
                                kge_matrix: Kge_finetune, now_candidate_set, conj_tnorm, exist_tnorm, now_variable, device,
                                max_enumeration, mode):
    # now_candidate_set = copy.deepcopy(now_candidate_set)
    for adjacency_node in adjacency_node_list:
        adj_candidate_vec = neural_existential_update(to_cut_node, adjacency_node, sub_graph, neg_sub_graph, kge_matrix,
                                                      now_candidate_set[to_cut_node], now_candidate_set[adjacency_node],
                                                      conj_tnorm, exist_tnorm, mode)
        now_candidate_set[adjacency_node] = adj_candidate_vec
    new_sub_graph, new_sub_neg_graph = kg_remove_node(sub_graph, to_cut_node), \
                                       kg_remove_node(neg_sub_graph, to_cut_node)
    now_candidate_set.pop(to_cut_node)
    sub_answer = neural_solve_conjunctive(new_sub_graph, new_sub_neg_graph, kge_matrix, now_candidate_set, conj_tnorm,
                                          exist_tnorm, now_variable, device, max_enumeration, mode)
    return sub_answer



def neural_solve_conjunctive(positive_graph: KnowledgeGraph, negative_graph: KnowledgeGraph, kgematrix: Kge_finetune,
                             now_candidate_set: dict, conjunctive_tnorm, existential_tnorm, now_variable, device,
                             max_enumeration, mode):
    n_entity = kgematrix.n_entity
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
            sub_ans = neural_solve_conjunctive(sub_pos_g, sub_neg_g, kgematrix, now_candidate_set,
                                               conjunctive_tnorm, existential_tnorm, next_variable, device,
                                               max_enumeration, mode)
            final_ans = neural_extend_ans(now_leaf_node, adjacency_node, positive_graph, negative_graph, kgematrix,
                                          now_candidate_set[now_leaf_node], sub_ans, conjunctive_tnorm,
                                          existential_tnorm, mode)
            return final_ans
        else:
            answer = neural_cut_node_sub_problem(now_leaf_node, adjacency_node_list, positive_graph, negative_graph,
                                                 kgematrix, now_candidate_set, conjunctive_tnorm, existential_tnorm,
                                                 now_variable, device, max_enumeration, mode)
            return answer
    else:
        to_enumerate_node, adjacency_node_list = find_enumerate_node(positive_graph, negative_graph, now_candidate_set,
                                                                     now_variable)
        if max_enumeration:  # TODO: can not be set to 0, need to update
            easy_candidate = torch.count_nonzero(now_candidate_set[to_enumerate_node] == 1)
            enumeration_num = torch.count_nonzero(now_candidate_set[to_enumerate_node])
            max_enumeration_here = max_enumeration + easy_candidate
            to_enumerate_candidates = torch.argsort(now_candidate_set[to_enumerate_node],
                                                    descending=True)[:min(max_enumeration_here, enumeration_num)]
        else:
            to_enumerate_candidates = now_candidate_set[to_enumerate_node].nonzero()
        this_node_candidates = copy.deepcopy(now_candidate_set[to_enumerate_node])
        all_enumerate_ans = torch.zeros((to_enumerate_candidates.shape[0], n_entity)).to(device)
        if to_enumerate_candidates.shape[0] == 0:
            return torch.zeros(n_entity).to(device)
        for i, enumerate_candidate in enumerate(to_enumerate_candidates):
            single_candidate = torch.zeros_like(now_candidate_set[to_enumerate_node]).to(device)
            candidate_truth_value = this_node_candidates[enumerate_candidate]
            single_candidate[enumerate_candidate] = 1
            now_candidate_set[to_enumerate_node] = single_candidate
            answer = neural_cut_node_sub_problem(to_enumerate_node, adjacency_node_list, positive_graph, negative_graph,
                                                 kgematrix, now_candidate_set, conjunctive_tnorm, existential_tnorm,
                                                 now_variable, device, max_enumeration, mode)
            if conjunctive_tnorm == 'product':
                enumerate_ans = candidate_truth_value * answer
            elif conjunctive_tnorm == 'Godel':
                enumerate_ans = torch.minimum(candidate_truth_value, answer)
            else:
                raise NotImplementedError
            all_enumerate_ans[i] = enumerate_ans
        if existential_tnorm == 'Godel':
            final_ans = torch.amax(all_enumerate_ans, dim=-2)
        else:
            raise NotImplementedError
        return final_ans


