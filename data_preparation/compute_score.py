import argparse
import os.path as osp
from math import ceil

import torch

from data_preparation.create_matrix import create_matrix_from_ckpt
from src.structure.knowledge_graph import KnowledgeGraph
from src.structure.knowledge_graph_index import KGIndex

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path", type=str,
                    default='EFO-1_log/real_EFO1/FIT_NELL_EFO1.yaml230926.17:11:09d97349df/750.ckpt')
parser.add_argument("--ckpt_type", type=str, default='FIT', choices=['cqd', 'kge', 'FIT'])
parser.add_argument("--data_folder", type=str, default='data/NELL-EFO1')
parser.add_argument("--cuda", type=int, default=1)
parser.add_argument("--split_num", type=int, default=1)
parser.add_argument("--batch", type=int, default=10)
parser.add_argument("--output_folder", type=str, default='matrix/NELL')
parser.add_argument("--start_rel", type=int, default=340)
parser.add_argument("--end_rel", type=int, default=360)


def compute_batch_score_transe(rel, h_emb, t_emb):
    difference = rel + h_emb - t_emb
    score = -(torch.linalg.norm(difference, dim=-1))
    return score


def compute_batch_score_distmult(rel, h_emb, t_emb):
    score = torch.sum(rel * h_emb * t_emb, dim=-1)
    return score


def compute_batch_score_complex(rel, arg1, arg2, rank):
    rel_real, rel_img = rel[:, :, :rank], rel[:, :, rank:]
    arg1_real, arg1_img = arg1[:, :, :rank], arg1[:, :, rank:]
    arg2_real, arg2_img = arg2[:, :, :rank], arg2[:, :, rank:]

    # [B] Tensor
    score1 = torch.sum(rel_real * arg1_real * arg2_real, -1)
    score2 = torch.sum(rel_real * arg1_img * arg2_img, -1)
    score3 = torch.sum(rel_img * arg1_real * arg2_img, -1)
    score4 = torch.sum(rel_img * arg1_img * arg2_real, -1)
    res = score1 + score2 + score3 - score4
    del score1, score2, score3, score4, rel_real, rel_img, arg1_real, arg1_img, arg2_real, arg2_img

    return res


def create_whole_matrix(rel_id, n_entity, device, ent_emb, rel_emb, observed_kg, threshold, epsilon):
    batch_head = 10
    rel_matrix = torch.zeros((n_entity, n_entity)).to(device)
    head_total_batch = ceil(n_entity / batch_head)
    batch_head_list = torch.chunk(torch.arange(n_entity).to(device), head_total_batch)
    for head_batch_tensor in batch_head_list:
        if head_batch_tensor.ndim == 1:
            head_batch_tensor.unsqueeze_(-1)  # (batch_size, 1)
        batch_head_emb = ent_emb[head_batch_tensor]
        tail_emb = ent_emb.unsqueeze(0)
        # batch_head_emb = batch_head_emb.unsqueeze(-2)
        this_rel_emb = rel_emb[torch.tensor(rel_id, dtype=torch.int, device=device)] \
            .unsqueeze(0).unsqueeze(0)
        batch_score = compute_batch_score_complex(this_rel_emb, batch_head_emb, tail_emb, 1000)
        batch_score = batch_score.squeeze(1)
        batch_prob = torch.softmax(batch_score, dim=-1)
        del tail_emb, batch_head_emb, this_rel_emb
        for batch_index in range(batch_prob.shape[0]):
            head_id = int(head_batch_tensor[batch_index].data)
            observed_tail_set = observed_kg.hr2t[(head_id, rel_id)]
            observed_t_num = len(observed_tail_set)
            scaling = observed_t_num / torch.sum(
                batch_prob[batch_index, list(observed_tail_set)]) if observed_t_num else 1
            scaled_tail_prob = batch_prob[batch_index] * scaling
            sparse_topk_batch_prob = torch.where(
                scaled_tail_prob > threshold, scaled_tail_prob,
                torch.zeros_like(scaled_tail_prob).to(device))
            clamp_batch_prob = torch.clamp(sparse_topk_batch_prob, 0, 1 - epsilon)
            clamp_batch_prob[list(observed_tail_set)] = 1
            rel_matrix[head_id] = clamp_batch_prob
        del batch_prob, batch_score
    sparse_matrix = rel_matrix.to('cpu').to_sparse()
    return sparse_matrix


if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device('cuda:{}'.format(args.cuda))
    data_folder = args.data_folder
    cqd_path = args.ckpt_path
    kgidx = KGIndex.load(osp.join(data_folder, 'kgindex.json'))
    train_kg = KnowledgeGraph.create(
        triple_files=osp.join(data_folder, 'train_kg.tsv'),
        kgindex=kgidx)
    threshold, epsilon = 0.0002, 0.001

    cqd_ckpt = torch.load(cqd_path)
    if args.ckpt_type == 'cqd':
        ent_emb = cqd_ckpt['embeddings.0.weight'].to(device)
        rel_emb = cqd_ckpt['embeddings.1.weight'].to(device)
    elif args.ckpt_type == 'FIT':
        ent_emb = cqd_ckpt['model_parameter']['kge_matrix.ent_emb.weight'].to(device)
        rel_emb = cqd_ckpt['model_parameter']['kge_matrix.rel_emb.weight'].to(device)
    else:
        model_param = cqd_ckpt['model'][0]
        ent_emb = model_param['_entity_embedder.embeddings.weight']
        rel_emb = model_param['_relation_embedder.embeddings.weight']
    n_rel, n_ent = rel_emb.shape[0], ent_emb.shape[0]
    split_num = args.split_num
    batch_head = args.batch
    head_total_batch = ceil(n_ent / batch_head)
    sparse_list = []
    for relation_id in range(args.start_rel, args.end_rel):
        sparse_matrix = create_whole_matrix(relation_id, n_ent, device, ent_emb, rel_emb, train_kg, threshold, epsilon)
        sparse_list.append(sparse_matrix)
        if relation_id % 10 == 0 or relation_id == args.end_rel - 1:
            torch.save(sparse_list, osp.join(args.output_folder,
                                             f'new_{args.start_rel}_{args.end_rel}_matrix_{threshold}_{epsilon}.ckpt'))
        print(f"rel{relation_id} saved")
