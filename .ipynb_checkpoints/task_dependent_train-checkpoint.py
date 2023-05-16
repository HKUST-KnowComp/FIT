import argparse
from collections import defaultdict
import logging
import os
import os.path as osp
import json

import torch
import tqdm
from torch import nn
import torch.nn.functional as F
import numpy as np

from src.language.tnorm import GodelTNorm, ProductTNorm
from src.pipeline.reasoning_machine import GradientReasoningMachine
from src.structure.knowledge_graph import KnowledgeGraph
from src.structure.knowledge_graph_index import KGIndex
from src.structure.neural_binary_predicate import ComplEx, NeuralBinaryPredicate, TransE
from src.utils.data import (QueryAnsweringSeqDataLoader, QueryAnsweringMixDataLoader,
                            TrainRandomSentencePairDataLoader)
from src.utils.data_util import RaggedBatch

lstr2name = {'r1(s1,f)': '1p', '(r1(s1,e1))&(r2(e1,f))': '2p', '((r1(s1,e1))&(r2(e1,e2)))&(r3(e2,f))': '3p', '(r1(s1,f))&(r2(s2,f))': '2i', '((r1(s1,f))&(r2(s2,f)))&(r3(s3,f))': '3i', '((r1(s1,e1))&(r2(s2,e1)))&(r3(e1,f))': 'ip', '((r1(s1,e1))&(r2(e1,f)))&(r3(s2,f))': 'pi', '(r1(s1,f))&(!(r2(s2,f)))': '2in', '((r1(s1,f))&(r2(s2,f)))&(!(r3(s3,f)))': '3in', '((r1(s1,e1))&(!(r2(s2,e1))))&(r3(e1,f))': 'inp', '((r1(s1,e1))&(r2(e1,f)))&(!(r3(s2,f)))': 'pin', '((r1(s1,e1))&(!(r2(e1,f))))&(r3(s2,f))': 'pni', '(r1(s1,f))|(r2(s2,f))': '2u', '((r1(s1,e1))|(r2(s2,e1))))&(r3(e1,f))': 'up', '!((!(r1(s1,f)))&(!(r2(s2,f))))': '2u-dnf', '!(((!(r1(s1,e1)))|(r2(s2,e1)))&(r3(e1,f)))': 'up-dnf'}

parser = argparse.ArgumentParser()

# base environment
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--output_dir", type=str, default='log')

# input task folder, defines knowledge graph, index, and formulas
parser.add_argument("--task_folder", type=str, default='data/FB15k-237-q2b')

# model, defines the neural binary predicate
parser.add_argument("--model_name", type=str, default='complex')
parser.add_argument("--embedding_dim", type=int, default=500)
parser.add_argument("--margin", type=float, default=20)
parser.add_argument("--p", type=int, default=1)
parser.add_argument("--checkpoint_path")

# optimization
parser.add_argument("--epoch", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--reasoning_rate", type=float, default=1e-1)
parser.add_argument("--objective", type=str, choices=['kvsall', 'noisy', 'none'], default='none')
parser.add_argument("--noisy_sample_size", type=int, default=1024)

parser.add_argument("--metric_margin", type=float, default=50)
parser.add_argument("--sigma", type=float, default=10)
parser.add_argument("--neg_sigma_scaling", type=float, default=1)
parser.add_argument("--v", type=float, default=.9)


def train_lower_bound_noisy_likelihood(
    desc,
    train_dataloader: QueryAnsweringSeqDataLoader,
    nbp:NeuralBinaryPredicate,
    grm: GradientReasoningMachine,
    args):

    optimizer = torch.optim.Adam(nbp.parameters(), args.learning_rate)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.1, patience=50, verbose=True, threshold=1e-3)

    sigma = args.sigma
    trajectory = defaultdict(list)


    fof_list = train_dataloader.get_fof_list()
    t = tqdm.tqdm(enumerate(fof_list), desc=desc, total=len(fof_list))

    for ii, fof in t:
        ####################
        optimizer.zero_grad()
        fetch = grm.reasoning(fof, infer_free=True)
        loss = 0

        pos_tv_list = []
        neg_tv_list = []
        pos_nll_list = []
        neg_nll_list = []
        tv_list = []
        mle_loss_list = []

        # for each formula
        # for fof, fetch in zip(fof, fetched):
        batch_fvar_local_emb_dict = fetch['fvar_local_emb_dict']
        batch_tv = fetch['tv']

        for i, pos_answer_dict in enumerate(fof.easy_answer_list):
            tv = batch_tv[i]
            tv_list.append(tv.item())
            pos_tv = tv
            neg_tv = tv
            for f in pos_answer_dict:
                fvar_emb = batch_fvar_local_emb_dict[f][i]
                pos_answer = pos_answer_dict[f]

                pos_embs = nbp.get_head_emb(pos_answer)
                neg_embs = nbp.get_head_emb(torch.randint(0, nbp.num_entities, (args.noisy_sample_size,)))

                pos_ans_dist = torch.sum((pos_embs - fvar_emb)**2, dim=-1) / sigma ** 2
                pos_ans_tv = torch.exp(- pos_ans_dist)
                pos_tv = grm.tnorm.conjunction(pos_ans_tv, pos_tv)
                pos_tv_list.append(pos_tv.mean().item())

                neg_ans_dist = torch.sum((neg_embs - fvar_emb)**2, dim=-1) / sigma ** 2 / 10
                neg_ans_tv = torch.exp(- neg_ans_dist)
                neg_tv = grm.tnorm.conjunction(neg_ans_tv, neg_tv)
                # neg_tv = neg_ans_tv
                neg_tv_list.append(neg_tv.mean().item())

                pos_nll = - torch.log(pos_tv.min() + 1e-10)
                pos_nll_list.append(pos_nll.item())
                neg_nll = - torch.log(1 - neg_tv.max() + 1e-10)
                neg_nll_list.append(neg_nll.item())

                mle = pos_nll + neg_nll
                mle_loss_list.append(mle)

                    # embedding_reg = 0
                    # for symb in fof.symbol_dict:
                    #     symb_emb = nbp.get_head_emb(
                    #         fof.get_term_grounded_entity_id_list(symb)[i]
                    #     )
                    #     embedding_reg += torch.sum(nbp.regularization(symb_emb) ** 3, -1)

                    # for pred in fof.predicate_dict:
                    #     pred_emb = nbp.get_head_emb(
                    #         fof.get_pred_grounded_relation_id_list(pred)[i]
                    #     )
                    #     embedding_reg += torch.sum(nbp.regularization(pred_emb) ** 3, -1)

                    # embedding_regularization_list.append(embedding_reg)


        mle_loss = torch.mean(torch.stack(mle_loss_list))
        embedding_regularization = 0 # torch.mean(torch.stack(embedding_regularization_list)) * 0.05

        loss = mle_loss + embedding_regularization
        loss.backward()
        optimizer.step()

        pos_tv_mean = np.mean(pos_tv_list)
        neg_tv_mean = np.mean(neg_tv_list)

        ####################
        metric_step = {}
        metric_step['loss'] = loss.item()
        metric_step['pos_tv'] = pos_tv_mean
        metric_step['pos_nll'] = np.mean(pos_nll_list)
        metric_step['neg_tv'] = neg_tv_mean
        metric_step['neg_nll'] = np.mean(neg_nll_list)
        metric_step['tv'] = np.mean(tv_list)
        metric_step['mle_loss'] = mle_loss.item()
        metric_step['sigma'] = sigma
        # metric_step['emb_reg'].append(embedding_regularization.item())

        # if pos_tv_mean < 0.5 and neg_tv_mean < 0.5:
        #     sigma = sigma * (1+ 1e-3)

        # if pos_tv_mean > 0.5 and neg_tv_mean > 0.5:
        #     sigma = sigma * (1- 1e-3)

        # if pos_tv_mean - neg_tv_mean > 0.5:
        #     sigma = sigma * (1- 1e-3)

        # if neg_tv_mean > 0.25:
        #     sigma = sigma * (1 - 1e-3)

        logging.info(f"[{desc}] {json.dumps(metric_step)}")

        postfix = {'step': ii+1}
        for k in metric_step:
            postfix[k] = np.mean(metric_step[k])
            trajectory[k].append(postfix[k])
        postfix['acc_loss'] = np.mean(trajectory['loss'])
        # scheduler.step(postfix['acc_loss'])
        t.set_postfix(postfix)

    t.close()

    metric = {}
    for k in trajectory:
        metric[k] = np.mean(trajectory[k])
    return metric



def train_upper_bound_noisy_likelihood(
    desc,
    train_dataloader: QueryAnsweringSeqDataLoader,
    nbp:NeuralBinaryPredicate,
    grm: GradientReasoningMachine,
    args):

    optimizer = torch.optim.Adam(nbp.parameters(), args.learning_rate)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.1, patience=50, verbose=True, threshold=1e-3)

    sigma = args.sigma
    trajectory = defaultdict(list)


    fof_list = train_dataloader.get_fof_list()
    t = tqdm.tqdm(enumerate(fof_list), desc=desc, total=len(fof_list))

    for ii, fof in t:
        ####################
        optimizer.zero_grad()
        fetch = grm.reasoning(fof, infer_free=True, enable_target=True)
        loss = 0

        pos_tv_list = []
        neg_tv_list = []
        pos_nll_list = []
        neg_nll_list = []
        tv_list = []
        mle_loss_list = []

        # for each formula
        # for fof, fetch in zip(fof, fetched):
        batch_fvar_local_emb_dict = fetch['fvar_local_emb_dict']
        batch_tv = fetch['tv']

        for i, pos_answer_dict in enumerate(fof.easy_answer_list):
            tv = batch_tv[i]
            tv_list.append(tv.item())
            pos_tv = tv
            neg_tv = tv
            for f in pos_answer_dict:
                fvar_emb = batch_fvar_local_emb_dict[f][i]
                pos_answer = pos_answer_dict[f]

                pos_embs = nbp.get_head_emb(pos_answer)
                neg_embs = nbp.get_head_emb(torch.randint(0, nbp.num_entities, (args.noisy_sample_size,)))

                pos_ans_dist = torch.sum((pos_embs - fvar_emb)**2, dim=-1) / sigma ** 2
                pos_ans_tv = torch.exp(- pos_ans_dist)
                pos_tv = grm.tnorm.conjunction(pos_ans_tv, pos_tv)
                pos_tv_list.append(pos_tv.mean().item())

                neg_ans_dist = torch.sum((neg_embs - fvar_emb)**2, dim=-1) / sigma ** 2 / args.neg_sigma_scaling
                neg_ans_tv = torch.exp(- neg_ans_dist)
                neg_tv = grm.tnorm.conjunction(neg_ans_tv, neg_tv)
                # neg_tv = neg_ans_tv
                neg_tv_list.append(neg_tv.mean().item())

                pos_nll = - torch.log(pos_tv + 1e-10).mean()
                pos_nll_list.append(pos_nll.item())
                neg_nll = - torch.log(1 - neg_tv + 1e-10).mean()
                neg_nll_list.append(neg_nll.item())

                mle = pos_nll + neg_nll
                mle_loss_list.append(mle)

                    # embedding_reg = 0
                    # for symb in fof.symbol_dict:
                    #     symb_emb = nbp.get_head_emb(
                    #         fof.get_term_grounded_entity_id_list(symb)[i]
                    #     )
                    #     embedding_reg += torch.sum(nbp.regularization(symb_emb) ** 3, -1)

                    # for pred in fof.predicate_dict:
                    #     pred_emb = nbp.get_head_emb(
                    #         fof.get_pred_grounded_relation_id_list(pred)[i]
                    #     )
                    #     embedding_reg += torch.sum(nbp.regularization(pred_emb) ** 3, -1)

                    # embedding_regularization_list.append(embedding_reg)


        mle_loss = torch.mean(torch.stack(mle_loss_list))
        embedding_regularization = 0 # torch.mean(torch.stack(embedding_regularization_list)) * 0.05

        loss = mle_loss + embedding_regularization
        loss.backward()
        optimizer.step()

        pos_tv_mean = np.mean(pos_tv_list)
        neg_tv_mean = np.mean(neg_tv_list)

        ####################
        metric_step = {}
        metric_step['loss'] = loss.item()
        metric_step['pos_tv'] = pos_tv_mean
        metric_step['pos_nll'] = np.mean(pos_nll_list)
        metric_step['neg_tv'] = neg_tv_mean
        metric_step['neg_nll'] = np.mean(neg_nll_list)
        metric_step['tv'] = np.mean(tv_list)
        metric_step['mle_loss'] = mle_loss.item()
        metric_step['sigma'] = sigma
        # metric_step['emb_reg'].append(embedding_regularization.item())

        # if pos_tv_mean < 0.5 and neg_tv_mean < 0.5:
        #     sigma = sigma * (1+ 1e-3)

        # if pos_tv_mean > 0.5 and neg_tv_mean > 0.5:
        #     sigma = sigma * (1- 1e-3)

        # if pos_tv_mean - neg_tv_mean > 0.5:
        #     sigma = sigma * (1- 1e-3)

        # if neg_tv_mean > 0.25:
        #     sigma = sigma * (1 - 1e-3)

        logging.info(f"[{desc}] {json.dumps(metric_step)}")

        postfix = {'step': ii+1}
        for k in metric_step:
            postfix[k] = np.mean(metric_step[k])
            trajectory[k].append(postfix[k])
        postfix['acc_loss'] = np.mean(trajectory['loss'])
        # scheduler.step(postfix['acc_loss'])
        t.set_postfix(postfix)

    t.close()

    metric = {}
    for k in trajectory:
        metric[k] = np.mean(trajectory[k])
    return metric


def compute_evaluation_scores(fof, batch_entity_rankings, metric):
    k = 'f'
    for i, ranking in enumerate(torch.split(batch_entity_rankings, 1)):
        ranking = ranking.squeeze()
        if fof.hard_answer_list[i]:
        # [1, num_entities]
            hard_answers = torch.tensor(fof.hard_answer_list[i][k],
                                        device=nbp.device)
            hard_answer_rank = ranking[hard_answers]

            # remove better easy answers from its rankings
            if fof.easy_answer_list[i][k]:
                easy_answers = torch.tensor(fof.easy_answer_list[i][k],
                                            device=nbp.device)
                easy_answer_rank = ranking[easy_answers].view(-1, 1)

                num_skipped_answers = torch.sum(
                    hard_answer_rank > easy_answer_rank, dim=0)
                pure_hard_ans_rank = hard_answer_rank - num_skipped_answers
            else:
                pure_hard_ans_rank = hard_answer_rank.squeeze()

        else:
            pure_hard_ans_rank = ranking[
                torch.tensor(fof.easy_answer_list[i][k], device=nbp.device)]

        # remove better hard answers from its ranking
        _reference_hard_ans_rank = pure_hard_ans_rank.reshape(-1, 1)
        num_skipped_answers = torch.sum(
            pure_hard_ans_rank > _reference_hard_ans_rank, dim=0
        )
        pure_hard_ans_rank -= num_skipped_answers.reshape(pure_hard_ans_rank.shape)

        rr = (1 / (1+pure_hard_ans_rank)).detach().cpu().numpy()
        hit1 = (pure_hard_ans_rank < 1).detach().cpu().numpy()
        hit3 =  (pure_hard_ans_rank < 3).detach().cpu().numpy()
        hit10 =  (pure_hard_ans_rank < 10).detach().cpu().numpy()

        metric['mrr'].append(rr.mean())
        metric['hit1'].append(hit1.mean())
        metric['hit3'].append(hit3.mean())
        metric['hit10'].append(hit10.mean())


def evaluate_by_search_emb_then_rank_truth_value(
    desc, dataloader, nbp:NeuralBinaryPredicate, grm: GradientReasoningMachine, target_lstr=[]):
    """
    Evaluation used in CQD, two phase computation
    1. continuous optimiation of embeddings quant. + free
    2. evaluate all sentences with intermediate optimized
    """
    # first level key: lstr
    # second level key: metric name
    metric = defaultdict(lambda: defaultdict(list))
    _fofs = dataloader.get_fof_list()
    # filter the desired lstr
    fofs = []
    for f in _fofs:
        if (((len(target_lstr) == 0) or
                (f.lstr() in target_lstr))
            and len(f.free_variable_dict) == 1):
            fofs.append(f)

    # conduct reasoning
    fetched = grm.reasoning(fofs, infer_free=False)
    with tqdm.tqdm(zip(fofs, fetched)) as t:
        for fof, fof_reasoning_kv in t:
            truth_value_entity_batch = fof_reasoning_kv['tv']  # [num_entities batch_size]
            ranking_score = torch.transpose(truth_value_entity_batch, 0, 1)
            # batch_entity_rankings = nbp.get_all_entity_rankings(batch_est_emb)
            # ranked_entity_ids[ranking] = {entity_id} at the {rankings}-th place
            ranked_entity_ids = torch.argsort(ranking_score, dim=-1, descending=True)
            # entity_rankings[entity_id] = {rankings} of the entity
            batch_entity_rankings = torch.argsort(ranked_entity_ids, dim=-1, descending=False)
            # [batch_size, num_entities]
            compute_evaluation_scores(fof, batch_entity_rankings, metric[fof.lstr()])


            sum_metric = defaultdict(dict)
            for lstr in metric:
                for score_name in metric[lstr]:
                    sum_metric[lstr2name[lstr]][score_name] = float(np.mean(metric[lstr][score_name]))

            postfix = {}
            for name in ['1p', '2p', '3p', '2i', 'inp']:
                if name in sum_metric:
                    postfix[name + '_hit3'] = sum_metric[name]['hit3']
            t.set_postfix(postfix)

    logging.info(f"[{desc}][final] {json.dumps(sum_metric)}")
    torch.cuda.empty_cache()

def evaluate_by_nearest_search(
    desc, dataloader, nbp:NeuralBinaryPredicate, grm: GradientReasoningMachine, target_lstr=[]):
    """
    Evaluation used by nearest neighbor
    1. continuous optimiation of embeddings quant. + free
    2. evaluate all sentences with intermediate optimized
    """
    # first level key: lstr
    # second level key: metric name
    metric = defaultdict(lambda: defaultdict(list))
    _fofs = dataloader.get_fof_list()
    # filter the desired lstr
    fofs = []
    for f in _fofs:
        if (((len(target_lstr) == 0) or
                (f.lstr() in target_lstr))
            and len(f.free_variable_dict) == 1):
            fofs.append(f)

    # conduct reasoning
    with tqdm.tqdm(fofs, desc=desc) as t:
        for fof in t:
            fof_reasoning_kv = grm.reasoning(fof, infer_free=False)
            fvar_emb_dict = fof_reasoning_kv['fvar_local_emb_dict']  # [num_entities batch_size]
            batch_entity_rankings = nbp.get_all_entity_rankings(fvar_emb_dict['f'])
            # [batch_size, num_entities]
            compute_evaluation_scores(fof, batch_entity_rankings, metric[fof.lstr()])
            t.set_postfix({'lstr': fof.lstr()})

        print("sum metric")
        sum_metric = defaultdict(dict)
        for lstr in metric:
            for score_name in metric[lstr]:
                sum_metric[lstr2name[lstr]][score_name] = float(np.mean(metric[lstr][score_name]))

        postfix = {'lstr': fof.lstr()}
        for name in ['1p', '2p', '3p', '2i', 'inp']:
            if name in sum_metric:
                postfix[name + '_hit3'] = sum_metric[name]['hit3']

    logging.info(f"[{desc}][final] {json.dumps(sum_metric)}")
    torch.cuda.empty_cache()


if __name__ == "__main__":

    args = parser.parse_args()
    print(args)
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(filename=osp.join(args.output_dir, 'output.log'),
                        format='%(asctime)s %(message)s',
                        level=logging.INFO,
                        filemode='wt')

    kgidx = KGIndex.load(
        osp.join(args.task_folder, "kgindex.json"))

    if args.model_name.lower() == 'transe':
        nbp_class = TransE
    elif args.model_name.lower() == 'complex':
        nbp_class = ComplEx
    else:
        raise NotImplementedError

    nbp = nbp_class(
        num_entities=kgidx.num_entities,
        num_relations=kgidx.num_relations,
        embedding_dim=args.embedding_dim,
        p=args.p,
        margin=args.margin,
        device=args.device)

    if args.checkpoint_path:
        nbp.load_state_dict(torch.load(args.checkpoint_path))
        print(f"model loaded from {args.checkpoint_path}")

    nbp.to(args.device)

    # this dataloader is for k-vs-all objective. works for noisy v1
    # depreciated
    train_dataloader = QueryAnsweringSeqDataLoader(
        osp.join(args.task_folder, 'train-qaa.json'),
        target_lstr=["r1(s1,f)", "r1(s1,f)&r2(s2,f)"],
        # target_lstr=["r1(s1,f)&r2(s2,f)"],
        # size_limit=1024,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0)

    # for noisy objective v2
    # train_dataloader = TrainNoisyAnswerDataLoader(
    #     osp.join(args.task_folder, 'train-qaa.json'),
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     answer_size=kgidx.num_entities,
    #     noisy_sample_size=args.noisy_sample_size,
    #     num_workers=2)

    valid_dataloader = QueryAnsweringSeqDataLoader(
        osp.join(args.task_folder, 'valid-qaa.json'),
        batch_size=500,
        shuffle=False,
        num_workers=1
    )

    test_dataloader = QueryAnsweringSeqDataLoader(
        osp.join(args.task_folder, 'test-qaa.json'),
        batch_size=500,
        shuffle=False,
        num_workers=1
    )


    # train_epoch_K_verses_All(f"initial from cold start",
                            #    train_dataloader, nbp, grm0, args)
    train_grm = GradientReasoningMachine(
        reasoning_rate=args.reasoning_rate,
        reasoning_steps=10,
        reasoning_optimizer='Adam',
        nbp=nbp,
        tnorm=ProductTNorm,
        sigma=args.sigma)

    eval_grm = GradientReasoningMachine(
        reasoning_rate=args.reasoning_rate,
        reasoning_steps=1000,
        reasoning_optimizer='Adam',
        nbp=nbp,
        tnorm=ProductTNorm)


    evaluate_by_search_emb_then_rank_truth_value(f"evaluate CQD validate set {0}",
                                                 valid_dataloader, nbp, eval_grm)
    evaluate_by_search_emb_then_rank_truth_value(f"evaluate CQD test set {0}",
                                                 test_dataloader, nbp, eval_grm)

    # evaluate_by_nearest_search(f"evaluate NN train epoch {0}",
    #                             train_dataloader, nbp, train_grm)
    evaluate_by_nearest_search(f"evaluate NN validate epoch {0}",
                               valid_dataloader, nbp, train_grm)
    evaluate_by_nearest_search(f"evaluate NN test epoch {0}",
                               test_dataloader, nbp, train_grm)

    eval_only = False
    for e in range(args.epoch):

        if args.objective.lower() == 'noisy':
            # # train_epoch_noisy_v2(f"training epoch {e}",
            #                      train_dataloader, nbp, train_grm, args)
            # train_lower_bound_noisy_likelihood(f"train lower bound noisy", train_dataloader, nbp, train_grm, args)
            train_upper_bound_noisy_likelihood(f"train upper bound noisy", train_dataloader, nbp, train_grm, args)
        else:
            print("no training")
            eval_only = True

        if (e+1) % 1 == 0:
            evaluate_by_nearest_search(f"evaluate NN train epoch {e+1}",
                     train_dataloader, nbp, train_grm)
            evaluate_by_nearest_search(f"evaluate NN validate epoch {e+1}",
                     valid_dataloader, nbp, train_grm)
            evaluate_by_nearest_search(f"evaluate NN test epoch {e+1}",
                     test_dataloader, nbp, train_grm)

            if eval_only:
                break
