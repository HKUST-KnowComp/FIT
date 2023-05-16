from typing import List
import random
import torch

from .abstract_learner import Learner
from ..utils.data_util import tensorize_batch_entities
from ..structure import KnowledgeGraph, NeuralBinaryPredicate


class BatchedEFG:
    def __init__(self,
                 finite_model: KnowledgeGraph,
                 neural_model: NeuralBinaryPredicate,
                 efg_round=5,
                 efg_rand_thr=0.5,
                 k_neural=5,
                 k_subgraph=5,
                 **kwargs):

        self.finite_model = finite_model
        self.neural_model = neural_model

        assert self.finite_model.device == self.neural_model.device

        self.device = self.finite_model.device
        self.round = efg_round
        self.k_neural = k_neural
        self.k_subgraph = k_subgraph
        self.efg_rand_thr = efg_rand_thr

    def play(self, begin_batch_entities: List[int], spoiler_mode=None):
        """
        play the EFG with the batch of very begining entities
        the round is justified by this
        at the beginning of k-th round, there are k entities
        at the end of k-th round, there are k+1 entities
        """
        # prepare initial entities
        entity_tensor = tensorize_batch_entities(begin_batch_entities)
        batch_size = entity_tensor.size(0)

        round_mask = torch.ones(size=(batch_size, self.round+1),
                                dtype=torch.long,
                                device=self.device)

        # prepare spolier argument
        if spoiler_mode is None:
            spoiler_mode = 'random'
            spoiler_args = {'threshold': self.efg_rand_thr}

        # inside the game
        for i in range(1, self.round + 1):
            new_batch_entity, _round_mask = self._spoiler_step(
                batch_entities, round_mask[:, i-1].detach().clone(),
                mode=spoiler_mode, **spoiler_args)
            batch_entities = torch.cat([batch_entities, new_batch_entity],
                                       dim=-1)
            round_mask[:, i] = _round_mask

        # get sub_graph from self.finite_model
        outputs = self.finite_model.get_subgraph(batch_entities)
        return outputs

    def _spoiler_step(self, batch_entities, round_mask, mode, **kwargs):
        """
        the entry point for each spoiler step
            Args:
                - batch_entities: batch entities to pass
                - round_mask: vector: batch size where 1 indicates runnable
                - mode: string: indicate how the spolier select on two models
                - kwargs: dict: parameters for the action on two sides
        """
        if mode == 'random':
            return self._spoiler_random_step(batch_entities, round_mask, **kwargs)
        else:
            raise NotImplementedError(
                f"spoiler step mode {mode} is not implemented")

    def _spoiler_random_step(self, batch_entities, round_mask, threshold=0.5, **kwargs):
        """
        Random pick the side
        """
        rand = random.random()
        if rand < threshold:
            new_entity_id = self._spoiler_act_on_finite_model(
                batch_entities, round_mask, *kwargs)
        else:
            new_entity_id = self._spoiler_act_on_neural_model(
                batch_entities, round_mask, **kwargs)
        return new_entity_id

    def _spoiler_act_on_finite_model(self, batch_entities, round_mask):
        batch_size = batch_entities.size(0)
        first_index = torch.arange(batch_size, device=self.device)

        # get triples whose head is new entities
        ragged_head_triples = self.finite_model.get_neighbor_triples_by_target(
            batch_entities, filtered=True)

        # make the batch head scores
        ragged_head_scores = ragged_head_triples.run_ops_on_flatten(
            opfunc=lambda x: self.neural_model.batch_predicate_score(x))
        batch_head_scores = ragged_head_scores.to_dense_matrix(
            padding_value=torch.inf)
        # make the batch head ids
        ragged_head_ids = ragged_head_triples.run_ops_on_flatten(
            opfunc=lambda x: x[:, 0]
        )
        batch_head_ids = ragged_head_ids.to_dense_matrix(
            padding_value=-1)

        head_min_index = torch.argmin(batch_head_scores, dim=-1)
        head_min_score = batch_head_scores[first_index, head_min_index]
        head_min_entity = batch_head_ids[first_index, head_min_index]

        # get triples whose tail is new entities
        ragged_tail_triples = self.finite_model.get_neighbor_triples_by_target(
            batch_entities, filtered=True)

        # make the batch tail scores
        ragged_tail_scores = ragged_tail_triples.run_ops_on_flatten(
            opfunc=lambda x: self.neural_model.batch_predicate_score(x))
        batch_tail_scores = ragged_tail_scores.to_dense_matrix(
            padding_value=torch.inf)
        # make the batch tail ids
        ragged_tail_ids = ragged_tail_triples.run_ops_on_flatten(
            opfunc=lambda x: x[:, 2]
        )
        batch_tail_ids = ragged_tail_ids.to_dense_matrix(
            padding_value=-1)

        tail_min_index = torch.argmin(batch_tail_scores, dim=-1)
        tail_min_score = batch_tail_scores[first_index, tail_min_index]
        tail_min_entity = batch_tail_ids[first_index, tail_min_index]

        batch_new_entity = torch.where(
            head_min_score < tail_min_score, head_min_entity, tail_min_entity)

        return batch_new_entity, batch_new_entity >= 0

    def _spoiler_act_on_neural_model(self, batch_entities, round_mask):

        Thead, Trel, Ttail = self.finite_model._get_non_neightbor_triples(
            batch_entities, k=self.k_neural, reverse=False)
        Tscores = self.neural_model.batch_pred_score(
            Thead, Trel, Ttail).squeeze()

        Hhead, Hrel, Htail = self.finite_model._get_non_neightbor_triples(
            batch_entities, k=self.k_neural, reverse=True)
        Hscores = self.neural_model.batch_pred_score(
            Hhead, Hrel, Htail).squeeze()

        batch_new_entity = batch_entities[:, -1].detach().clone().view(-1, 1)

        # adhoc may be improved by ragged tensor if one uses TF
        Tmax_index = Tscores.argmax(-1)
        Hmax_index = Hscores.argmax(-1)

        first_indices = torch.arange(
            batch_entities.size(0), device=self.device)

        Tmax_scores = Tscores[first_indices, Tmax_index]
        Hmax_scores = Tscores[first_indices, Hmax_index]

        batch_new_entity = torch.where(
            Tmax_scores > Hmax_scores, Tmax_index, Hmax_index)

        return batch_new_entity.view(-1, 1), torch.ones_like(batch_new_entity)


class ElementaryLearner(Learner):
    def __init__(self,
                 kg: KnowledgeGraph,
                 nbp: NeuralBinaryPredicate,
                 round,
                 **kwargs):
        self.kg = kg
        self.nbp = nbp
        self.device = nbp.device
        self.round = round

        self.efg = BatchedEFG(self.kg, self.nbp)

    def get_data_iterator(self):
        return super().get_data_iterator()

    def forward(self, batch_input, num_negative_samples, aggregate_level='triple'):

        batch_entity_set = self.efg.play(batch_input)

        pass

    # def random_training_triple(self):
    #     entity_list = list(self.finite_model.entity_set)
    #     elist = random.sample(
    #         entity_list, k=self.batch_size // self.round)

    #     output = self.efg.play(begin_entity_id_list=elist)

    #     return output

    # def get_next_batch_of_triples(self, epoch=False):
    #     if epoch:
    #         try:
    #             batch = next(self.node_iter)
    #         except StopIteration:
    #             self.num_epoch += 1
    #             print("train epoch", self.num_epoch)
    #             self.node_iter = self.get_train_node_efg_iterator()
    #             batch = next(self.node_iter)
    #     else:
    #         batch = self.random_training_triple()
    #     return batch

    # def learning_step(self, optimizer, log=True):
    #     if log:
    #         log_dict = {}

    #     optimizer.zero_grad()

    #     output = self.get_next_batch_of_triples()

    #     loss = self.neural_model.compute_efg_nce_loss(
    #         **output, k_nce=self.k_nce, margin=self.margin)
    #     # loss = self.neural_model.compute_efg_pair_loss(**output)

    #     loss.backward()
    #     optimizer.step()

    #     if log:
    #         log_dict['loss'] = loss.item()
    #         log_dict['num_triples'] = len(output['subgraph_flat_triples'])
    #         return log_dict
