from torch.utils.data import DataLoader

from .abstract_learner import Learner, LearnerForwardOutput
from .sampler import lcwa_negative_sampling, rel_negative_sampling
from ..structure import KnowledgeGraph, NeuralBinaryPredicate


class IsomorphicLearner(Learner):
    def __init__(self,
                 kg: KnowledgeGraph,
                 nbp: NeuralBinaryPredicate,
                 **kwargs):
        self.kg = kg
        self.nbp = nbp
        self.device = self.nbp.device

    def get_data_iterator(self, **kwargs):
        it = DataLoader(self.kg.triples, **kwargs)
        for phead, prel, ptail in it:
            yield (phead.to(self.device).view(-1, 1),
                   prel.to(self.device).view(-1, 1),
                   ptail.to(self.device).view(-1, 1))

    def forward(self, batch_input, num_neg_samples=1, strategy='lcwa', margin=1):
        """
        In this case we assume the batch input is a list of 3 tensors
        """
        phead, prel, ptail = batch_input

        assert 'lcwa' in strategy
        nhead, ntail = lcwa_negative_sampling(
            phead_id_ten=phead.squeeze(),
            ptail_id_ten=ptail.squeeze(),
            num_neg_samples=num_neg_samples,
            num_entities=self.kg.num_entities)

        pos_scores = self.nbp.batch_predicate_score(
            [phead, prel, ptail])
        neg_scores = self.nbp.batch_predicate_score(
            [nhead, prel, ntail])

        if 'rel' in strategy:
            pass

        output = LearnerForwardOutput(
            pos_score=pos_scores,
            pos_prob=self.nbp.score2truth_value(pos_scores),
            neg_score=neg_scores,
            neg_prob=self.nbp.score2truth_value(neg_scores),
        )

        return output
