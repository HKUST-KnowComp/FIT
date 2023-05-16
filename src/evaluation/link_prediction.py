from collections import defaultdict

from tqdm import tqdm
import torch
import numpy as np

from .abstract_task import AbstractTask

from ..structure import KnowledgeGraph, NeuralBinaryPredicate


class LinkPrediction(AbstractTask):
    def __init__(self, kg: KnowledgeGraph, observed_kg: KnowledgeGraph):
        """
        kg: the triples to be evaluate
        observed_kg: the kgs observed to filter the results
        """
        self.kg = kg
        self.observed_kg = observed_kg
        self.device = self.observed_kg.device

    @classmethod
    def create(cls, filelist, kgindex, observed_kg, device):
        kg = KnowledgeGraph.create(filelist, kgindex, tensorize=False, device=device)
        return cls(kg, observed_kg)

    def evaluate_nbp(self, nbp: NeuralBinaryPredicate, init_batch_size=10000, prefix=""):
        # return self._evaluate_nbp(nbp, init_batch_size, prefix)
        if init_batch_size == 0:
            raise RuntimeError("zero batch size")
        if self.device == 'cpu':
            init_batch_size = 100

        oom = False
        try:
            return self._evaluate_nbp(nbp, init_batch_size, prefix)
        except RuntimeError as error:
            print(error)
            oom = True
        if oom:
            next_batch_size = init_batch_size // 2
            return self.evaluate_nbp(nbp, next_batch_size, prefix)

    def _evaluate_nbp(self, nbp: NeuralBinaryPredicate, batch_size, prefix):
        # nbp.eval()
        record = defaultdict(list)

        cand_id_ten = torch.arange(
            0,
            end=self.kg.num_entities,
            step=1,
            device=nbp.device)
        # raise RuntimeError
        cand_id_ten = torch.reshape(cand_id_ten, (1, -1))

        def cfn(batch):
            hl, rl, tl = [], [], []
            ot_coo_index, oh_coo_index = [[], []], [[], []]

            for i, (h, r, t) in enumerate(batch):
                hl.append(h)
                rl.append(r)
                tl.append(t)

                ot_list = self.observed_kg.hr2t[(h, r)]
                ot_coo_index[0] += [i] * len(ot_list)
                ot_coo_index[1] += ot_list
                assert len(oh_coo_index[0]) == len(oh_coo_index[1])

                oh_list = self.observed_kg.tr2h[(t, r)]
                oh_coo_index[0] += [i] * len(oh_list)
                oh_coo_index[1] += oh_list
                assert len(oh_coo_index[0]) == len(oh_coo_index[1])

            return [torch.tensor(l, device=nbp.device).view((-1, 1))
                    for l in [hl, rl, tl]] + [ot_coo_index, oh_coo_index]

        def extend_link_prediction_record(rank, key):
            record[key + 'hit1'].extend((rank < 1).tolist())
            record[key + 'hit3'].extend((rank < 3).tolist())
            record[key + 'hit10'].extend((rank < 10).tolist())
            record[key + 'mrr'].extend((1 / (1 + rank)).tolist())
            record[key + 'mr'].extend(rank.tolist())

        with tqdm(self.kg.get_triple_dataloader(batch_size=batch_size,
                                                collate_fn=cfn),
                  desc=f"{prefix} Link Prediction Evaluation") as t:
            for head_id_ten, rel_id_ten, tail_id_ten, ot_idx, oh_idx in t:

                # predict head
                head_cand_score_tensor = nbp.batch_predicate_score(
                    [cand_id_ten, rel_id_ten, tail_id_ten])  # [num_cases, num_candidates]

                # > raw evaluation
                head_score = torch.take_along_dim(input=head_cand_score_tensor,
                                                  indices=head_id_ten,
                                                  dim=1)
                head_rank = torch.sum(head_cand_score_tensor >
                                      head_score, -1).cpu().numpy()

                extend_link_prediction_record(head_rank, "raw/agg/")
                extend_link_prediction_record(head_rank, "raw/head/")

                # > filter evaluation
                head_cand_score_tensor[oh_idx[0], oh_idx[1]] = - torch.inf
                head_score = torch.take_along_dim(input=head_cand_score_tensor,
                                                  indices=head_id_ten,
                                                  dim=1)
                head_rank = torch.sum(head_cand_score_tensor >
                                      head_score, -1).cpu().numpy()

                extend_link_prediction_record(head_rank, "filter/agg/")
                extend_link_prediction_record(head_rank, "filter/head/")

                tail_cand_score_tensor = nbp.batch_predicate_score(
                    [head_id_ten, rel_id_ten, cand_id_ten])  # [num_cases, num_candidates]

                tail_score = torch.take_along_dim(input=tail_cand_score_tensor,
                                                  indices=tail_id_ten,
                                                  dim=1)

                tail_rank = torch.sum(tail_cand_score_tensor >
                                      tail_score, -1).cpu().numpy()

                extend_link_prediction_record(tail_rank, "raw/agg/")
                extend_link_prediction_record(tail_rank, "raw/tail/")

                tail_cand_score_tensor[ot_idx[0], ot_idx[1]] = - torch.inf

                tail_score = torch.take_along_dim(input=tail_cand_score_tensor,
                                                  indices=tail_id_ten,
                                                  dim=1)

                tail_rank = torch.sum(tail_cand_score_tensor >
                                      tail_score, -1).cpu().numpy()

                extend_link_prediction_record(tail_rank, "filter/agg/")
                extend_link_prediction_record(tail_rank, "filter/tail/")

                metric = {}
                for k in record:
                    metric[k] = np.mean(record[k])

                t.set_postfix({k: v for k, v in metric.items() if 'agg' in k})

        return metric
