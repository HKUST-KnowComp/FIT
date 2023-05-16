import torch


def lcwa_negative_sampling(phead_id_ten,
                           ptail_id_ten,
                           num_entities,
                           num_neg_samples):
    device = phead_id_ten.device

    original_shape = phead_id_ten.shape

    neg_sample_shape = original_shape + (num_neg_samples,)

    phead_id_ten = phead_id_ten.unsqueeze(-1).expand(neg_sample_shape)

    ptail_id_ten = ptail_id_ten.unsqueeze(-1).expand(neg_sample_shape)

    random_entities = torch.randint(low=0,
                                    high=num_entities,
                                    size=neg_sample_shape,
                                    device=device)

    head_collapse = torch.randint(low=0,
                                  high=2,
                                  size=neg_sample_shape,
                                  device=device).bool()

    tail_collapse = head_collapse.logical_not()

    nhead = torch.where(head_collapse, random_entities, phead_id_ten)
    ntail = torch.where(tail_collapse, random_entities, ptail_id_ten)

    return nhead, ntail

# TODO: implement the negative sampling
def rel_negative_sampling(prel_id_ten, num_relations, num_neg_samples):
    return
