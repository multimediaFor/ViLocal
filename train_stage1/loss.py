import warnings

warnings.filterwarnings("ignore")
import torch.nn.functional as F
import torch
import torch.nn as nn


class info_nce_loss(nn.Module):
    """
        pred : [B, C, H, W]
        target : [B, 1, H, W]
    """

    def __init__(self):
        super(info_nce_loss, self).__init__()

    def forward(self, pred, target):
        pred = pred.permute(0, 3, 2, 1)  # [B, W, H, C]
        target = target.permute(0, 1, 3, 2)  # [B, C, W, H]
        B, W, H, C = pred.shape

        loss_list = []
        for i in range(B):
            pred_i = pred[i]
            target_i = target[i][0]

            query = pred_i[target_i == 0]
            negative = pred_i[target_i == 1]

            dict_size = 3000  # could be larger according to gpu memory

            # 使用随机排列的索引从 query 中采样 dict_size 个查询样本
            query_sample = query[torch.randperm(query.size()[0])[:dict_size]]

            # 使用随机排列的索引从 negative 中采样 dict_size 个负样本
            negative_sample = negative[torch.randperm(negative.size(0))[:dict_size]]

            loss = info_nce(query_sample, query_sample, negative_sample)
            loss_list.append(loss)

        batch_loss = torch.mean(torch.stack(loss_list).squeeze())
        return batch_loss


def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired',
             metric='cosine'):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError(
                "If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        # positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        sim_matrix = my_distance(query, query, metric)
        mask = (torch.ones_like(sim_matrix) - torch.eye(sim_matrix.shape[0], device=sim_matrix.device)).bool()
        sim_matrix = sim_matrix.masked_select(mask).view(sim_matrix.shape[0], -1)

        positive_logit = torch.mean(sim_matrix, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = my_distance(query, negative_keys, metric)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def my_distance(x, y, metric='cosine'):
    if metric == 'cosine':
        return torch.mm(x, y.t())
    else:
        return torch.cdist(x, y)


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]
