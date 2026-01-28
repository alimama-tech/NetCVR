
import numpy as np
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
import torch
def ndcg_at_k(y_true, y_pred, k):
    """计算单个样本的 NDCG@K"""
    if y_true in y_pred[:k]:
        rank = np.where(y_pred[:k] == y_true)[0][0] + 1
        return 1 / np.log2(rank + 1)
    return 0

def recall_at_k(y_true, y_pred, k):
    """计算单个样本的 Recall@K"""
    return 1 if y_true in y_pred[:k] else 0

def precision_at_k(y_true, y_pred, k):
    """计算单个样本的 Precision@K"""
    return 1 / k if y_true in y_pred[:k] else 0

def evaluate_basic_recommendations_metrics(pos_items, ratings, topk=[10, 20, 50]):
    """计算 NDCG@K, Recall@K, Precision@K"""
    B, num_items = ratings.shape
    ratings = ratings.cpu().numpy()
    max_k = max(topk)
    topk_indices = np.argsort(-ratings, axis=1)[:, :max_k]


    results = {k: {"NDCG": 0, "Recall": 0, "Precision": 0} for k in topk}
    auc_total = 0
    for i in range(B):
        y_true = pos_items[i].item()
        y_pred = topk_indices[i]

        for k in topk:
            results[k]["NDCG"] += ndcg_at_k(y_true, y_pred, k)
            results[k]["Recall"] += recall_at_k(y_true, y_pred, k)
            results[k]["Precision"] += precision_at_k(y_true, y_pred, k)

        labels = np.zeros(num_items)
        labels[y_true] = 1
        scores = ratings[i]
        try:
            auc = roc_auc_score(labels, scores)
        except ValueError:
            auc = 0.0
            print("ValueError")
        auc_total += auc
    for k in topk:
        results[k]["NDCG"] /= B
        results[k]["Recall"] /= B
        results[k]["Precision"] /= B
    results["AUC"] = auc_total / B
    return results


def auc_score(y_true,y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.clone().detach().cpu().numpy()

    if np.isnan(y_pred).any():
        y_pred = np.nan_to_num(y_pred, nan=0)

    if len(np.unique(y_true)) == 2:
        auc = roc_auc_score(y_true, y_pred)
    else:
        auc = float('nan')
        print("Warning: Only one class present in y_true, AUC cannot be calculated.")
    return auc

def nll_score(y_true,y_pred):
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    y_pred = torch.nan_to_num(y_pred.view(-1), nan=0)
    return F.binary_cross_entropy(y_pred,y_true).item()

import torch
import torch.nn.functional as F

def nll_score_split(y_true, y_pred):
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    y_pred = torch.clamp(torch.nan_to_num(y_pred, nan=0.5), 1e-7, 1 - 1e-7)

    total_samples = len(y_true)
    if total_samples == 0:
        return float('nan'), float('nan')
    pos_mask = (y_true == 1)
    neg_mask = (y_true == 0)
    nll_global = F.binary_cross_entropy(y_pred, y_true).item()
    num_pos = pos_mask.sum().item()
    num_neg = neg_mask.sum().item()
    pos_ratio = num_pos / total_samples
    neg_ratio = num_neg / total_samples
    if num_pos > 0:
        nll_pos_avg = F.binary_cross_entropy(y_pred[pos_mask], y_true[pos_mask]).item()
        nll_pos_contrib = pos_ratio * nll_pos_avg
    else:
        nll_pos_contrib = 0.0

    if num_neg > 0:
        nll_neg_avg = F.binary_cross_entropy(y_pred[neg_mask], y_true[neg_mask]).item()
        nll_neg_contrib = neg_ratio * nll_neg_avg
    else:
        nll_neg_contrib = 0.0


    return nll_pos_contrib, nll_neg_contrib


def prauc_score(y_true,y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.clone().detach().cpu().numpy()
    if len(np.unique(y_true)) == 2:
        pr = average_precision_score(y_true, y_pred)
    else:
        pr = float('nan')
        print("Warning: Only one class present in y_true, PR AUC cannot be calculated.")
    return pr

def pcoc_score(y_true,y_pred,eps = 1e-6):
    pcoc = y_pred.mean().item() / (y_true.mean().item() + eps)
    return pcoc


def stable_log1pex(x):
    return -torch.minimum(x, torch.tensor(0.0, device=x.device)) + torch.log1p(torch.exp(-torch.abs(x)))

