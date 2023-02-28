#! /usr/bin/env python3
#

"""Utility functions for PyTorch."""

import torch
import torch.nn.functional as F
from sklearn import metrics
from jactorch.utils.meta import as_float
from jactorch.utils.meta import as_tensor

__all__ = [
    'binary_accuracy', 'rms', 'monitor_saturation', 'monitor_paramrms',
    'monitor_gradrms'
]


def debug_log(pred, label):
    pos_pred = pred[label > 0.5].detach().cpu()
    neg_pred = pred[label <= 0.5].detach().cpu()
    try:
        print(pos_pred.size(0), pos_pred.mean(), pos_pred.std(), pos_pred.min(), pos_pred.max())
        print(neg_pred.size(0), neg_pred.mean(), neg_pred.std(), neg_pred.min(), neg_pred.max())
    except Exception as e:
        print(e)

def binary_accuracy(label, raw_pred, eps=1e-20, return_float=True, task='binary'):
    """get accuracy for binary classification problem."""
    # print(label.shape)
    # print(raw_pred.shape)
    label = label.flatten()
    raw_pred = raw_pred.flatten()
    pred = as_tensor(raw_pred).squeeze(-1)
    task_metric = {}
    # debug_log(pred, label)
    label = as_tensor(label > 0.5).float()

    precision, recall, _ = metrics.precision_recall_curve(label.detach().cpu(), pred.detach().cpu())
    auc_pr = metrics.auc(recall, precision)

    pred = (pred > 0.5).float()
    # The $acc is micro accuracy = the correct ones / total
    acc = label.eq(pred).float()

    # The $balanced_accuracy is macro accuracy, with class-wide balance.
    nr_total = torch.ones(
        label.size(), dtype=label.dtype, device=label.device).sum(dim=-1)
    nr_pos = label.sum(dim=-1)
    nr_neg = nr_total - nr_pos
    true_pos = (acc * label).sum(dim=-1)
    false_neg = nr_pos - true_pos
    true_neg = acc.sum(dim=-1) - true_pos
    false_pos = nr_neg - true_neg
    balanced_acc = ((true_pos + eps) / (nr_pos + eps) + (true_neg + eps) /
                    (nr_neg + eps)) / 2.0
    precision = true_pos / (true_pos + false_pos + eps)
    recall = true_pos / (true_pos + false_neg + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    # $sat means the saturation rate of the predication,
    # measure how close the predections are to 0 or 1.
    sat = 1 - (raw_pred - pred).abs()
    if return_float:
        acc = as_float(acc.mean())
        balanced_acc = as_float(balanced_acc.mean())
        f1 = as_float(f1.mean())
        precision = as_float(precision.mean())
        recall = as_float(recall.mean())
        sat_mean = as_float(sat.mean())
        sat_min = as_float(sat.min())
    else:
        sat_mean = sat.mean(dim=-1)
        sat_min = sat.min(dim=-1)[0]

    return {
        'accuracy': acc,
        'balanced_accuracy': balanced_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_pr': auc_pr,
        'saturation/mean': sat_mean,
        'saturation/min': sat_min,
        **task_metric
    }


def rms(p):
    """Root mean square function."""
    return as_float((as_tensor(p)**2).mean()**0.5)


def monitor_saturation(model):
    """Monitor the saturation rate."""
    monitors = {}
    for name, p in model.named_parameters():
        p = F.sigmoid(p)
        sat = 1 - (p - (p > 0.5).float()).abs()
        monitors['sat/' + name] = sat
    return monitors


def monitor_paramrms(model):
    """Monitor the rms of the parameters."""
    monitors = {}
    for name, p in model.named_parameters():
        monitors['paramrms/' + name] = rms(p)
    return monitors


def monitor_gradrms(model):
    """Monitor the rms of the gradients of the parameters."""
    monitors = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            monitors['gradrms/' + name] = rms(p.grad) / max(rms(p), 1e-8)
    return monitors
