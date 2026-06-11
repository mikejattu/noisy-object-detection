"""Metrics, all derived from one (logits, target) pass — no recompute.

top1/top5/nll are accumulated as running sums (divide by n at the end).
ECE is a *global* statistic (not averageable per batch), so it is accumulated as
per-bin (conf_sum, acc_sum, count) and reduced once at the end.
"""
from __future__ import annotations

import torch


@torch.inference_mode()
def topk_correct(logits: torch.Tensor, target: torch.Tensor, ks=(1, 5)) -> dict[int, float]:
    maxk = max(ks)
    _, pred = logits.topk(maxk, dim=1)             # (B, maxk)
    correct = pred.eq(target.view(-1, 1))          # (B, maxk)
    return {k: correct[:, :k].any(dim=1).float().sum().item() for k in ks}


@torch.inference_mode()
def nll_sum(logits: torch.Tensor, target: torch.Tensor) -> float:
    return torch.nn.functional.cross_entropy(logits, target, reduction="sum").item()


@torch.inference_mode()
def ece_bin_stats(logits: torch.Tensor, target: torch.Tensor, n_bins: int = 15):
    """Per-bin accumulators for expected calibration error. Sum across batches."""
    probs = logits.softmax(dim=1)
    conf, pred = probs.max(dim=1)
    correct = pred.eq(target).float()
    edges = torch.linspace(0.0, 1.0, n_bins + 1, device=logits.device)
    idx = torch.bucketize(conf, edges[1:-1])       # bin index 0..n_bins-1
    zeros = torch.zeros(n_bins, device=logits.device)
    conf_sum = zeros.clone().scatter_add_(0, idx, conf)
    acc_sum = zeros.clone().scatter_add_(0, idx, correct)
    count = zeros.clone().scatter_add_(0, idx, torch.ones_like(conf))
    return conf_sum, acc_sum, count


def ece_from_bins(conf_sum: torch.Tensor, acc_sum: torch.Tensor, count: torch.Tensor) -> float:
    n = count.sum().clamp(min=1)
    safe = count.clamp(min=1)
    nonempty = count > 0
    avg_conf = torch.where(nonempty, conf_sum / safe, torch.zeros_like(count))
    avg_acc = torch.where(nonempty, acc_sum / safe, torch.zeros_like(count))
    return ((count / n) * (avg_conf - avg_acc).abs()).sum().item()
