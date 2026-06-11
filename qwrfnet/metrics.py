"""Evaluation metrics for precipitation nowcasting."""

from __future__ import annotations

import torch


def ssim_simple(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6, data_range: float = 255.0) -> torch.Tensor:
    """Compute a lightweight image-level SSIM approximation."""
    if x.dim() == 2:
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
    mu_x = x.mean(dim=(-2, -1))
    mu_y = y.mean(dim=(-2, -1))
    var_x = x.var(dim=(-2, -1), unbiased=False)
    var_y = y.var(dim=(-2, -1), unbiased=False)
    cov_xy = ((x - mu_x[..., None, None]) * (y - mu_y[..., None, None])).mean(dim=(-2, -1))
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    score = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2))
    score = score / ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2) + eps)
    return score.mean()


def contingency_counts(pred: torch.Tensor, target: torch.Tensor, threshold: float):
    """Return TP, FP, FN, and TN counts for a precipitation threshold."""
    pred_event = pred >= threshold
    target_event = target >= threshold
    tp = (pred_event & target_event).sum().item()
    fp = (pred_event & ~target_event).sum().item()
    fn = (~pred_event & target_event).sum().item()
    tn = (~pred_event & ~target_event).sum().item()
    return tp, fp, fn, tn


def skill_scores(tp: int, fp: int, fn: int, tn: int, eps: float = 1e-8) -> dict[str, float]:
    """Compute CSI, HSS, POD, and FAR from contingency counts."""
    csi = tp / (tp + fp + fn + eps)
    pod = tp / (tp + fn + eps)
    far = fp / (tp + fp + eps)
    hss_num = 2 * (tp * tn - fn * fp)
    hss_den = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn) + eps
    hss = hss_num / hss_den
    return {"csi": csi, "hss": hss, "pod": pod, "far": far}


def is_extreme_event(
    targets_phy: torch.Tensor,
    peak_th: float = 219.0,
    area_th: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Identify high-intensity precipitation samples."""
    peak = targets_phy.amax(dim=tuple(range(1, targets_phy.dim())))
    area_ratio = (targets_phy > peak_th).float().mean(dim=tuple(range(1, targets_phy.dim())))
    extreme = (peak >= peak_th) & (area_ratio >= area_th)
    return extreme, peak, area_ratio

