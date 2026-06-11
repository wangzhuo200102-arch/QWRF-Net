"""Visualization utilities for QWRF-Net predictions."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def save_compare_grid_12(
    gt: torch.Tensor,
    pred: torch.Tensor,
    file_path: str,
    max_vil: float = 255.0,
    title: str | None = None,
    t_stride: int = 1,
    show_error: bool = True,
) -> None:
    """Save a GT-vs-prediction comparison grid for 12 nowcast frames."""
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    gt = gt.detach().float().clamp(0, 1).cpu() * max_vil
    pred = pred.detach().float().clamp(0, 1).cpu() * max_vil
    err = (pred - gt).abs()

    gt_np = gt.numpy()
    pred_np = pred.numpy()
    err_np = err.numpy()
    num_frames = gt_np.shape[0]
    frame_ids = list(range(0, num_frames, t_stride))
    ncol = len(frame_ids)
    nrow = 3 if show_error else 2

    fig, axes = plt.subplots(nrow, ncol, figsize=(2.1 * ncol, 2.8 * nrow), dpi=140)
    axes = np.array(axes).reshape(nrow, ncol)
    if title:
        fig.suptitle(title, fontsize=12)

    image = None
    for col, frame_id in enumerate(frame_ids):
        image = axes[0, col].imshow(gt_np[frame_id], cmap="gist_ncar", vmin=0, vmax=max_vil)
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])
        axes[0, col].set_title(f"T+{5 * (frame_id + 1)} min", fontsize=9)
        if col == 0:
            axes[0, col].set_ylabel("GT", fontsize=10)

        axes[1, col].imshow(pred_np[frame_id], cmap="gist_ncar", vmin=0, vmax=max_vil)
        axes[1, col].set_xticks([])
        axes[1, col].set_yticks([])
        if col == 0:
            axes[1, col].set_ylabel("Pred", fontsize=10)

        if show_error:
            axes[2, col].imshow(err_np[frame_id], cmap="magma")
            axes[2, col].set_xticks([])
            axes[2, col].set_yticks([])
            if col == 0:
                axes[2, col].set_ylabel("|Err|", fontsize=10)

    cax = fig.add_axes([0.92, 0.36 if show_error else 0.20, 0.015, 0.50])
    fig.colorbar(image, cax=cax, ticks=[0, 50, 100, 150, 200, 255], label="VIL")
    fig.subplots_adjust(left=0.02, right=0.90, bottom=0.08, top=0.90, wspace=0.05, hspace=0.15)
    fig.savefig(file_path, bbox_inches="tight")
    plt.close(fig)

