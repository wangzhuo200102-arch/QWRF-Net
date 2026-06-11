"""Evaluate QWRF-Net on a directory of prepared precipitation nowcasting samples."""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch
import yaml
from tqdm import tqdm

from qwrfnet import QWRFNet
from qwrfnet.metrics import contingency_counts, skill_scores, ssim_simple
from qwrfnet.visualization import save_compare_grid_12


def load_config(path: str) -> dict:
    if not os.path.exists(path):
        return {
            "model": {"in_channels": 12, "cond_channels": 6, "base_embed": 8, "input_hw": 64, "num_timesteps": 1000},
            "data": {"max_vil": 255.0},
            "evaluation": {"thresholds": [16, 74, 133, 160, 181, 219]},
        }
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_model(config: dict, device: torch.device) -> QWRFNet:
    model_cfg = config.get("model", {})
    return QWRFNet(
        in_channels=model_cfg.get("in_channels", 12),
        cond_channels=model_cfg.get("cond_channels", 6),
        base_embed=model_cfg.get("base_embed", 24),
        input_hw=model_cfg.get("input_hw", 288),
        num_timesteps=model_cfg.get("num_timesteps", 1000),
        use_ckpt=False,
        use_quantum_wavelet=model_cfg.get("use_quantum_wavelet", False),
    ).to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sevir_qwrfnet.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data-dir", required=True, help="Directory containing prepared .npz samples.")
    parser.add_argument("--output", default="outputs/evaluation")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    config = load_config(args.config)
    device = torch.device(args.device)
    model = build_model(config, device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state, strict=False)

    sample_files = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    if not sample_files:
        raise RuntimeError(f"No .npz samples found in {args.data_dir}.")

    max_vil = float(config.get("data", {}).get("max_vil", 255.0))
    thresholds = config.get("evaluation", {}).get("thresholds", [16, 74, 133, 160, 181, 219])
    metric_sums = {"mse": 0.0, "mae": 0.0, "ssim": 0.0}
    counts = {str(th): {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for th in thresholds}

    model.eval()
    with torch.no_grad():
        for idx, path in enumerate(tqdm(sample_files, desc="Evaluating")):
            sample = np.load(path)
            cond = torch.from_numpy(sample["condition"]).float().unsqueeze(0).to(device)
            target = torch.from_numpy(sample["target"]).float().unsqueeze(0).to(device)
            x = torch.zeros_like(target)
            t = torch.full((1,), config.get("model", {}).get("num_timesteps", 1000) - 1, device=device)
            pred = model(x, t, cond=cond).sigmoid().clamp(0, 1)

            pred_phy = pred * max_vil
            target_phy = target * max_vil
            metric_sums["mse"] += torch.mean((pred_phy - target_phy) ** 2).item()
            metric_sums["mae"] += torch.mean((pred_phy - target_phy).abs()).item()
            metric_sums["ssim"] += ssim_simple(pred_phy.flatten(0, 1), target_phy.flatten(0, 1)).item()

            for th in thresholds:
                tp, fp, fn, tn = contingency_counts(pred_phy, target_phy, th)
                key = str(th)
                counts[key]["tp"] += tp
                counts[key]["fp"] += fp
                counts[key]["fn"] += fn
                counts[key]["tn"] += tn

            if idx == 0:
                save_compare_grid_12(target[0], pred[0], os.path.join(args.output, "sample_preview.png"))

    num_samples = len(sample_files)
    metrics = {key: value / num_samples for key, value in metric_sums.items()}
    metrics["rmse"] = metrics["mse"] ** 0.5
    metrics["skill_scores"] = {
        th: skill_scores(c["tp"], c["fp"], c["fn"], c["tn"]) for th, c in counts.items()
    }

    with open(os.path.join(args.output, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
