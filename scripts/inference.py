"""Run QWRF-Net inference on a single input sample."""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import yaml

from qwrfnet import QWRFNet
from qwrfnet.visualization import save_compare_grid_12


def load_config(path: str) -> dict:
    if not os.path.exists(path):
        return {
            "model": {"in_channels": 12, "cond_channels": 6, "base_embed": 8, "input_hw": 64, "num_timesteps": 1000},
            "data": {"max_vil": 255.0},
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
    parser.add_argument("--input", required=True, help="Input .npy file with shape [6, H, W].")
    parser.add_argument("--output", default="outputs/inference")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    config = load_config(args.config)
    device = torch.device(args.device)
    model = build_model(config, device)

    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state, strict=False)

    condition = np.load(args.input).astype(np.float32)
    if condition.shape[0] != 6:
        raise ValueError(f"Expected input shape [6, H, W], got {condition.shape}.")

    cond = torch.from_numpy(condition).unsqueeze(0).to(device)
    x = torch.zeros((1, 12, condition.shape[-2], condition.shape[-1]), device=device)
    t = torch.full((1,), config.get("model", {}).get("num_timesteps", 1000) - 1, device=device)

    model.eval()
    with torch.no_grad():
        prediction = model(x, t, cond=cond).sigmoid().clamp(0, 1)[0].cpu()

    np.save(os.path.join(args.output, "prediction.npy"), prediction.numpy())
    save_compare_grid_12(prediction, prediction, os.path.join(args.output, "prediction_preview.png"), show_error=False)
    print(f"Saved inference outputs to {args.output}")


if __name__ == "__main__":
    main()
