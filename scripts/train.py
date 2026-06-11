"""Minimal QWRF-Net training entry point."""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from qwrfnet import QWRFNet


class NPZNowcastingDataset(Dataset):
    """Dataset for prepared nowcasting samples stored as NPZ files."""

    def __init__(self, data_dir: str) -> None:
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if not self.files:
            raise RuntimeError(f"No .npz samples found in {data_dir}.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        sample = np.load(self.files[idx])
        condition = torch.from_numpy(sample["condition"]).float()
        target = torch.from_numpy(sample["target"]).float()
        return condition, target


def load_config(path: str) -> dict:
    if not os.path.exists(path):
        return {
            "model": {
                "in_channels": 12,
                "cond_channels": 6,
                "base_embed": 8,
                "input_hw": 64,
                "num_timesteps": 1000,
                "use_checkpointing": False,
                "use_quantum_wavelet": False,
            },
            "training": {
                "batch_size_per_gpu": 2,
                "learning_rate": 1e-4,
                "weight_decay": 1e-4,
                "epochs": 1,
                "gradient_clip_norm": 1.0,
            },
        }
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sevir_qwrfnet.yaml")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default="outputs/train")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    config = load_config(args.config)
    device = torch.device(args.device)
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})

    dataset = NPZNowcastingDataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=int(train_cfg.get("batch_size_per_gpu", 2)), shuffle=True)

    model = QWRFNet(
        in_channels=model_cfg.get("in_channels", 12),
        cond_channels=model_cfg.get("cond_channels", 6),
        base_embed=model_cfg.get("base_embed", 24),
        input_hw=model_cfg.get("input_hw", 288),
        num_timesteps=model_cfg.get("num_timesteps", 1000),
        use_ckpt=bool(model_cfg.get("use_checkpointing", False)),
        use_quantum_wavelet=model_cfg.get("use_quantum_wavelet", False),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    epochs = args.epochs or int(train_cfg.get("epochs", 1))
    num_timesteps = int(model_cfg.get("num_timesteps", 1000))

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        for condition, target in tqdm(loader, desc=f"Epoch {epoch}/{epochs}"):
            condition = condition.to(device)
            target = target.to(device)
            x = torch.zeros_like(target)
            t = torch.randint(1, num_timesteps, (target.shape[0],), device=device)
            pred = model(x, t, cond=condition)
            loss = F.mse_loss(pred.sigmoid(), target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("gradient_clip_norm", 1.0)))
            optimizer.step()
            loss_sum += loss.item() * target.shape[0]

        avg_loss = loss_sum / len(dataset)
        print(f"Epoch {epoch}: loss={avg_loss:.6f}")
        torch.save(model.state_dict(), os.path.join(args.output, "best_model.pth"))

    print(f"Saved checkpoint to {os.path.join(args.output, 'best_model.pth')}")


if __name__ == "__main__":
    main()
