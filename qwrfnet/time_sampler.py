"""Time-step samplers for 2D rectified-flow precipitation nowcasting."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions import LogisticNormal


def extract_hw(model_kwargs: dict | None) -> torch.Tensor:
    """Extract the spatial resolution from model keyword arguments."""
    if model_kwargs is None:
        return torch.ones(1)

    for key in ["height", "width"]:
        if key in model_kwargs:
            value = model_kwargs[key]
            if isinstance(value, torch.Tensor) and value.dtype == torch.float16:
                model_kwargs[key] = value.float()

    if "height" in model_kwargs and "width" in model_kwargs:
        return model_kwargs["height"] * model_kwargs["width"]
    return torch.ones(1)


def timestep_transform_2d(
    t: torch.Tensor,
    model_kwargs: dict | None = None,
    base_resolution: int = 512 * 512,
    scale: float = 1.0,
    num_timesteps: int = 1,
    ret_ratio: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Apply a resolution-aware time-step transform for 2D fields."""
    t = t / num_timesteps

    if model_kwargs is not None:
        resolution = extract_hw(model_kwargs).to(t.device)
        ratio = (resolution / base_resolution).sqrt() * scale
    else:
        ratio = torch.tensor(scale, device=t.device, dtype=t.dtype)

    t = ratio * t / (1 + (ratio - 1) * t)
    t = t * num_timesteps

    if ret_ratio:
        return t, ratio
    return t


class TimeSampler2D:
    """Sample training time steps for 2D rectified-flow learning."""

    def __init__(
        self,
        sample_method: str = "uniform",
        use_discrete_timesteps: bool = False,
        use_timestep_transform: bool = False,
        transform_scale: float = 1.0,
        loc: float = 0.0,
        scale: float = 1.0,
    ) -> None:
        if sample_method not in ["uniform", "logit-normal"]:
            raise ValueError(f"Unknown sample_method: {sample_method}")
        if sample_method != "uniform" and use_discrete_timesteps:
            raise ValueError("Only uniform sampling supports discrete timesteps.")

        self.sample_method = sample_method
        self.use_discrete_timesteps = use_discrete_timesteps
        self.use_timestep_transform = use_timestep_transform
        self.transform_scale = transform_scale

        if sample_method == "logit-normal":
            self.distribution = LogisticNormal(torch.tensor([loc]), torch.tensor([scale]))

    def sample(
        self,
        x_start: torch.Tensor,
        num_timesteps: int,
        model_kwargs: dict | None = None,
    ) -> torch.Tensor:
        """Sample time steps for a batch."""
        if self.use_discrete_timesteps:
            t = torch.randint(1, num_timesteps, (x_start.shape[0],), device=x_start.device)
        elif self.sample_method == "uniform":
            t = torch.rand((x_start.shape[0],), device=x_start.device) * num_timesteps
        else:
            t = self.distribution.sample((x_start.shape[0],))[:, 0].to(x_start.device)
            t = t * num_timesteps

        if not self.use_timestep_transform:
            return t

        return timestep_transform_2d(
            t,
            model_kwargs,
            scale=self.transform_scale,
            num_timesteps=num_timesteps,
        )

    def visualize(self, height: int = 512, width: int = 512, num_timesteps: int = 16):
        """Visualize the original and transformed time-step distributions."""
        batch_size = 1000
        x_start = torch.randn(batch_size)

        self.use_timestep_transform = False
        original_t_values = self.sample(x_start, num_timesteps)

        self.use_timestep_transform = True
        model_kwargs = {
            "height": torch.full((batch_size,), height),
            "width": torch.full((batch_size,), width),
        }
        transformed_t_values, ratio = timestep_transform_2d(
            original_t_values,
            model_kwargs,
            scale=self.transform_scale,
            num_timesteps=num_timesteps,
            ret_ratio=True,
        )

        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        axes[0].scatter(original_t_values.numpy(), transformed_t_values.numpy(), alpha=0.6, s=10)
        axes[0].plot([0, num_timesteps], [0, num_timesteps], "r--", alpha=0.5, label="y=x")
        axes[0].set_xlabel("Original t")
        axes[0].set_ylabel("Transformed t")
        axes[0].set_title(f"Time transform (ratio={ratio.flatten()[0].item():.3f})")
        axes[0].legend()
        axes[0].grid(True)

        bins = np.linspace(0, num_timesteps, 50)
        axes[1].hist(original_t_values.numpy(), bins=bins, alpha=0.6, label="Original", density=True)
        axes[1].hist(transformed_t_values.numpy(), bins=bins, alpha=0.6, label="Transformed", density=True)
        axes[1].set_xlabel("Time step")
        axes[1].set_ylabel("Density")
        axes[1].set_title("Time distribution")
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig("timestep_sampling_2d.png", dpi=150)
        return original_t_values, transformed_t_values


class SimpleTimeSampler:
    """Minimal time sampler for sequence data."""

    def __init__(self, sample_method: str = "uniform") -> None:
        self.sample_method = sample_method

    def sample(self, batch_size: int, num_timesteps: int, device: torch.device) -> torch.Tensor:
        """Sample discrete or continuous time steps."""
        if self.sample_method == "uniform":
            return torch.randint(0, num_timesteps, (batch_size,), device=device)
        if self.sample_method == "continuous":
            return torch.rand(batch_size, device=device) * num_timesteps
        raise ValueError(f"Unknown sample_method: {self.sample_method}")

