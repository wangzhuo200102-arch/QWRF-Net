"""Sampling utilities for rectified-flow precipitation generation."""

from __future__ import annotations

import torch
from tqdm import tqdm

from .rectified_flow import RFlowScheduler
from .time_sampler import timestep_transform_2d


def dynamic_thresholding(x: torch.Tensor, ratio: float = 0.995, base: float = 6.0) -> torch.Tensor:
    """Apply dynamic thresholding to stabilize generated fields."""
    s = torch.quantile(x.abs().flatten(), ratio)
    s = max(float(s), base)
    return x.clip(-s, s) * base / s


class RFLOW2D:
    """Rectified-flow sampler for 2D precipitation nowcasting."""

    def __init__(
        self,
        num_sampling_steps: int = 16,
        num_timesteps: int = 16,
        cfg_scale: float = 4.0,
        use_discrete_timesteps: bool = True,
        use_timestep_transform: bool = False,
        transform_scale: float = 1.0,
        **kwargs,
    ) -> None:
        self.num_sampling_steps = num_sampling_steps
        self.num_timesteps = num_timesteps
        self.cfg_scale = cfg_scale
        self.use_discrete_timesteps = use_discrete_timesteps
        self.use_timestep_transform = use_timestep_transform
        self.transform_scale = transform_scale

        self.scheduler = RFlowScheduler(
            num_timesteps=num_timesteps,
            num_sampling_steps=num_sampling_steps,
            use_discrete_timesteps=use_discrete_timesteps,
            use_timestep_transform=use_timestep_transform,
            transform_scale=transform_scale,
            **kwargs,
        )

    def _make_timesteps(self, batch_size: int, device: torch.device, additional_args: dict | None = None):
        timesteps = [
            (1.0 - i / self.num_sampling_steps) * self.num_timesteps
            for i in range(self.num_sampling_steps)
        ][1:]
        if self.use_discrete_timesteps:
            timesteps = [int(round(t)) for t in timesteps]

        tensors = [torch.tensor([t] * batch_size, device=device) for t in timesteps]
        if self.use_timestep_transform and additional_args is not None:
            tensors = [
                timestep_transform_2d(
                    t,
                    additional_args,
                    scale=self.transform_scale,
                    num_timesteps=self.num_timesteps,
                )
                for t in tensors
            ]
        return tensors

    def sample_simple(
        self,
        model,
        z: torch.Tensor,
        device: torch.device,
        additional_args: dict | None = None,
        condition: torch.Tensor | None = None,
        progress: bool = True,
    ) -> torch.Tensor:
        """Generate a full future sequence from an initial state.

        If z has shape [B, T, H, W], the first frame is used as the initial
        state and the returned tensor has shape [B, T, H, W].
        """
        if z.dim() == 4:
            batch_size, target_steps = z.shape[0], z.shape[1]
            current = z[:, 0, ...]
            context = z.clone()
        elif z.dim() == 5:
            batch_size, target_steps = z.shape[0], z.shape[2]
            current = z[:, :, 0, ...]
            context = z.clone()
        else:
            raise ValueError(f"Unsupported z shape: {tuple(z.shape)}")

        timesteps = self._make_timesteps(batch_size, device, additional_args)
        progress_wrap = tqdm if progress else (lambda x: x)
        trajectory = [current.clone()]

        for i, t in progress_wrap(list(enumerate(reversed(timesteps)))):
            kwargs = dict(additional_args or {})
            if condition is not None:
                kwargs["cond"] = condition
            pred = model(current, t, **kwargs)
            v_pred = pred[0] if isinstance(pred, tuple) else pred
            if v_pred.shape[1] != current.shape[1] and v_pred.shape[1] % 2 == 0:
                v_pred = v_pred.chunk(2, dim=1)[0]

            dt = timesteps[i] - timesteps[i + 1] if i < len(timesteps) - 1 else timesteps[i]
            dt = dt / self.num_timesteps
            while dt.dim() < current.dim():
                dt = dt.unsqueeze(-1)

            if z.dim() == 4 and i < target_steps:
                residual = context[:, i, ...]
            elif z.dim() == 5 and i < target_steps:
                residual = context[:, :, i, ...]
            else:
                residual = 0

            current = current + v_pred * dt + residual
            trajectory.append(current.clone())

        result = torch.stack(trajectory[:target_steps], dim=1)
        return result

    def sampleno(self, *args, **kwargs) -> torch.Tensor:
        """Backward-compatible alias used by earlier experiment scripts."""
        return self.sample_simple(*args, **kwargs)

    def training_losses(self, *args, **kwargs):
        """Forward training-loss calls to the scheduler."""
        return self.scheduler.training_losses(*args, **kwargs)

