"""Rectified-flow training scheduler for QWRF-Net."""

from __future__ import annotations

import torch

from .time_sampler import TimeSampler2D


def mean_flat(x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Average all non-batch dimensions, optionally using a mask."""
    if mask is None:
        return x.mean(dim=tuple(range(1, x.dim())))

    while mask.dim() < x.dim():
        mask = mask.unsqueeze(1)
    weighted = x * mask
    denom = mask.expand_as(x).sum(dim=tuple(range(1, x.dim()))).clamp_min(1.0)
    return weighted.sum(dim=tuple(range(1, x.dim()))) / denom


def extract_into_tensor(arr: torch.Tensor, timesteps: torch.Tensor, broadcast_shape: torch.Size) -> torch.Tensor:
    """Extract values by time step and reshape for broadcasting."""
    if not torch.is_tensor(arr):
        arr = torch.tensor(arr, device=timesteps.device, dtype=torch.float32)
    arr = arr.to(device=timesteps.device)
    out = arr.gather(0, timesteps.long().clamp(0, arr.shape[0] - 1))
    while out.dim() < len(broadcast_shape):
        out = out.unsqueeze(-1)
    return out.expand(broadcast_shape)


class RFlowScheduler:
    """Compute rectified-flow training losses for precipitation sequences."""

    def __init__(
        self,
        num_timesteps: int = 1000,
        num_sampling_steps: int = 10,
        sample_method: str = "uniform",
        use_discrete_timesteps: bool = False,
        use_timestep_transform: bool = False,
        transform_scale: float = 1.0,
        scale_temporal: bool = True,
        uniform_over_threshold: float | None = None,
        drop_condition: dict | None = None,
        x_cond_weight: float = 1.0,
    ) -> None:
        self.num_timesteps = num_timesteps
        self.num_sampling_steps = num_sampling_steps
        self.time_sampler = TimeSampler2D(
            sample_method=sample_method,
            use_discrete_timesteps=use_discrete_timesteps,
            use_timestep_transform=use_timestep_transform,
            transform_scale=transform_scale,
        )
        self.scale_temporal = scale_temporal
        self.uniform_over_threshold = uniform_over_threshold
        self.drop_condition = drop_condition
        self.x_cond_weight = x_cond_weight

    def training_losses(
        self,
        model,
        x_start: torch.Tensor,
        x_pre: torch.Tensor | None = None,
        model_kwargs: dict | None = None,
        noise: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
        x_gt: torch.Tensor | None = None,
        mask_index=None,
        noise_disable_threshold: float | None = None,
        text_uncond_prob: float | None = None,
        x_noisy_ref: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Compute the rectified-flow velocity matching loss.

        The expected sequence format is [B, T, H, W] or [B, C, T, H, W].
        For QWRF-Net nowcasting, x_start is typically [B, 12, H, W].
        """
        if model_kwargs is None:
            model_kwargs = {}
        if condition is not None:
            model_kwargs = {**model_kwargs, "cond": condition}

        if x_start.dim() == 4:
            sequence = x_start
        elif x_start.dim() == 5:
            sequence = x_start.squeeze(1) if x_start.shape[1] == 1 else x_start
            if sequence.dim() == 5:
                raise ValueError("Expected [B, T, H, W] or [B, 1, T, H, W] for x_start.")
        else:
            raise ValueError(f"Unsupported x_start shape: {tuple(x_start.shape)}")

        batch_size, num_frames = sequence.shape[0], sequence.shape[1]
        if t is None:
            sampled = self.time_sampler.sample(sequence, num_frames, model_kwargs)
            t = sampled.long().clamp(1, num_frames - 1)
        else:
            t = t.long().clamp(1, num_frames - 1)

        if noise_disable_threshold is not None and x_gt is not None:
            no_noise_mask = t > noise_disable_threshold
            sequence[no_noise_mask] = x_gt[no_noise_mask]

        batch_indices = torch.arange(batch_size, device=sequence.device)
        previous_frame = sequence[batch_indices, t - 1, ...]
        target_frame = sequence[batch_indices, t, ...]

        if x_pre is None:
            model_input = previous_frame
        elif x_pre.dim() == 4:
            model_input = x_pre[batch_indices, t, ...]
        else:
            model_input = x_pre

        model_output = model(model_input, t, **model_kwargs)
        velocity_pred = model_output.chunk(2, dim=1)[0] if model_output.shape[1] != target_frame.shape[1] else model_output
        target_velocity = target_frame - previous_frame

        if velocity_pred.shape != target_velocity.shape:
            if target_velocity.dim() == 3:
                target_velocity = target_velocity.unsqueeze(1)
            if velocity_pred.shape != target_velocity.shape:
                raise ValueError(
                    f"Velocity shape mismatch: prediction={tuple(velocity_pred.shape)}, "
                    f"target={tuple(target_velocity.shape)}"
                )

        squared_error = (velocity_pred - target_velocity).pow(2)
        if weights is not None:
            weight = extract_into_tensor(weights, t, target_velocity.shape)
            squared_error = squared_error * weight

        return {"loss": mean_flat(squared_error, mask=mask)}

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Linearly interpolate between samples and noise for compatibility."""
        timepoints = 1 - timesteps.float() / self.num_timesteps
        while timepoints.dim() < noise.dim():
            timepoints = timepoints.unsqueeze(-1)
        return timepoints * original_samples + (1 - timepoints) * noise

