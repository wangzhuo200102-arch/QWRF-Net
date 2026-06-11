"""QWRF-Net model architecture for short-term precipitation nowcasting."""

from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.checkpoint import checkpoint as checkpoint_fn

try:
    import pennylane as qml

    QML_AVAILABLE = True
except ImportError:
    qml = None
    QML_AVAILABLE = False

try:
    import pytorch_wavelets as pwt

    PWT_AVAILABLE = True
except ImportError:
    pwt = None
    PWT_AVAILABLE = False


class ChannelLayerNorm2d(nn.Module):
    """Layer normalization over the channel dimension for 2D feature maps."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias


def maybe_checkpoint(module: nn.Module, x: torch.Tensor, use_checkpoint: bool) -> torch.Tensor:
    """Apply gradient checkpointing when requested."""
    if use_checkpoint and x.requires_grad:
        try:
            return checkpoint_fn(module, x, use_reentrant=False, preserve_rng_state=False)
        except TypeError:
            return checkpoint_fn(module, x, preserve_rng_state=False)
    return module(x)


def add_coords(x: torch.Tensor) -> torch.Tensor:
    """Append normalized x/y coordinate channels."""
    batch, _, height, width = x.shape
    device = x.device
    xx = torch.linspace(-1, 1, width, device=device).view(1, 1, 1, width).expand(batch, 1, height, width)
    yy = torch.linspace(-1, 1, height, device=device).view(1, 1, height, 1).expand(batch, 1, height, width)
    return torch.cat([x, xx, yy], dim=1)


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal time-step embedding followed by an MLP projection."""

    def __init__(self, dim: int, max_period: int = 10000) -> None:
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, t: torch.Tensor, num_timesteps: int) -> torch.Tensor:
        t = t.float().clamp(min=0) / max(float(num_timesteps), 1.0)
        half = self.dim // 2
        freqs = torch.exp(-math.log(self.max_period) * torch.arange(0, half, device=t.device) / half)
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.proj(emb)


class WindowAttention(nn.Module):
    """Local window attention with coordinate channels."""

    def __init__(self, in_channels: int, window_size: int = 7) -> None:
        super().__init__()
        self.window_size = window_size
        self.inner_dim = in_channels + 2
        self.qkv = nn.Conv2d(self.inner_dim, 3 * self.inner_dim, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(self.inner_dim, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_with_coords = add_coords(x)
        batch, channels, height, width = x_with_coords.shape
        window = self.window_size
        pad_h = (window - height % window) % window
        pad_w = (window - width % window) % window
        x_padded = F.pad(x_with_coords, (0, pad_w, 0, pad_h))
        padded_h, padded_w = x_padded.shape[2:]

        qkv = self.qkv(x_padded)
        qkv = qkv.unfold(2, window, window).unfold(3, window, window)
        qkv = qkv.contiguous().view(batch, 3 * channels, -1, window, window)
        qkv = qkv.permute(0, 2, 1, 3, 4).contiguous().view(-1, 3 * channels, window * window)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        attn = (q.transpose(-2, -1) @ k) / (channels**0.5)
        attn = attn.softmax(dim=-1)
        out = (v @ attn.transpose(-2, -1)).view(-1, channels, window, window)

        num_windows_h = padded_h // window
        out = out.view(batch, num_windows_h, padded_w // window, channels, window, window)
        out = out.permute(0, 3, 1, 4, 2, 5).contiguous().view(batch, channels, padded_h, padded_w)
        return self.proj(out[:, :, :height, :width])


class DownSampleBlock(nn.Module):
    """Layer-normalized strided convolution."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 2) -> None:
        super().__init__()
        self.norm = ChannelLayerNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.norm(x))


class MultiScaleAttention(nn.Module):
    """Multi-scale window attention over large, medium, and small branches."""

    def __init__(self, channels: int, window_size: int = 7) -> None:
        super().__init__()
        self.down_medium = DownSampleBlock(channels, channels * 2)
        self.down_small = DownSampleBlock(channels, channels * 4, stride=4)
        self.attn_large = WindowAttention(channels, window_size)
        self.attn_medium = WindowAttention(channels * 2, window_size)
        self.attn_small = WindowAttention(channels * 4, window_size)
        self.conv_medium = nn.Conv2d(channels * 2, channels, 1)
        self.conv_small = nn.Conv2d(channels * 4, channels, 1)
        self.fuse = nn.Conv2d(channels * 3, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        large = self.attn_large(x)
        medium = self.conv_medium(self.attn_medium(self.down_medium(x)))
        medium = F.interpolate(medium, (height, width), mode="bilinear", align_corners=False)
        small = self.conv_small(self.attn_small(self.down_small(x)))
        small = F.interpolate(small, (height, width), mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([large, medium, small], dim=1))


class FeedForward2d(nn.Module):
    """Convolutional feed-forward block."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 1),
            nn.GELU(),
            nn.Conv2d(channels * 4, channels * 4, 3, padding=1, groups=channels * 4),
            nn.Conv2d(channels * 4, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiScaleModule(nn.Module):
    """Residual multi-scale attention and feed-forward block."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm1 = ChannelLayerNorm2d(channels)
        self.attn = MultiScaleAttention(channels)
        self.norm2 = ChannelLayerNorm2d(channels)
        self.ffn = FeedForward2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class AdaptiveDualWeightModule(nn.Module):
    """Adaptive fusion between two feature maps."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv_x = nn.Conv2d(channels, channels, 1)
        self.conv_y = nn.Conv2d(channels, channels, 1)
        self.conv_alpha = nn.Sequential(nn.Conv2d(channels, 1, 1), nn.Sigmoid())
        self.fuse = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        residual = x + y
        alpha = self.conv_alpha(self.conv_x(x) * self.conv_y(y)).expand(-1, x.shape[1], -1, -1)
        out = (1 - alpha) * x + alpha * y
        return residual + self.fuse(out)


class MultiConvModule(nn.Module):
    """Depthwise convolutional mixing block."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = ChannelLayerNorm2d(channels)
        self.dw1 = nn.Conv2d(channels, channels * 6, 3, padding=1, groups=channels)
        self.pw1 = nn.Conv2d(channels * 6, channels * 6, 1)
        self.dw2 = nn.Conv2d(channels * 3, channels * 3, 3, padding=1, groups=channels * 3)
        self.pw2 = nn.Conv2d(channels * 3, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.norm(x)
        a, b = torch.chunk(self.pw1(self.dw1(shortcut)), 2, dim=1)
        return shortcut + self.pw2(self.dw2(F.relu(a) * b))


class Encoder(nn.Module):
    """QWRF-Net encoder block."""

    def __init__(self, channels: int, num_blocks: int, use_checkpoint: bool = True) -> None:
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([MultiScaleModule(channels) for _ in range(num_blocks)])
        self.adjust = nn.ModuleList([nn.Conv2d(channels * 2, channels, 1) for _ in range(num_blocks)])
        self.mcm = MultiConvModule(channels)
        self.adwm = AdaptiveDualWeightModule(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        block_out = x
        for block, adjust in zip(self.blocks, self.adjust):
            block_out = adjust(torch.cat([maybe_checkpoint(block, block_out, self.use_checkpoint), block_out], dim=1))
        return self.adwm(block_out, self.mcm(x))


class DecoderUnit(nn.Module):
    """QWRF-Net decoder refinement block."""

    def __init__(self, channels: int, num_blocks: int, use_checkpoint: bool = True) -> None:
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([MultiScaleModule(channels) for _ in range(num_blocks)])
        self.adjust = nn.ModuleList([nn.Conv2d(channels * 2, channels, 1) for _ in range(num_blocks)])
        self.mcm = MultiConvModule(channels)
        self.adwm = AdaptiveDualWeightModule(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        block_out = x
        for block, adjust in zip(self.blocks, self.adjust):
            block_out = adjust(torch.cat([maybe_checkpoint(block, block_out, self.use_checkpoint), block_out], dim=1))
        return residual + self.adwm(block_out, self.mcm(x))


class QuantumLayer(nn.Module):
    """Small quantum-inspired nonlinear layer backed by PennyLane."""

    def __init__(self, in_features: int, n_qubits: int = 8, n_q_layers: int = 3) -> None:
        super().__init__()
        if not QML_AVAILABLE:
            raise ImportError("PennyLane is required for QuantumLayer.")

        self.q_in_linear = nn.Linear(in_features, n_qubits)
        self.q_out_linear = nn.Linear(n_qubits, in_features)
        self.q_weights = nn.Parameter(torch.rand(n_q_layers, n_qubits, 3) * 2 * math.pi)
        self.n_qubits = n_qubits
        self.n_q_layers = n_q_layers
        self.device = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.device, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            for layer in range(n_q_layers):
                for i in range(n_qubits):
                    qml.Rot(weights[layer, i, 0], weights[layer, i, 1], weights[layer, i, 2], wires=i)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                if n_qubits > 1:
                    qml.CNOT(wires=[n_qubits - 1, 0])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_in = self.q_in_linear(x)
        outputs = []
        for item in q_in.float():
            q_out = self.circuit(item, self.q_weights.float())
            outputs.append(torch.stack(q_out))
        q_tensor = torch.stack(outputs, dim=0).to(dtype=x.dtype, device=x.device)
        return self.q_out_linear(q_tensor)


class ClassicalWaveletFallback(nn.Module):
    """Fallback bottleneck used when quantum or wavelet dependencies are unavailable."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ChannelLayerNorm2d(channels),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class HybridQuantumWaveletBottleneck(nn.Module):
    """Hybrid quantum-wavelet bottleneck for multi-scale nonlinear transformation."""

    def __init__(
        self,
        channels: int,
        resolution_hw: tuple[int, int],
        n_qubits: int = 8,
        n_q_layers: int = 3,
        wavelet: str = "haar",
        allow_fallback: bool = True,
    ) -> None:
        super().__init__()
        self.use_fallback = False

        if not PWT_AVAILABLE or not QML_AVAILABLE:
            if not allow_fallback:
                missing = []
                if not PWT_AVAILABLE:
                    missing.append("pytorch_wavelets")
                if not QML_AVAILABLE:
                    missing.append("PennyLane")
                raise ImportError(f"Missing required dependencies: {', '.join(missing)}")
            warnings.warn(
                "PennyLane or pytorch_wavelets is unavailable. "
                "Using a classical convolutional fallback bottleneck.",
                RuntimeWarning,
            )
            self.use_fallback = True
            self.fallback = ClassicalWaveletFallback(channels)
            return

        self.dwt = pwt.DWTForward(J=1, wave=wavelet, mode="zero")
        self.idwt = pwt.DWTInverse(wave=wavelet, mode="zero")

        height, width = resolution_hw
        if height <= 0 or width <= 0:
            raise ValueError(f"Invalid bottleneck resolution: {(height, width)}")

        sub_height, sub_width = math.ceil(height / 2), math.ceil(width / 2)
        flat_dim = channels * sub_height * sub_width

        args = (flat_dim, n_qubits, n_q_layers)
        self.q_processor_ll = QuantumLayer(*args)
        self.q_processor_lh = QuantumLayer(*args)
        self.q_processor_hl = QuantumLayer(*args)
        self.q_processor_hh = QuantumLayer(*args)
        self.fuse = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_fallback:
            return self.fallback(x)

        batch, _, height, width = x.shape
        with autocast(device_type=x.device.type, enabled=False):
            low, high_list = self.dwt(x.float())

        high = high_list[0]
        sub_shape = low.shape
        low_freq = low
        high_lh = high[:, :, 0, :, :]
        high_hl = high[:, :, 1, :, :]
        high_hh = high[:, :, 2, :, :]

        low_proc = self.q_processor_ll(low_freq.reshape(batch, -1)).view(sub_shape)
        lh_proc = self.q_processor_lh(high_lh.reshape(batch, -1)).view(sub_shape)
        hl_proc = self.q_processor_hl(high_hl.reshape(batch, -1)).view(sub_shape)
        hh_proc = self.q_processor_hh(high_hh.reshape(batch, -1)).view(sub_shape)

        high_processed = [torch.stack([lh_proc, hl_proc, hh_proc], dim=2)]
        with autocast(device_type=x.device.type, enabled=False):
            reconstructed = self.idwt((low_proc.float(), [h.float() for h in high_processed]))

        if reconstructed.shape[-2:] != x.shape[-2:]:
            reconstructed = F.interpolate(reconstructed, size=(height, width), mode="bilinear", align_corners=False)
        return x + self.fuse(reconstructed.to(x.dtype))


class Decoder(nn.Module):
    """Upsampling decoder with skip connection fusion."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, num_blocks: int, use_checkpoint: bool = True):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        self.fuse = nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=1)
        self.unit = DecoderUnit(out_channels, num_blocks, use_checkpoint=use_checkpoint)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.unit(self.fuse(torch.cat([x, skip], dim=1)))


class QWRFNet(nn.Module):
    """Quantum-wavelet rectified-flow network for precipitation nowcasting."""

    __is_rf__ = True

    def __init__(
        self,
        in_channels: int = 12,
        base_embed: int = 24,
        num_timesteps: int = 1000,
        cond_channels: int = 6,
        use_ckpt: bool = True,
        input_hw: int = 288,
        allow_dependency_fallback: bool = True,
        use_quantum_wavelet: bool = True,
    ) -> None:
        super().__init__()
        self.num_timesteps = num_timesteps
        self.in_channels = in_channels
        self.cond_channels = cond_channels
        embed = base_embed

        self.convin = nn.Conv2d(in_channels, embed, 3, 1, 1)
        self.cond_encoder = nn.ModuleDict(
            {
                "init": nn.Conv2d(cond_channels, embed, 3, 1, 1),
                "down1": nn.Sequential(nn.GELU(), nn.Conv2d(embed, embed * 2, 3, 2, 1)),
                "down2": nn.Sequential(nn.GELU(), nn.Conv2d(embed * 2, embed * 4, 3, 2, 1)),
                "down3": nn.Sequential(nn.GELU(), nn.Conv2d(embed * 4, embed * 8, 3, 2, 1)),
            }
        )

        self.enc_cond_proj = nn.ModuleDict(
            {
                "proj1": nn.Conv2d(embed, embed, 1),
                "proj2": nn.Conv2d(embed * 2, embed * 2, 1),
                "proj3": nn.Conv2d(embed * 4, embed * 4, 1),
                "proj_bott": nn.Conv2d(embed * 8, embed * 8, 1),
            }
        )
        self.enc_cond_fuse = nn.ModuleDict(
            {
                "fuse0": nn.Conv2d(embed * 2, embed, 1),
                "fuse1": nn.Conv2d(embed * 2, embed, 1),
                "fuse2": nn.Conv2d(embed * 4, embed * 2, 1),
                "fuse3": nn.Conv2d(embed * 8, embed * 4, 1),
                "fuse_bott": nn.Conv2d(embed * 16, embed * 8, 1),
            }
        )

        self.temb = SinusoidalTimestepEmbedding(embed)
        self.to_s1 = nn.Linear(embed, embed)
        self.to_s2 = nn.Linear(embed, embed * 2)
        self.to_s3 = nn.Linear(embed, embed * 4)
        self.to_s4 = nn.Linear(embed, embed * 8)

        self.enc1 = Encoder(embed, 5, use_checkpoint=use_ckpt)
        self.d1 = DownSampleBlock(embed, embed * 2)
        self.enc2 = Encoder(embed * 2, 6, use_checkpoint=use_ckpt)
        self.d2 = DownSampleBlock(embed * 2, embed * 4)
        self.enc3 = Encoder(embed * 4, 6, use_checkpoint=use_ckpt)
        self.d3 = DownSampleBlock(embed * 4, embed * 8)

        bottleneck_hw = max(1, input_hw // 8)
        if use_quantum_wavelet:
            self.bottom = HybridQuantumWaveletBottleneck(
                channels=embed * 8,
                resolution_hw=(bottleneck_hw, bottleneck_hw),
                n_qubits=10,
                n_q_layers=3,
                allow_fallback=allow_dependency_fallback,
            )
        else:
            self.bottom = ClassicalWaveletFallback(embed * 8)

        self.dec1 = Decoder(embed * 8, embed * 4, embed * 4, 6, use_checkpoint=use_ckpt)
        self.dec2 = Decoder(embed * 4, embed * 2, embed * 2, 6, use_checkpoint=use_ckpt)
        self.dec3 = Decoder(embed * 2, embed, embed, 5, use_checkpoint=use_ckpt)

        self.head = nn.Conv2d(embed, in_channels, kernel_size=3, padding=1)
        self.horizon_affine = nn.Parameter(torch.zeros(in_channels, 2))

    def inject_time(self, feat: torch.Tensor, tvec: torch.Tensor, proj: nn.Linear) -> torch.Tensor:
        """Inject a projected time embedding into a feature map."""
        return feat + proj(tvec).unsqueeze(-1).unsqueeze(-1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1).repeat(1, self.in_channels, 1, 1)
        if x.shape[1] != self.in_channels:
            if x.shape[1] == 1:
                x = x.repeat(1, self.in_channels, 1, 1)
            else:
                raise ValueError(f"Expected {self.in_channels} input channels, got {x.shape[1]}.")

        tvec = self.temb(t, self.num_timesteps)
        x0 = self.convin(x)

        cond_feats = {}
        if cond is not None:
            c0 = self.cond_encoder["init"](cond)
            c1 = self.cond_encoder["down1"](c0)
            c2 = self.cond_encoder["down2"](c1)
            c3 = self.cond_encoder["down3"](c2)
            cond_feats = {"c0": c0, "c1": c1, "c2": c2, "c3": c3}
            x0 = self.enc_cond_fuse["fuse0"](torch.cat([x0, c0], dim=1))

        x0 = self.inject_time(x0, tvec, self.to_s1)
        x1 = self.enc1(x0)
        if cond is not None:
            x1 = self.enc_cond_fuse["fuse1"](torch.cat([x1, self.enc_cond_proj["proj1"](cond_feats["c0"])], dim=1))

        x2 = self.inject_time(self.d1(x1), tvec, self.to_s2)
        x3 = self.enc2(x2)
        if cond is not None:
            x3 = self.enc_cond_fuse["fuse2"](torch.cat([x3, self.enc_cond_proj["proj2"](cond_feats["c1"])], dim=1))

        x4 = self.inject_time(self.d2(x3), tvec, self.to_s3)
        x5 = self.enc3(x4)
        if cond is not None:
            x5 = self.enc_cond_fuse["fuse3"](torch.cat([x5, self.enc_cond_proj["proj3"](cond_feats["c2"])], dim=1))

        x6 = self.inject_time(self.d3(x5), tvec, self.to_s4)
        bottleneck = self.bottom(x6)
        if cond is not None:
            bottleneck = self.enc_cond_fuse["fuse_bott"](
                torch.cat([bottleneck, self.enc_cond_proj["proj_bott"](cond_feats["c3"])], dim=1)
            )

        d1 = self.dec1(bottleneck, x5)
        d2 = self.dec2(d1, x3)
        d3 = self.dec3(d2, x1)

        out = self.head(d3)
        scale = (1 + self.horizon_affine[:, 0]).view(1, -1, 1, 1)
        bias = self.horizon_affine[:, 1].view(1, -1, 1, 1)
        return out * scale + bias
