import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _ckpt
from torch.amp import autocast

# 尝试导入可选依赖
try:
    import pennylane as qml
    from pennylane import qnn
    _QML_AVAILABLE = True
except ImportError:
    _QML_AVAILABLE = False
    print("警告：PennyLane 未安装。量子模块将不可用。")

try:
    import pytorch_wavelets as pwt
    _PWT_AVAILABLE = True
except ImportError:
    _PWT_AVAILABLE = False
    print("警告：pytorch_wavelets 未安装，瓶颈模块将无法工作。请运行 'pip install pytorch_wavelets'")


__all__ = ["HQWUNetRFTenW"]


# ===================================================================
# Part 1: 基础工具和辅助模块
# ===================================================================

class ChannelLayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias


def _maybe_ckpt(fn, x, use_ckpt: bool):
    if use_ckpt and x.requires_grad:
        try:
            return _ckpt(fn, x, use_reentrant=False, preserve_rng_state=False)
        except TypeError:
            return _ckpt(fn, x, preserve_rng_state=False)
    return fn(x)

def add_coords(x):
    B, C, H, W = x.shape
    d = x.device
    xx = torch.linspace(-1, 1, W, device=d).view(1, 1, 1, W).expand(B, 1, H, W)
    yy = torch.linspace(-1, 1, H, device=d).view(1, 1, H, 1).expand(B, 1, H, W)
    return torch.cat([x, xx, yy], dim=1)

class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, t, num_timesteps):
        t = t.float().clamp(min=0) / max(float(num_timesteps), 1.0)
        half = self.dim // 2
        freqs = torch.exp(-math.log(self.max_period) * torch.arange(0, half, device=t.device) / half)
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.proj(emb)

# ===================================================================
# Part 2: 核心构建模块
# ===================================================================

class WindowAttention(nn.Module):
    def __init__(self, in_channels, window_size=7):
        super().__init__()
        self.window_size = window_size
        self.inner_dim = in_channels + 2
        self.qkv = nn.Conv2d(self.inner_dim, 3 * self.inner_dim, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(self.inner_dim, in_channels, kernel_size=1)

    def forward(self, x):
        x_with_coords = add_coords(x)
        B, C, H, W = x_with_coords.shape
        ws = self.window_size
        pad_h, pad_w = (ws - H % ws) % ws, (ws - W % ws) % ws
        x_p = F.pad(x_with_coords, (0, pad_w, 0, pad_h))
        Hp, Wp = x_p.shape[2:]
        qkv = self.qkv(x_p)
        qkv = qkv.unfold(2, ws, ws).unfold(3, ws, ws).contiguous().view(B, 3 * C, -1, ws, ws).permute(0, 2, 1, 3, 4).contiguous().view(-1, 3 * C, ws * ws)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        attn = (q.transpose(-2, -1) @ k) / (C ** 0.5)
        attn = attn.softmax(dim=-1)
        out = (v @ attn.transpose(-2, -1)).view(-1, C, ws, ws)
        nW = Hp // ws
        out = out.view(B, nW, Wp // ws, C, ws, ws).permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, Hp, Wp)
        return self.proj(out[:, :, :H, :W])

class DownSampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.norm = ChannelLayerNorm2d(in_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
    def forward(self, x): return self.conv(self.norm(x))

class MSAtten(nn.Module):
    def __init__(self, channels, window_size=7):
        super().__init__()
        self.down_med = DownSampleBlock(channels, channels * 2)
        self.down_sml = DownSampleBlock(channels, channels * 4, stride=4)
        self.attn_lg = WindowAttention(channels, window_size)
        self.attn_md = WindowAttention(channels * 2, window_size)
        self.attn_sm = WindowAttention(channels * 4, window_size)
        self.conv_md = nn.Conv2d(channels * 2, channels, 1)
        self.conv_sm = nn.Conv2d(channels * 4, channels, 1)
        self.fuse = nn.Conv2d(channels * 3, channels, 1)
    def forward(self, x):
        B, C, H, W = x.shape
        lg = self.attn_lg(x)
        md = self.conv_md(self.attn_md(self.down_med(x)))
        md = F.interpolate(md, (H, W), mode='bilinear', align_corners=False)
        sm = self.conv_sm(self.attn_sm(self.down_sml(x)))
        sm = F.interpolate(sm, (H, W), mode='bilinear', align_corners=False)
        return self.fuse(torch.cat([lg, md, sm], dim=1))

class FFN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 1), nn.GELU(),
            nn.Conv2d(channels * 4, channels * 4, 3, padding=1, groups=channels * 4),
            nn.Conv2d(channels * 4, channels, 1)
        )
    def forward(self, x): return self.net(x)

class MSM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm1 = ChannelLayerNorm2d(channels)
        self.attn = MSAtten(channels)
        self.norm2 = ChannelLayerNorm2d(channels)
        self.ffn = FFN(channels)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))

class ADWM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_x = nn.Conv2d(channels, channels, 1)
        self.conv_y = nn.Conv2d(channels, channels, 1)
        self.conv_a = nn.Sequential(nn.Conv2d(channels, 1, 1), nn.Sigmoid())
        self.fuse = nn.Conv2d(channels, channels, 1)
    def forward(self, x, y):
        idt = x + y
        a = self.conv_a(self.conv_x(x) * self.conv_y(y)).expand(-1, x.shape[1], -1, -1)
        out = (1 - a) * x + a * y
        return idt + self.fuse(out)

class MCM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = ChannelLayerNorm2d(channels)
        self.dw1 = nn.Conv2d(channels, channels * 6, 3, padding=1, groups=channels)
        self.pw1 = nn.Conv2d(channels * 6, channels * 6, 1)
        self.dw2 = nn.Conv2d(channels * 3, channels * 3, 3, padding=1, groups=channels * 3)
        self.pw2 = nn.Conv2d(channels * 3, channels, 1)
    def forward(self, x):
        sc = self.norm(x)
        a, b = torch.chunk(self.pw1(self.dw1(sc)), 2, dim=1)
        return sc + self.pw2(self.dw2(F.relu(a) * b))

class Encoder(nn.Module):
    def __init__(self, channels, num_msm, use_ckpt=True):
        super().__init__()
        self.use_ckpt = use_ckpt
        self.msms = nn.ModuleList([MSM(channels) for _ in range(num_msm)])
        self.adj = nn.ModuleList([nn.Conv2d(channels * 2, channels, 1) for _ in range(num_msm)])
        self.mcm = MCM(channels)
        self.adwm = ADWM(channels)
    def forward(self, x):
        msm_out = x
        for msm, conv in zip(self.msms, self.adj):
            msm_out = conv(torch.cat([_maybe_ckpt(msm, msm_out, self.use_ckpt), msm_out], dim=1))
        return self.adwm(msm_out, self.mcm(x))

class DecoderUnit(nn.Module):
    def __init__(self, channels, num_msm, use_ckpt=True):
        super().__init__()
        self.use_ckpt = use_ckpt
        self.msms = nn.ModuleList([MSM(channels) for _ in range(num_msm)])
        self.adj = nn.ModuleList([nn.Conv2d(channels * 2, channels, 1) for _ in range(num_msm)])
        self.mcm = MCM(channels)
        self.adwm = ADWM(channels)
    def forward(self, x):
        res, msm_out = x, x
        for msm, conv in zip(self.msms, self.adj):
            msm_out = conv(torch.cat([_maybe_ckpt(msm, msm_out, self.use_ckpt), msm_out], dim=1))
        return res + self.adwm(msm_out, self.mcm(x))


# ===================================================================
# Part 3: ★★★ 最终版：四路并行混合量子小波瓶颈 ★★★
# ===================================================================

def create_quantum_circuit(n_qubits, n_q_layers):
    if not _QML_AVAILABLE: raise ImportError("PennyLane is not installed.")
    dev = qml.device("default.qubit", wires=n_qubits)
    @qml.qnode(dev, interface='torch', diff_method='backprop')
    def quantum_circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        for layer in range(n_q_layers):
            for i in range(n_qubits): qml.Rot(weights[layer, i, 0], weights[layer, i, 1], weights[layer, i, 2], wires=i)
            for i in range(n_qubits - 1): qml.CNOT(wires=[i, i + 1])
            if n_qubits > 1: qml.CNOT(wires=[n_qubits - 1, 0])
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    return quantum_circuit

class QuantumLayer(nn.Module):
    def __init__(self, in_features, n_qubits=8, n_q_layers=3): # 默认量子电路为3层
        super().__init__()
        if not _QML_AVAILABLE: raise ImportError("PennyLane is not installed.")
        self.q_in_linear = nn.Linear(in_features, n_qubits)
        self.q_circuit = create_quantum_circuit(n_qubits, n_q_layers)
        self.q_weights = nn.Parameter(torch.rand(n_q_layers, n_qubits, 3) * 2 * math.pi)
        self.q_out_linear = nn.Linear(n_qubits, in_features)

    def forward(self, x):
        q_in = self.q_in_linear(x)
        q_out_list = self.q_circuit(q_in.float(), self.q_weights.float())
        q_out = torch.stack(q_out_list, dim=1).to(x.dtype)
        return self.q_out_linear(q_out)

class HybridQuantumWaveletBottleneck(nn.Module):
    def __init__(self, channels, resolution_hw, n_qubits=8, n_q_layers=3, wavelet='haar'):
        super().__init__()
        if not _PWT_AVAILABLE: raise ImportError("pytorch_wavelets is not installed.")
        self.dwt = pwt.DWTForward(J=1, wave=wavelet, mode='zero')
        self.idwt = pwt.DWTInverse(wave=wavelet, mode='zero')

        h, w = resolution_hw
        if h <= 0 or w <= 0: raise ValueError(f"无效的瓶颈分辨率: {(h, w)}")
        
        sh, sw = math.ceil(h / 2), math.ceil(w / 2)
        flat_dim = channels * sh * sw
        
        # ★★★ 改进：为 LL, LH, HL, HH 四个分量创建独立的量子处理器 ★★★
        args = (flat_dim, n_qubits, n_q_layers)
        self.q_processor_ll = QuantumLayer(*args) # 处理低频 (轮廓)
        self.q_processor_lh = QuantumLayer(*args) # 处理水平细节
        self.q_processor_hl = QuantumLayer(*args) # 处理垂直细节
        self.q_processor_hh = QuantumLayer(*args) # 处理对角细节

        self.fuse = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 1. 应用2D DWT
        with autocast(device_type='cuda', enabled=False):
            yl, yh_list = self.dwt(x.float())
        
        yh = yh_list[0]
        sub_shape = yl.shape # (B, C, H/2, W/2)

        # 2. ★★★ 改进：并行处理所有四个子带 ★★★
        # a. 提取所有子带
        y_ll = yl
        y_lh = yh[:, :, 0, :, :]
        y_hl = yh[:, :, 1, :, :]
        y_hh = yh[:, :, 2, :, :]

        # b. 压平并送入各自的量子处理器
        y_ll_proc = self.q_processor_ll(y_ll.reshape(B, -1)).view(sub_shape)
        y_lh_proc = self.q_processor_lh(y_lh.reshape(B, -1)).view(sub_shape)
        y_hl_proc = self.q_processor_hl(y_hl.reshape(B, -1)).view(sub_shape)
        y_hh_proc = self.q_processor_hh(y_hh.reshape(B, -1)).view(sub_shape)
        
        # 3. ★★★ 改进：用所有处理过的子带重组高频张量 ★★★
        yh_processed_list = [torch.stack([y_lh_proc, y_hl_proc, y_hh_proc], dim=2)]

        # 4. 应用逆2D IDWT
        with autocast(device_type='cuda', enabled=False):
            reconstructed = self.idwt((y_ll_proc.float(), [h.float() for h in yh_processed_list]))
        
        # 5. 确保尺寸匹配并进行残差融合
        if reconstructed.shape[-2:] != x.shape[-2:]:
            reconstructed = F.interpolate(reconstructed, size=(H, W), mode='bilinear', align_corners=False)
            
        return x + self.fuse(reconstructed.to(x.dtype))

# ===================================================================
# Part 4: 主模型架构
# ===================================================================

class Decoder(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, num_msm, use_ckpt: bool = True):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        )
        self.fuse = nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=1)
        self.unit = DecoderUnit(out_ch, num_msm, use_ckpt=use_ckpt)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            # The provided snippet used F.interpolate(x, ...) which is a bug.
            # It should interpolate the skip connection to match the upsampled x if needed.
            # Sticking to the original logic for safety.
            skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
        fused = self.fuse(torch.cat([x, skip], dim=1))
        return self.unit(fused)

class QWRFNet(nn.Module):
    __is_rf__ = True
    def __init__(self, in_channels=12, base_embed=24, num_timesteps=1000, cond_channels=6, use_ckpt: bool = True, input_hw: int = 288):
        super().__init__()
        self.num_timesteps, self.in_channels, self.cond_channels = num_timesteps, in_channels, cond_channels
        E = base_embed

        self.convin = nn.Conv2d(in_channels, E, 3, 1, 1)

        self.cond_encoder = nn.ModuleDict({
            'init': nn.Conv2d(cond_channels, E, 3, 1, 1),
            'down1': nn.Sequential(nn.GELU(), nn.Conv2d(E, E * 2, 3, 2, 1)),
            'down2': nn.Sequential(nn.GELU(), nn.Conv2d(E * 2, E * 4, 3, 2, 1)),
            'down3': nn.Sequential(nn.GELU(), nn.Conv2d(E * 4, E * 8, 3, 2, 1)),
        })

        self.enc_cond_proj = nn.ModuleDict({
            'proj1': nn.Conv2d(E, E, 1),
            'proj2': nn.Conv2d(E * 2, E * 2, 1),
            'proj3': nn.Conv2d(E * 4, E * 4, 1),
            'proj_bott': nn.Conv2d(E * 8, E * 8, 1),
        })

        self.enc_cond_fuse = nn.ModuleDict({
            'fuse0': nn.Conv2d(E * 2, E, 1),
            'fuse1': nn.Conv2d(E * 2, E, 1),
            'fuse2': nn.Conv2d(E * 4, E * 2, 1),
            'fuse3': nn.Conv2d(E * 8, E * 4, 1),
            'fuse_bott': nn.Conv2d(E * 16, E * 8, 1),
        })

        self.temb = SinusoidalTimestepEmbedding(E)
        self.to_s1, self.to_s2 = nn.Linear(E, E), nn.Linear(E, E*2)
        self.to_s3, self.to_s4 = nn.Linear(E, E*4), nn.Linear(E, E*8)

        self.enc1, self.d1 = Encoder(E, 5, use_ckpt=use_ckpt), DownSampleBlock(E, E*2)
        self.enc2, self.d2 = Encoder(E*2, 6, use_ckpt=use_ckpt), DownSampleBlock(E*2, E*4)
        self.enc3, self.d3 = Encoder(E*4, 6, use_ckpt=use_ckpt), DownSampleBlock(E*4, E*8)
        
        # ★★★ 使用最终版的瓶颈模块 ★★★
        bottleneck_h_w = input_hw // 8
        self.bottom = HybridQuantumWaveletBottleneck(
            channels=E * 8,
            resolution_hw=(bottleneck_h_w, bottleneck_h_w),
            n_qubits=10,
            n_q_layers=3 # 指定每个量子处理器内部有3层
        )
        
        self.dec1 = Decoder(E*8, E*4, E*4, 6, use_ckpt=use_ckpt)
        self.dec2 = Decoder(E*4, E*2, E*2, 6, use_ckpt=use_ckpt)
        self.dec3 = Decoder(E*2, E, E, 5, use_ckpt=use_ckpt)

        self.head = nn.Conv2d(E, 12, kernel_size=3, padding=1)
        self.horizon_affine = nn.Parameter(torch.zeros(12, 2))

    def inject_time(self, feat, tvec, proj):
        return feat + proj(tvec).unsqueeze(-1).unsqueeze(-1)

    def forward(self, x, t, cond=None):
        tvec = self.temb(t, self.num_timesteps)
        x0 = self.convin(x)

        cond_feats = {}
        if cond is not None:
            c0 = self.cond_encoder['init'](cond)
            cond_feats['c0'] = c0
            c1 = self.cond_encoder['down1'](c0)
            cond_feats['c1'] = c1
            c2 = self.cond_encoder['down2'](c1)
            cond_feats['c2'] = c2
            c3 = self.cond_encoder['down3'](c2)
            cond_feats['c3'] = c3
            x0 = self.enc_cond_fuse['fuse0'](torch.cat([x0, c0], dim=1))

        x0 = self.inject_time(x0, tvec, self.to_s1)
        x1 = self.enc1(x0)
        if cond is not None: x1 = self.enc_cond_fuse['fuse1'](torch.cat([x1, self.enc_cond_proj['proj1'](cond_feats['c0'])], dim=1))

        x2 = self.d1(x1)
        x2 = self.inject_time(x2, tvec, self.to_s2)
        x3 = self.enc2(x2)
        if cond is not None: x3 = self.enc_cond_fuse['fuse2'](torch.cat([x3, self.enc_cond_proj['proj2'](cond_feats['c1'])], dim=1))
        
        x4 = self.d2(x3)
        x4 = self.inject_time(x4, tvec, self.to_s3)
        x5 = self.enc3(x4)
        if cond is not None: x5 = self.enc_cond_fuse['fuse3'](torch.cat([x5, self.enc_cond_proj['proj3'](cond_feats['c2'])], dim=1))

        x6 = self.d3(x5)
        x6 = self.inject_time(x6, tvec, self.to_s4)

        bott = self.bottom(x6)
        if cond is not None: bott = self.enc_cond_fuse['fuse_bott'](torch.cat([bott, self.enc_cond_proj['proj_bott'](cond_feats['c3'])], dim=1))

        d1 = self.dec1(bott, x5)
        d2 = self.dec2(d1,   x3)
        d3 = self.dec3(d2,   x1)

        out = self.head(d3)
        scale = (1 + self.horizon_affine[:, 0]).view(1, -1, 1, 1)
        bias  = self.horizon_affine[:, 1].view(1, -1, 1, 1)
        return out * scale + bias