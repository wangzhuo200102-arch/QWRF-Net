
import os
import random
import h5py
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import json
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import Resize
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.multiprocessing import spawn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ================== 与库接口保持一致的统一配置 ==================
NUM_TIMESTEPS = 1000
NUM_SAMPLING_STEPS_TRAIN = 10   # 调度器（训练）
NUM_SAMPLING_STEPS_EVAL  = 50  # 评估/可视化更稳
USE_DISCRETE_TIMESTEPS = False
USE_TIMESTEP_TRANSFORM = False
SCALE_TEMPORAL = True

# 预测帧数（6 → 12）
T_FRAMES = 12

# ================== 评估增强开关 ==================
ENABLE_SSIM = True     # 结构指标
# 可选 PSD 你之前说“可选”，这里先不默认启用（需要额外频谱统计与画图）
ENABLE_PSD  = False

# ================== Extreme 子集定义 ==================
EXTREME_PEAK_TH = 219.0     # VIL 阈值（常用 16/74/133/219，这里 extreme 用 219）
EXTREME_AREA_TH = 0.02      # 超过阈值的像素占比 >= 2%
EXTREME_MAX_CASES = 20      # 每模型最多可视化多少个 extreme case

# ================== 普通样本对比图 ==================
RANDOM_VIS_SAMPLES = 12     # 每模型随机画多少个样本
RANDOM_VIS_SHOW_ERROR = False

# ================== 你的 RF-UNet（forward(x,t,cond)） ==================
# 假设这些模型文件存在于您的项目中
from model import QWRFNet

# ================== 生成流调度器 & 采样器（不改库源码） ==================
from schedulers.rf.rectified_flow import RFlowScheduler
from schedulers.rf import RFLOW as RFLOW2D

# ================== 静默采样封装 ==================
import io, contextlib
@contextlib.contextmanager
def suppress_rf_prints():
    _out, _err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(_out), contextlib.redirect_stderr(_err):
        yield

def rf_sample_quiet(rf_sampler, model, z, device, condition=None, additional_args=None, progress=False):
    with suppress_rf_prints():
        if hasattr(rf_sampler, 'sampleno'):
            return rf_sampler.sampleno(
                model=model, z=z, condition=condition, device=device,
                additional_args=additional_args, progress=progress
            )
        raise AttributeError("RFLOW 不包含 'sampleno'。")

# ===================================================================
# 0. 工具：简化 SSIM（torch版）
# ===================================================================
def _ssim_simple(x, y, eps=1e-6):
    """
    x,y: [B,H,W] or [H,W]  (float)
    返回 batch 平均 SSIM（简化版，按全图统计，不滑窗）
    """
    if x.dim() == 2:
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
    # [B,H,W]
    mu_x = x.mean(dim=(1,2))
    mu_y = y.mean(dim=(1,2))
    var_x = x.var(dim=(1,2), unbiased=False)
    var_y = y.var(dim=(1,2), unbiased=False)
    cov_xy = ((x - mu_x[:,None,None]) * (y - mu_y[:,None,None])).mean(dim=(1,2))

    # 常用常数（这里用动态范围 255，跟你 physical 一致）
    L = 255.0
    C1 = (0.01 * L)**2
    C2 = (0.03 * L)**2

    ssim = ((2*mu_x*mu_y + C1) * (2*cov_xy + C2)) / ((mu_x**2 + mu_y**2 + C1) * (var_x + var_y + C2) + eps)
    return ssim.mean()

# ===================================================================
# 0. 工具：Extreme 判断
# ===================================================================
def is_extreme_event(targets_phy, peak_th=EXTREME_PEAK_TH, area_th=EXTREME_AREA_TH):
    """
    targets_phy: [B,12,H,W] in physical scale (0..255)
    return: extreme_mask [B] bool, peak [B], area_ratio [B]
    """
    peak = targets_phy.amax(dim=(1,2,3))  # [B]
    area_ratio = (targets_phy > peak_th).float().mean(dim=(1,2,3))  # [B]
    extreme = (peak >= peak_th) & (area_ratio >= area_th)
    return extreme, peak, area_ratio

# ===================================================================
# 0. 可视化：GT12 vs Pred12（可加 |Err|）
# ===================================================================
def save_compare_grid_12(gt, pred, file_path, max_vil, title=None, t_stride=1, show_error=True):
    """
    gt/pred: [T,H,W] tensor, 值域 [0,1]
    输出：
      show_error=False: 2行×T列 (GT/Pred)
      show_error=True : 3行×T列 (GT/Pred/|Error|)
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    gt   = gt.detach().float().clamp(0, 1).cpu() * max_vil
    pred = pred.detach().float().clamp(0, 1).cpu() * max_vil
    err  = (pred - gt).abs()

    gt_np, pred_np, err_np = gt.numpy(), pred.numpy(), err.numpy()
    T, H, W = gt_np.shape
    idxs = list(range(0, T, t_stride))
    ncol = len(idxs)

    nrow = 3 if show_error else 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.1 * ncol, 2.8 * nrow), dpi=140)
    axes = np.array(axes).reshape(nrow, ncol)

    if title:
        fig.suptitle(title, fontsize=12)

    cmap, vmin, vmax = 'gist_ncar', 0, 255
    im = None

    for ci, t in enumerate(idxs):
        # GT
        ax = axes[0, ci]
        im = ax.imshow(gt_np[t], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
        if ci == 0: ax.set_ylabel('GT', fontsize=10)
        ax.set_title(f'T+{5*(t+1)}min', fontsize=9)

        # Pred
        ax = axes[1, ci]
        ax.imshow(pred_np[t], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
        if ci == 0: ax.set_ylabel('Pred', fontsize=10)

        # Error
        if show_error:
            ax = axes[2, ci]
            ax.imshow(err_np[t], cmap='magma')
            ax.set_xticks([]); ax.set_yticks([])
            if ci == 0: ax.set_ylabel('|Err|', fontsize=10)

    # colorbar for VIL (GT/Pred)
    cax = fig.add_axes([0.92, 0.36 if show_error else 0.20, 0.015, 0.50])
    fig.colorbar(im, cax=cax, ticks=[0, 50, 100, 150, 200, 255], label='VIL')

    fig.subplots_adjust(left=0.02, right=0.90, bottom=0.08, top=0.90, wspace=0.05, hspace=0.15)
    fig.savefig(file_path, bbox_inches='tight')
    plt.close(fig)

# ===================================================================
# 0. 可视化：随机样本对比
# ===================================================================
def visualize_random_samples_12(model, loader, device, out_dir, max_vil, num_samples=8, seed=0, show_error=False):
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)

    rf_sampler = RFLOW2D(
        num_timesteps=NUM_TIMESTEPS,
        num_sampling_steps=NUM_SAMPLING_STEPS_EVAL,
        use_discrete_timesteps=USE_DISCRETE_TIMESTEPS,
        use_timestep_transform=USE_TIMESTEP_TRANSFORM,
        scale_temporal=SCALE_TEMPORAL,
    )

    model.eval()
    picked = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            if picked >= num_samples:
                break
            b = inputs.shape[0]
            sel = rng.randrange(b)

            inputs = inputs.to(device)
            targets = targets.to(device)

            H, W = targets.shape[-2], targets.shape[-1]
            z = torch.randn((b, T_FRAMES, H, W), device=device)
            preds = rf_sample_quiet(rf_sampler, model=model, z=z, device=device, condition=inputs, additional_args=None, progress=False).clamp(0, 1)

            gt_one   = targets[sel]
            pred_one = preds[sel]

            save_path = os.path.join(out_dir, f"sample_{picked:04d}_b{batch_idx:04d}_i{sel:02d}.png")
            save_compare_grid_12(
                gt=gt_one, pred=pred_one, file_path=save_path, max_vil=max_vil,
                title=f"Random Sample #{picked} (batch={batch_idx}, idx={sel})",
                show_error=show_error
            )
            picked += 1

# ===================================================================
# 0. 可视化：Extreme cases 对比（GT/Pred/Err）
# ===================================================================
def visualize_extreme_cases_12(model, loader, device, out_dir, max_vil,
                               peak_th=EXTREME_PEAK_TH, area_th=EXTREME_AREA_TH, max_cases=20, show_error=True):
    os.makedirs(out_dir, exist_ok=True)

    rf_sampler = RFLOW2D(
        num_timesteps=NUM_TIMESTEPS,
        num_sampling_steps=NUM_SAMPLING_STEPS_EVAL,
        use_discrete_timesteps=USE_DISCRETE_TIMESTEPS,
        use_timestep_transform=USE_TIMESTEP_TRANSFORM,
        scale_temporal=SCALE_TEMPORAL,
    )

    model.eval()
    saved = 0
    meta_list = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            if saved >= max_cases:
                break

            inputs = inputs.to(device)
            targets = targets.to(device)

            targets_phy = targets * max_vil
            extreme_mask, peak, area_ratio = is_extreme_event(targets_phy, peak_th=peak_th, area_th=area_th)
            if extreme_mask.sum().item() == 0:
                continue

            B = inputs.shape[0]
            H, W = targets.shape[-2], targets.shape[-1]
            z = torch.randn((B, T_FRAMES, H, W), device=device)
            preds = rf_sample_quiet(rf_sampler, model=model, z=z, device=device, condition=inputs, additional_args=None, progress=False).clamp(0, 1)

            idxs = torch.nonzero(extreme_mask).flatten().tolist()
            for sel in idxs:
                if saved >= max_cases:
                    break
                gt_one   = targets[sel]
                pred_one = preds[sel]
                p = float(peak[sel].item())
                a = float(area_ratio[sel].item())

                save_path = os.path.join(out_dir, f"extreme_{saved:04d}_b{batch_idx:04d}_i{sel:02d}_peak{p:.1f}_area{a:.4f}.png")
                save_compare_grid_12(
                    gt=gt_one, pred=pred_one, file_path=save_path, max_vil=max_vil,
                    title=f"Extreme #{saved} | peak={p:.1f} | area@{peak_th:.0f}={a:.4f}",
                    show_error=show_error
                )
                meta_list.append({"batch": batch_idx, "idx": sel, "peak": p, "area_ratio": a, "png": os.path.basename(save_path)})
                saved += 1

    with open(os.path.join(out_dir, "extreme_list.json"), "w") as f:
        json.dump(meta_list, f, indent=2)

    print(f"[Extreme] saved {saved} cases to: {out_dir}")

# ===================================================================
# 1. 数据预处理（保持不变）
# ===================================================================
def process_h5_file_worker(args):
    file_path, max_vil = args
    passed_events = []
    try:
        with h5py.File(file_path, 'r') as hf:
            if 'vil' not in hf or hf['vil'].ndim != 4:
                return []
            all_events_in_file = hf['vil']
            for event_idx in range(all_events_in_file.shape[0]):
                identifier = (file_path, event_idx)
                frame_hwt = all_events_in_file[event_idx, :, :, 9]
                total_pixels = frame_hwt.size
                is_quality_failed = False
                if np.sum(frame_hwt == 0) > 0.8 * total_pixels or \
                   np.sum(frame_hwt == max_vil) > 0.8 * total_pixels:
                    is_quality_failed = True
                else:
                    event_data_hwt = all_events_in_file[event_idx, :, :, :18]
                    if event_data_hwt.max() <= 0:
                        is_quality_failed = True
                if not is_quality_failed:
                    event_data_thw = np.transpose(event_data_hwt, (2, 0, 1)).astype(np.float32)
                    stats = [event_data_thw.max(), event_data_thw.min(), event_data_thw.mean(), event_data_thw.var()]
                    passed_events.append((identifier, stats))
    except Exception:
        return []
    return passed_events

def prepare_dataset_index(data_path, max_vil, index_file_path):
    print("--- 开始数据预处理（此过程仅在第一次运行时执行）---")
    all_h5_files = []
    for year in ['2017', '2018', '2019']:
        year_dir = os.path.join(data_path, year)
        if not os.path.exists(year_dir):
            continue
        for root, _, files in os.walk(year_dir):
            for file in files:
                if file.endswith('.h5'):
                    all_h5_files.append(os.path.join(root, file))
    if not all_h5_files:
        raise RuntimeError("在指定路径下未找到任何 .h5 文件。")

    tasks = [(fp, max_vil) for fp in all_h5_files]
    quality_check_passed, stats_for_kmeans = [], []
    with Pool(processes=cpu_count()) as pool:
        results_list = list(tqdm(pool.imap_unordered(process_h5_file_worker, tasks),
                                 total=len(tasks), desc="并行质量检查中"))
    for single_file_results in results_list:
        for identifier, stats in single_file_results:
            quality_check_passed.append(identifier)
            stats_for_kmeans.append(stats)
    if not quality_check_passed:
        raise RuntimeError("没有任何事件通过质量检查")

    stats_array = np.array(stats_for_kmeans)
    scaler = StandardScaler()
    kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(scaler.fit_transform(stats_array))
    cluster_centers_unscaled = scaler.inverse_transform(kmeans.cluster_centers_)
    valid_cluster_id = np.argmin(cluster_centers_unscaled[:, 2])
    final_valid_identifiers = [identifier for i, identifier in enumerate(quality_check_passed) if labels[i] == valid_cluster_id]

    with open(index_file_path, 'w') as f:
        json.dump(final_valid_identifiers, f)
    print(f"数据索引已创建并保存至: {index_file_path}  (len={len(final_valid_identifiers)})")
    return final_valid_identifiers

# ===================================================================
# 2. 数据集
# ===================================================================
class SEVIRPrecipitationDataset(Dataset):
    def __init__(self, index_file_path, max_vil, img_size):
        self.max_vil = max_vil
        self.resize = Resize(img_size, antialias=True)
        try:
            with open(index_file_path, 'r') as f:
                self.sample_info = json.load(f)
        except FileNotFoundError:
            raise RuntimeError(f"错误：索引文件 {index_file_path} 未找到。请先运行数据预处理。")
    def __len__(self):
        return len(self.sample_info)
    def __getitem__(self, idx):
        file_path, frame_idx = self.sample_info[idx]
        with h5py.File(file_path, 'r') as hf:
            v_hwt = hf['vil'][frame_idx, ..., :18].astype(np.float32)
        v_thw = np.transpose(v_hwt, (2, 0, 1))
        v_tensor = torch.from_numpy(v_thw).contiguous()
        inputs = self.resize(v_tensor[:6, :, :]) / self.max_vil          # [6,H,W]  -> condition
        targets = self.resize(v_tensor[6:18, :, :]) / self.max_vil        # [12,H,W] -> x_start
        return inputs, targets

# ===================================================================
# 3. 评估（增强：MSE/RMSE/MAE/SSIM + Extreme subset）
# ===================================================================
def evaluate_all_metrics_per_timestep_rf(model, loader, device, max_vil, thresholds, desc="Final Evaluation (RF)"):
    model.eval()
    T = T_FRAMES
    is_main_process = dist.get_rank() == 0

    # 连续指标：per step
    mse_sum = torch.zeros(T, device=device)
    mae_sum = torch.zeros(T, device=device)
    ssim_sum = torch.zeros(T, device=device) if ENABLE_SSIM else None
    pixel_count = torch.zeros(T, device=device)

    # 分类指标：per step
    stats_per_step = {t: {th: {'tp': 0, 'fp': 0, 'fn': 0} for th in thresholds} for t in range(T)}

    # extreme 子集累计（sample级别：只对 extreme 样本累加）
    extreme_count = torch.tensor(0, device=device)
    extreme_mse_sum = torch.zeros(T, device=device)
    extreme_mae_sum = torch.zeros(T, device=device)
    extreme_ssim_sum = torch.zeros(T, device=device) if ENABLE_SSIM else None
    extreme_pixel_count = torch.zeros(T, device=device)
    extreme_stats_per_step = {t: {th: {'tp': 0, 'fp': 0, 'fn': 0} for th in thresholds} for t in range(T)}

    # 强度分桶（按 GT peak 分桶，统计 MAE/RMSE）
    # bins: [0,74), [74,133), [133,160), [160,181), [181,219), [219,255]
    intensity_bins = [(0,74), (74,133), (133,160), (160,181), (181,219), (219,256)]
    bin_mae_sum = torch.zeros(len(intensity_bins), device=device)
    bin_mse_sum = torch.zeros(len(intensity_bins), device=device)
    bin_pix = torch.zeros(len(intensity_bins), device=device)

    rf_sampler = RFLOW2D(
        num_timesteps=NUM_TIMESTEPS,
        num_sampling_steps=NUM_SAMPLING_STEPS_EVAL,
        use_discrete_timesteps=USE_DISCRETE_TIMESTEPS,
        use_timestep_transform=USE_TIMESTEP_TRANSFORM,
        scale_temporal=SCALE_TEMPORAL,
    )

    eval_bar = tqdm(loader, desc=desc, disable=not is_main_process, leave=False)

    with torch.no_grad():
        for inputs, targets in eval_bar:
            inputs = inputs.to(device)
            targets = targets.to(device)  # [B,12,H,W]
            B = inputs.shape[0]
            H, W = targets.shape[-2], targets.shape[-1]
            z = torch.randn((B, T, H, W), device=device)

            preds = rf_sample_quiet(
                rf_sampler, model=model, z=z, device=device,
                condition=inputs, additional_args=None, progress=False
            ).clamp(0, 1)

            pred_phy = preds * max_vil
            tgt_phy  = targets * max_vil

            # extreme mask（按整个 12帧 stack 判断）
            extreme_mask, peak, area_ratio = is_extreme_event(tgt_phy, peak_th=EXTREME_PEAK_TH, area_th=EXTREME_AREA_TH)

            # 分桶：按 GT peak 归类（按样本）
            # peak: [B]
            for bi,(lo,hi) in enumerate(intensity_bins):
                in_bin = (peak >= lo) & (peak < hi)
                if in_bin.any():
                    # 用所有像素 & 所有时间步的误差累计（更稳定）
                    diff = (pred_phy[in_bin] - tgt_phy[in_bin]).abs()
                    bin_mae_sum[bi] += diff.sum()
                    bin_mse_sum[bi] += (diff**2).sum()
                    bin_pix[bi] += torch.tensor(diff.numel(), device=device)

            # per-step 指标
            for t_step in range(T):
                pf = pred_phy[:, t_step]
                tf = tgt_phy[:, t_step]

                diff = pf - tf
                mse_sum[t_step] += (diff**2).sum()
                mae_sum[t_step] += diff.abs().sum()
                pixel_count[t_step] += tf.numel()

                if ENABLE_SSIM:
                    ssim_sum[t_step] += _ssim_simple(pf, tf)

                # categorical
                for th in thresholds:
                    pm, tm = pf > th, tf > th
                    stats_per_step[t_step][th]['tp'] += (pm & tm).sum().item()
                    stats_per_step[t_step][th]['fp'] += (pm & ~tm).sum().item()
                    stats_per_step[t_step][th]['fn'] += (~pm & tm).sum().item()

            # extreme 子集 per-step 累计
            if extreme_mask.any():
                extreme_count += extreme_mask.sum()
                for t_step in range(T):
                    pf = pred_phy[extreme_mask, t_step]
                    tf = tgt_phy[extreme_mask, t_step]
                    diff = pf - tf
                    extreme_mse_sum[t_step] += (diff**2).sum()
                    extreme_mae_sum[t_step] += diff.abs().sum()
                    extreme_pixel_count[t_step] += tf.numel()
                    if ENABLE_SSIM:
                        extreme_ssim_sum[t_step] += _ssim_simple(pf, tf)
                    for th in thresholds:
                        pm, tm = pf > th, tf > th
                        extreme_stats_per_step[t_step][th]['tp'] += (pm & tm).sum().item()
                        extreme_stats_per_step[t_step][th]['fp'] += (pm & ~tm).sum().item()
                        extreme_stats_per_step[t_step][th]['fn'] += (~pm & tm).sum().item()

    # ========= DDP 汇总 =========
    dist.all_reduce(mse_sum); dist.all_reduce(mae_sum); dist.all_reduce(pixel_count)
    if ENABLE_SSIM:
        dist.all_reduce(ssim_sum)
    dist.all_reduce(extreme_count)
    dist.all_reduce(extreme_mse_sum); dist.all_reduce(extreme_mae_sum); dist.all_reduce(extreme_pixel_count)
    if ENABLE_SSIM:
        dist.all_reduce(extreme_ssim_sum)
    dist.all_reduce(bin_mae_sum); dist.all_reduce(bin_mse_sum); dist.all_reduce(bin_pix)

    # categorical 汇总（用 tensor all_reduce）
    def _reduce_stats(stats_dict):
        for t_step in range(T):
            for th in thresholds:
                for key in ['tp','fp','fn']:
                    v = torch.tensor(stats_dict[t_step][th][key], device=device, dtype=torch.long)
                    dist.all_reduce(v)
                    stats_dict[t_step][th][key] = int(v.item())

    _reduce_stats(stats_per_step)
    _reduce_stats(extreme_stats_per_step)

    # ========= 计算连续指标 =========
    mse_per_step = (mse_sum / (pixel_count + 1e-8)).cpu().tolist()
    rmse_per_step = (torch.sqrt(mse_sum / (pixel_count + 1e-8))).cpu().tolist()
    mae_per_step = (mae_sum / (pixel_count + 1e-8)).cpu().tolist()
    ssim_per_step = (ssim_sum / (torch.ones_like(ssim_sum) + 1e-8)).cpu().tolist() if ENABLE_SSIM else None

    # ========= 计算 categorical（per-step曲线 + overall） =========
    def _calc_categorical(stats_dict):
        csi_per_step = {th: [] for th in thresholds}
        hss_per_step = {th: [] for th in thresholds}
        pod_per_step = {th: [] for th in thresholds}
        far_per_step = {th: [] for th in thresholds}

        for t_step in range(T):
            tot = int(pixel_count[t_step].item())  # 用全样本的像素数
            if tot == 0:
                continue
            for th in thresholds:
                tp = stats_dict[t_step][th]['tp']
                fp = stats_dict[t_step][th]['fp']
                fn = stats_dict[t_step][th]['fn']
                tn = tot - (tp + fp + fn)
                csi_per_step[th].append(tp / (tp + fp + fn + 1e-8))
                hss_per_step[th].append((2*(tp*tn - fp*fn))/((tp+fn)*(fn+tn)+(tp+fp)*(fp+tn)+1e-8))
                pod_per_step[th].append(tp / (tp + fn + 1e-8))
                far_per_step[th].append(fp / (tp + fp + 1e-8))

        total_pixels_all = int(pixel_count.sum().item())
        csi_overall, hss_overall, pod_overall, far_overall = {}, {}, {}, {}
        if total_pixels_all > 0:
            for th in thresholds:
                TP = sum(stats_dict[t][th]['tp'] for t in range(T))
                FP = sum(stats_dict[t][th]['fp'] for t in range(T))
                FN = sum(stats_dict[t][th]['fn'] for t in range(T))
                TN = total_pixels_all - (TP + FP + FN)
                csi_overall[th] = TP / (TP + FP + FN + 1e-8)
                hss_overall[th] = (2*(TP*TN - FP*FN))/((TP+FN)*(FN+TN)+(TP+FP)*(FP+TN)+1e-8)
                pod_overall[th] = TP / (TP + FN + 1e-8)
                far_overall[th] = FP / (TP + FP + 1e-8)
        return csi_per_step, hss_per_step, pod_per_step, far_per_step, csi_overall, hss_overall, pod_overall, far_overall

    csi_per_step, hss_per_step, pod_per_step, far_per_step, csi_overall, hss_overall, pod_overall, far_overall = _calc_categorical(stats_per_step)

    # ========= Extreme 连续指标 =========
    extreme_mse_per_step = (extreme_mse_sum / (extreme_pixel_count + 1e-8)).cpu().tolist()
    extreme_rmse_per_step = (torch.sqrt(extreme_mse_sum / (extreme_pixel_count + 1e-8))).cpu().tolist()
    extreme_mae_per_step = (extreme_mae_sum / (extreme_pixel_count + 1e-8)).cpu().tolist()
    extreme_ssim_per_step = (extreme_ssim_sum / (torch.ones_like(extreme_ssim_sum) + 1e-8)).cpu().tolist() if ENABLE_SSIM else None

    # extreme categorical（注意：pixel_count 换成 extreme_pixel_count 更合理）
    # 为简单一致，这里只输出 per-step 和 overall（overall 按 extreme_pixel_count.sum 计）
    def _calc_categorical_extreme(stats_dict, pix_count_tensor):
        csi_per_step = {th: [] for th in thresholds}
        hss_per_step = {th: [] for th in thresholds}
        pod_per_step = {th: [] for th in thresholds}
        far_per_step = {th: [] for th in thresholds}
        for t_step in range(T):
            tot = int(pix_count_tensor[t_step].item())
            if tot == 0:
                continue
            for th in thresholds:
                tp = stats_dict[t_step][th]['tp']
                fp = stats_dict[t_step][th]['fp']
                fn = stats_dict[t_step][th]['fn']
                tn = tot - (tp + fp + fn)
                csi_per_step[th].append(tp / (tp + fp + fn + 1e-8))
                hss_per_step[th].append((2*(tp*tn - fp*fn))/((tp+fn)*(fn+tn)+(tp+fp)*(fp+tn)+1e-8))
                pod_per_step[th].append(tp / (tp + fn + 1e-8))
                far_per_step[th].append(fp / (tp + fp + 1e-8))

        total_pixels_all = int(pix_count_tensor.sum().item())
        csi_overall, hss_overall, pod_overall, far_overall = {}, {}, {}, {}
        if total_pixels_all > 0:
            for th in thresholds:
                TP = sum(stats_dict[t][th]['tp'] for t in range(T))
                FP = sum(stats_dict[t][th]['fp'] for t in range(T))
                FN = sum(stats_dict[t][th]['fn'] for t in range(T))
                TN = total_pixels_all - (TP + FP + FN)
                csi_overall[th] = TP / (TP + FP + FN + 1e-8)
                hss_overall[th] = (2*(TP*TN - FP*FN))/((TP+FN)*(FN+TN)+(TP+FP)*(FP+TN)+1e-8)
                pod_overall[th] = TP / (TP + FN + 1e-8)
                far_overall[th] = FP / (TP + FP + 1e-8)
        return csi_per_step, hss_per_step, pod_per_step, far_per_step, csi_overall, hss_overall, pod_overall, far_overall

    ex_csi_per_step, ex_hss_per_step, ex_pod_per_step, ex_far_per_step, ex_csi_overall, ex_hss_overall, ex_pod_overall, ex_far_overall = _calc_categorical_extreme(extreme_stats_per_step, extreme_pixel_count)

    # ========= 强度分桶误差 =========
    bin_mae = (bin_mae_sum / (bin_pix + 1e-8)).cpu().tolist()
    bin_rmse = (torch.sqrt(bin_mse_sum / (bin_pix + 1e-8))).cpu().tolist()
    bin_info = []
    for bi,(lo,hi) in enumerate(intensity_bins):
        bin_info.append({
            "bin": f"[{lo},{hi})",
            "mae": float(bin_mae[bi]),
            "rmse": float(bin_rmse[bi]),
            "pixels": float(bin_pix[bi].item())
        })

    metrics = {
        "mse_per_step": mse_per_step,
        "rmse_per_step": rmse_per_step,
        "mae_per_step": mae_per_step,
        "ssim_per_step": ssim_per_step,

        "csi_per_step": csi_per_step,
        "hss_per_step": hss_per_step,
        "pod_per_step": pod_per_step,
        "far_per_step": far_per_step,
        "csi_overall": csi_overall,
        "hss_overall": hss_overall,
        "pod_overall": pod_overall,
        "far_overall": far_overall,

        "extreme": {
            "num_extreme_samples": int(extreme_count.item()),
            "peak_th": float(EXTREME_PEAK_TH),
            "area_th": float(EXTREME_AREA_TH),

            "mse_per_step": extreme_mse_per_step,
            "rmse_per_step": extreme_rmse_per_step,
            "mae_per_step": extreme_mae_per_step,
            "ssim_per_step": extreme_ssim_per_step,

            "csi_per_step": ex_csi_per_step,
            "hss_per_step": ex_hss_per_step,
            "pod_per_step": ex_pod_per_step,
            "far_per_step": ex_far_per_step,
            "csi_overall": ex_csi_overall,
            "hss_overall": ex_hss_overall,
            "pod_overall": ex_pod_overall,
            "far_overall": ex_far_overall,

            "intensity_bins_error": bin_info
        }
    }
    return metrics

# ===================================================================
# 4. 指标保存与画图
# ===================================================================
def save_metrics_to_file(metrics, model_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"final_metrics_{model_name}.json")
    with open(file_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"已将所有详细评估指标保存至: {file_path}")

def plot_skill_score_curves(metrics, model_name, thresholds, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    time_points = np.arange(5, 5*(T_FRAMES+1), 5)

    csi = metrics["csi_per_step"]
    hss = metrics["hss_per_step"]
    pod = metrics["pod_per_step"]
    far = metrics["far_per_step"]

    num_thresholds = len(thresholds)
    fig, axes = plt.subplots(num_thresholds, 4, figsize=(28, 5 * num_thresholds), squeeze=False)
    fig.suptitle(f'Skill Scores vs. Time | Model: {model_name}', fontsize=16, y=1.0)

    for i, th in enumerate(thresholds):
        axes[i, 0].plot(time_points, csi.get(th, []), 'o-'); axes[i, 0].set_title(f'CSI (Th={th})'); axes[i, 0].grid(True)
        axes[i, 1].plot(time_points, hss.get(th, []), 'o-'); axes[i, 1].set_title(f'HSS (Th={th})'); axes[i, 1].grid(True)
        axes[i, 2].plot(time_points, pod.get(th, []), 'o-'); axes[i, 2].set_title(f'POD (Th={th})'); axes[i, 2].grid(True)
        axes[i, 3].plot(time_points, far.get(th, []), 'o-'); axes[i, 3].set_title(f'FAR (Th={th})'); axes[i, 3].grid(True)

    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    out = os.path.join(output_dir, "skill_score_curves.png")
    plt.savefig(out); plt.close(fig)
    print(f"已保存技能评分曲线图至: {out}")

def plot_continuous_metrics_curves(metrics, model_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    time_points = np.arange(5, 5*(T_FRAMES+1), 5)

    mse  = metrics["mse_per_step"]
    rmse = metrics["rmse_per_step"]
    mae  = metrics["mae_per_step"]
    ssim = metrics["ssim_per_step"]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=140)
    ax.plot(time_points, mse,  'o-', label='MSE')
    ax.plot(time_points, rmse, 'o-', label='RMSE')
    ax.plot(time_points, mae,  'o-', label='MAE')
    if ENABLE_SSIM and ssim is not None:
        ax.plot(time_points, ssim, 'o-', label='SSIM')
    ax.grid(True)
    ax.set_title(f'Continuous/Structural Metrics vs Time | {model_name}')
    ax.set_xlabel('Lead time (min)')
    ax.legend()
    out = os.path.join(output_dir, "continuous_metrics_curves.png")
    plt.savefig(out); plt.close(fig)
    print(f"已保存连续/结构指标曲线图至: {out}")

# ===================================================================
# 5. 早停（保持不变）
# ===================================================================
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss; self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

# ===================================================================
# 6. 训练主逻辑
# ===================================================================
def train_worker(rank, world_size, index_file_path):
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    device = torch.device(f'cuda:{rank}')
    is_main_process = (rank == 0)

    try:
        SEED, MAX_VIL, IMG_SIZE = 42, 255.0, (288, 288)
        PER_DEVICE_BATCH_SIZE, ACCUMULATION_STEPS = 6, 1
        NUM_WORKERS, EPOCHS, PATIENCE = 4, 300, 20
        THRESHOLDS = [16, 74, 133, 160, 181, 219]
        BASE_RESULTS_DIR = "resultfivexiaoronggfinabBC"

        # 训练中可视化（quick samples），如果你想训练中也看到对比图，调小这个
        SAMPLE_DURING_VAL = False
        VAL_SAMPLE_MAX_BATCHES = 4
        VIS_INTERVAL_EPOCHS = 50

        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = True

        dataset = SEVIRPrecipitationDataset(index_file_path, MAX_VIL, IMG_SIZE)
        dist.barrier()
        train_size = int(0.8 * len(dataset)); val_size = len(dataset) - train_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))

        train_sampler = DistributedSampler(train_ds, shuffle=True)
        val_sampler = DistributedSampler(val_ds, shuffle=False)

        train_loader = DataLoader(
            train_ds, batch_size=PER_DEVICE_BATCH_SIZE, sampler=train_sampler,
            num_workers=NUM_WORKERS, pin_memory=True, drop_last=True, persistent_workers=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=PER_DEVICE_BATCH_SIZE, sampler=val_sampler,
            num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True
        )

        model_dicts = {
            "QWRFNet":QWRFNet,
        }

        for model_name, model_class in model_dicts.items():
            model_results_dir = os.path.join(BASE_RESULTS_DIR, model_name)
            if is_main_process:
                print(f"\n{'='*25} 训练模型: {model_name} on {world_size} GPUs {'='*25}")
                os.makedirs(model_results_dir, exist_ok=True)

            ddp_model = DDP(
                model_class(in_channels=12, base_embed=24, num_timesteps=NUM_TIMESTEPS, cond_channels=6).to(device),
                device_ids=[rank],
                find_unused_parameters=False
            )

            optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-4, weight_decay=1e-4)
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
            early_stopping = EarlyStopping(patience=PATIENCE)
            scaler = torch.amp.GradScaler()
            best_val_loss = float('inf')

            rf_sched = RFlowScheduler(
                num_timesteps=NUM_TIMESTEPS,
                num_sampling_steps=NUM_SAMPLING_STEPS_TRAIN,
                sample_method="uniform",
                use_discrete_timesteps=USE_DISCRETE_TIMESTEPS,
                use_timestep_transform=USE_TIMESTEP_TRANSFORM,
                transform_scale=1.0,
                scale_temporal=SCALE_TEMPORAL,
            )

            rf_sampler = RFLOW2D(
                num_timesteps=NUM_TIMESTEPS,
                num_sampling_steps=NUM_SAMPLING_STEPS_EVAL,
                use_discrete_timesteps=USE_DISCRETE_TIMESTEPS,
                use_timestep_transform=USE_TIMESTEP_TRANSFORM,
                scale_temporal=SCALE_TEMPORAL,
            )

            for epoch in range(1, EPOCHS + 1):
                train_sampler.set_epoch(epoch)
                ddp_model.train()
                train_loss_sum = torch.tensor(0.0, device=device)
                train_items_count = torch.tensor(0, device=device)

                train_bar = tqdm(train_loader, desc=f'E{epoch}/{EPOCHS} [Train]', disable=not is_main_process)
                optimizer.zero_grad()

                for i, (inputs, targets) in enumerate(train_bar):
                    inputs = inputs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)

                    with torch.amp.autocast(device_type='cuda'):
                        x_start = targets
                        loss_dict = rf_sched.training_losses(
                            model=ddp_model, x_start=x_start, condition=inputs,
                            model_kwargs=None
                        )
                        loss = loss_dict["loss"]
                        if loss.dim() != 0:
                            loss = loss.mean()
                        loss_for_backward = loss / ACCUMULATION_STEPS

                    scaler.scale(loss_for_backward).backward()

                    train_loss_sum += loss.item() * targets.size(0)
                    train_items_count += targets.size(0)

                    if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()

                dist.all_reduce(train_loss_sum); dist.all_reduce(train_items_count)
                avg_train_loss = train_loss_sum.item() / (train_items_count.item() + 1e-8)

                # ---------- Val ----------
                val_loss_sum = torch.tensor(0.0, device=device)
                val_items_count = torch.tensor(0, device=device)

                ddp_model.eval()
                val_bar = tqdm(val_loader, desc=f'E{epoch}/{EPOCHS} [Val]', disable=not is_main_process)
                with torch.no_grad():
                    for j, (inputs, targets) in enumerate(val_bar):
                        inputs = inputs.to(device, non_blocking=True)
                        targets = targets.to(device, non_blocking=True)

                        with torch.amp.autocast(device_type='cuda'):
                            x_start = targets
                            vloss_dict = rf_sched.training_losses(
                                model=ddp_model, x_start=x_start, condition=inputs,
                                model_kwargs=None
                            )
                            vloss = vloss_dict["loss"]
                            if vloss.dim() != 0:
                                vloss = vloss.mean()

                        val_loss_sum += vloss.item() * targets.size(0)
                        val_items_count += targets.size(0)

                        # 可选：训练中抽样并保存 quick grid
                        if SAMPLE_DURING_VAL and (epoch % VIS_INTERVAL_EPOCHS == 0) and j < 1:
                            B = inputs.shape[0]
                            H, W = targets.shape[-2], targets.shape[-1]
                            z = torch.randn((B, T_FRAMES, H, W), device=device)
                            sampled = rf_sample_quiet(rf_sampler, model=ddp_model, z=z, device=device, condition=inputs, additional_args=None, progress=False).clamp(0,1)
                            if is_main_process:
                                quick_dir = os.path.join(model_results_dir, 'val_quick_samples')
                                os.makedirs(quick_dir, exist_ok=True)
                                quick_path = os.path.join(quick_dir, f'ep{epoch:03d}_b{j:04d}.png')
                                save_compare_grid_12(
                                    gt=targets[0], pred=sampled[0], file_path=quick_path,
                                    max_vil=MAX_VIL, title=f'{model_name} QuickVal (ep{epoch})',
                                    show_error=True
                                )

                dist.all_reduce(val_loss_sum); dist.all_reduce(val_items_count)
                val_loss = val_loss_sum.item() / (val_items_count.item() + 1e-8)

                lr_scheduler.step(val_loss)
                if is_main_process:
                    print(f'Epoch {epoch}/{EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}')
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(ddp_model.module.state_dict(), os.path.join(model_results_dir, 'best_model.pth'))
                        print(f"新低验证损失 (Val Loss: {best_val_loss:.6f})，已保存最优模型。")

                # ---------- Early stop 同步 ----------
                stop_decision = early_stopping(val_loss) if is_main_process else False
                stop_flag = torch.tensor(int(stop_decision), dtype=torch.int, device=device)
                dist.broadcast(stop_flag, src=0)
                if stop_flag.item() == 1:
                    if is_main_process:
                        print(f'\n早停触发于第 {epoch} 轮 (由 Rank 0 决策并同步)。')
                    break

                dist.barrier()

            # ========== 最优模型加载 & 最终评估 ==========
            if is_main_process:
                print(f"\n训练完成: {model_name}。加载最优模型进行最终评估...")
            dist.barrier()

            best_model_path = os.path.join(model_results_dir, 'best_model.pth')
            if os.path.exists(best_model_path):
                ddp_model.module.load_state_dict(torch.load(best_model_path, map_location=device))
            dist.barrier()

            final_metrics = evaluate_all_metrics_per_timestep_rf(ddp_model, val_loader, device, MAX_VIL, THRESHOLDS)

            dist.barrier()
            if is_main_process:
                # 打印连续指标（overall）
                print(f"\n--- 模型 '{model_name}' Final Continuous Metrics ---")
                print(f"  Overall MSE : {float(np.mean(final_metrics['mse_per_step'])):.4f}")
                print(f"  Overall RMSE: {float(np.mean(final_metrics['rmse_per_step'])):.4f}")
                print(f"  Overall MAE : {float(np.mean(final_metrics['mae_per_step'])):.4f}")
                if ENABLE_SSIM and final_metrics['ssim_per_step'] is not None:
                    print(f"  Overall SSIM: {float(np.mean(final_metrics['ssim_per_step'])):.4f}")

                # 打印分类指标表（overall）
                print(f"\n--- 模型 '{model_name}' Final Skill Scores (Overall) ---")
                header = f"{'Threshold':>10} | {'CSI':>8} | {'HSS':>8} | {'POD':>8} | {'FAR':>8}"
                print(header); print("-" * len(header))
                for th in THRESHOLDS:
                    csi = final_metrics['csi_overall'].get(th, -1)
                    hss = final_metrics['hss_overall'].get(th, -1)
                    pod = final_metrics['pod_overall'].get(th, -1)
                    far = final_metrics['far_overall'].get(th, -1)
                    print(f"{th:>10} | {csi:>8.4f} | {hss:>8.4f} | {pod:>8.4f} | {far:>8.4f}")
                print("-" * len(header))

                # 打印 extreme 子集概况
                ex = final_metrics["extreme"]
                print(f"\n--- Extreme Subset (peak>={ex['peak_th']}, area>={ex['area_th']}) ---")
                print(f"  num_extreme_samples: {ex['num_extreme_samples']}")
                if ex["num_extreme_samples"] > 0:
                    print(f"  Extreme Overall MSE : {float(np.mean(ex['mse_per_step'])):.4f}")
                    print(f"  Extreme Overall RMSE: {float(np.mean(ex['rmse_per_step'])):.4f}")
                    print(f"  Extreme Overall MAE : {float(np.mean(ex['mae_per_step'])):.4f}")
                    if ENABLE_SSIM and ex["ssim_per_step"] is not None:
                        print(f"  Extreme Overall SSIM: {float(np.mean(ex['ssim_per_step'])):.4f}")

                    print("  Intensity bins error (by GT peak):")
                    for b in ex["intensity_bins_error"]:
                        print(f"    {b['bin']:>10} | MAE={b['mae']:.4f} | RMSE={b['rmse']:.4f} | pixels={b['pixels']:.0f}")

                # 保存与画图
                save_metrics_to_file(final_metrics, model_name, model_results_dir)
                plot_skill_score_curves(final_metrics, model_name, THRESHOLDS, model_results_dir)
                plot_continuous_metrics_curves(final_metrics, model_name, model_results_dir)

                # ========== 可视化：普通样本 vs extreme 样本 ==========
                rand_dir = os.path.join(model_results_dir, "random_samples")
                visualize_random_samples_12(
                    model=ddp_model, loader=val_loader, device=device,
                    out_dir=rand_dir, max_vil=MAX_VIL,
                    num_samples=RANDOM_VIS_SAMPLES, seed=SEED, show_error=RANDOM_VIS_SHOW_ERROR
                )

                extreme_dir = os.path.join(model_results_dir, "extreme_cases")
                visualize_extreme_cases_12(
                    model=ddp_model, loader=val_loader, device=device,
                    out_dir=extreme_dir, max_vil=MAX_VIL,
                    peak_th=EXTREME_PEAK_TH, area_th=EXTREME_AREA_TH,
                    max_cases=EXTREME_MAX_CASES, show_error=True
                )

            # 清理
            if is_main_process:
                print(f"--- 完成对模型 '{model_name}' 的处理，正在清理资源... ---")
            del ddp_model, optimizer, lr_scheduler, early_stopping, scaler, rf_sched, rf_sampler
            torch.cuda.empty_cache()
            dist.barrier()

        dist.barrier()

    except Exception as e:
        if rank == 0:
            print(f"进程 {rank} 发生致命错误: {e}")
            traceback.print_exc()
    finally:
        dist.destroy_process_group()

# ===================================================================
# 7. 启动
# ===================================================================
def main():
    DATA_PATH = r'/home/ps/W-U/data/vil'
    GPUS_TO_USE = "0,1,2,3,4,5,6,7,8,9"
    INDEX_FILE = "sevir_valid_samples.json"
    MAX_VIL = 255.0

    if not os.path.exists(INDEX_FILE):
        prepare_dataset_index(DATA_PATH, MAX_VIL, INDEX_FILE)

    os.environ['CUDA_VISIBLE_DEVICES'] = GPUS_TO_USE
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29502'
    os.environ['NCCL_P2P_DISABLE'] = '0'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    try:
        world_size = len(GPUS_TO_USE.split(',')) if GPUS_TO_USE.strip() else 0
        if world_size > 0:
            spawn(train_worker, args=(world_size, INDEX_FILE), nprocs=world_size, join=True)
        else:
            print("没有配置GPU，程序退出。请在 'GPUS_TO_USE' 中设置要使用的GPU编号。")
    except Exception as e:
        print(f"启动 DDP 进程失败: {e}")

if __name__ == '__main__':
    main()
