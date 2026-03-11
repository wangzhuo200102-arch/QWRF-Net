import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions import LogisticNormal


def extract_hw(model_kwargs):
    """简化版本：只处理2D数据的高度和宽度"""
    # Force fp16 input to fp32 to avoid nan output
    for key in ["height", "width"]:
        if key in model_kwargs:
            if isinstance(model_kwargs[key], torch.Tensor) and model_kwargs[key].dtype == torch.float16:
                model_kwargs[key] = model_kwargs[key].float()

    # 如果有height和width，计算resolution；否则使用默认值
    if "height" in model_kwargs and "width" in model_kwargs:
        resolution = model_kwargs["height"] * model_kwargs["width"]
    else:
        resolution = torch.ones(1)  # 默认值
    
    return resolution


def timestep_transform_2d(
    t,
    model_kwargs=None,
    base_resolution=512 * 512,
    scale=1.0,
    num_timesteps=1,
    ret_ratio=False,
):
    """简化的2D时间步变换"""
    t = t / num_timesteps

    if model_kwargs is not None:
        resolution = extract_hw(model_kwargs)
        ratio = (resolution / base_resolution).sqrt() * scale
    else:
        ratio = torch.tensor(scale)  # 如果没有model_kwargs，直接使用scale
    
    # 应用变换
    t = ratio * t / (1 + (ratio - 1) * t)
    t = t * num_timesteps
    
    if ret_ratio:
        return t, ratio
    return t


class TimeSampler2D:
    """简化的2D时间采样器"""
    def __init__(
        self,
        sample_method="uniform",
        use_discrete_timesteps=False,
        use_timestep_transform=False,
        transform_scale=1.0,
        loc=0.0,
        scale=1.0,
    ):
        assert sample_method in ["uniform", "logit-normal"]
        assert (
            sample_method == "uniform" or not use_discrete_timesteps
        ), "Only uniform sampling is supported for discrete timesteps"

        self.sample_method = sample_method
        self.use_discrete_timesteps = use_discrete_timesteps
        self.use_timestep_transform = use_timestep_transform
        self.transform_scale = transform_scale
        
        if sample_method == "logit-normal":
            self.distribution = LogisticNormal(torch.tensor([loc]), torch.tensor([scale]))
            self.sample_t = lambda x: self.distribution.sample((x.shape[0],))[:, 0].to(x.device)

    def sample(self, x_start, num_timesteps, model_kwargs=None):
        """采样时间步"""
        if self.use_discrete_timesteps:
            t = torch.randint(1, num_timesteps, (x_start.shape[0],), device=x_start.device)# 从第一帧数采样,第0帧数算作起点
        elif self.sample_method == "uniform":
            t = torch.rand((x_start.shape[0],), device=x_start.device) * num_timesteps
        elif self.sample_method == "logit-normal":
            t = self.sample_t(x_start) * num_timesteps

        if not self.use_timestep_transform:
            return t

        t = timestep_transform_2d(
            t,
            model_kwargs,
            scale=self.transform_scale,
            num_timesteps=num_timesteps,
        )

        return t

    def visualize(self, height=512, width=512, num_timesteps=16):
        """可视化时间采样分布"""
        bs = 1000
        x_start = torch.randn(bs)
        
        # 原始时间分布
        self.use_timestep_transform = False
        original_t_values = self.sample(x_start, num_timesteps)
        
        # 变换后的时间分布
        self.use_timestep_transform = True
        model_kwargs = {
            "height": torch.full((bs,), height),
            "width": torch.full((bs,), width),
        }
        
        transformed_t_values, ratio = timestep_transform_2d(
            original_t_values,
            model_kwargs,
            scale=self.transform_scale,
            num_timesteps=num_timesteps,
            ret_ratio=True,
        )

        # 绘图
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # 散点图显示变换关系
        axes[0].scatter(
            original_t_values.numpy(),
            transformed_t_values.numpy(),
            alpha=0.6,
            s=10,
        )
        axes[0].plot([0, num_timesteps], [0, num_timesteps], 'r--', alpha=0.5, label='y=x')
        axes[0].set_xlabel("Original t")
        axes[0].set_ylabel("Transformed t")
        axes[0].set_title(f"Time Transform (ratio={ratio[0].item():.3f})")
        axes[0].legend()
        axes[0].grid(True)

        # 直方图显示分布
        bins = np.linspace(0, num_timesteps, 50)
        axes[1].hist(original_t_values.numpy(), bins=bins, alpha=0.6, label="Original", density=True)
        axes[1].hist(transformed_t_values.numpy(), bins=bins, alpha=0.6, label="Transformed", density=True)
        axes[1].set_xlabel("Time step")
        axes[1].set_ylabel("Density")
        axes[1].set_title("Time Distribution")
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig("./timestep_sampling_2d.png", dpi=150)
        print("Saved visualization to timestep_sampling_2d.png")
        
        return original_t_values, transformed_t_values


# 更简单的版本，如果你不需要复杂的时间变换
class SimpleTimeSampler:
    """最简单的时间采样器，适用于序列数据"""
    def __init__(self, sample_method="uniform"):
        self.sample_method = sample_method
    
    def sample(self, batch_size, num_timesteps, device):
        """直接采样时间步"""
        if self.sample_method == "uniform":
            return torch.randint(0, num_timesteps, (batch_size,), device=device)
        elif self.sample_method == "continuous":
            return torch.rand(batch_size, device=device) * num_timesteps
        else:
            raise ValueError(f"Unknown sample_method: {self.sample_method}")


# if __name__ == "__main__":
#     # 测试简化的时间采样器
#     time_sampler = TimeSampler2D(
#         use_timestep_transform=True,
#         sample_method="uniform",
#         use_discrete_timesteps=True,
#         transform_scale=1.0,
#     )
#     x_ = torch.rand(size=(4,4,32,32))
#     t = time_sampler.sample(x_start=x_,num_timesteps=16)
#     print(t)
#     # 可视化
#     original, transformed = time_sampler.visualize(num_timesteps=16)
#     print(f"Original range: [{original.min():.2f}, {original.max():.2f}]")
#     print(f"Transformed range: [{transformed.min():.2f}, {transformed.max():.2f}]")