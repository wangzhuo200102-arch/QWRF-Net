import torch
from tqdm import tqdm

from .rectified_flow import RFlowScheduler
from .time_sampler import timestep_transform_2d as timestep_transform


def dynamic_thresholding(x, ratio=0.995, base=6.0):
    s = torch.quantile(x.abs().flatten(), ratio)
    s = max(s, base)
    x = x.clip(-s, s) * base / s
    return x


class RFLOW2D:
    def __init__(
        self,
        num_sampling_steps=16,  # 序列长度
        num_timesteps=16,       # 序列长度
        cfg_scale=4.0,
        use_discrete_timesteps=True,  # 离散时间
        use_timestep_transform=False,
        transform_scale=1.0,
        **kwargs,
    ):
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

    # def sample_simple(
    #     self,
    #     model,
    #     z,
    #     device,
    #     additional_args=None,
    #     progress=True,
    # ):
    #     """
    #     2D数据
    #     z: [batch_size, channels, height, width] - 初始状态
    #     """
       
    #     timesteps = [(1.0 - i / self.num_sampling_steps) * self.num_timesteps 
    #                 for i in range(self.num_sampling_steps)]
    #     timesteps = timesteps[1:]
    #     if self.use_discrete_timesteps:
    #         timesteps = [int(round(t)) for t in timesteps]
        
    #     timesteps = [torch.tensor([t] * z.shape[0], device=device) for t in timesteps]
    #     # timesteps = timesteps[:len(timesteps)-2]
    #     if self.use_timestep_transform and additional_args is not None:
            
    #         timesteps = [
    #             timestep_transform(
    #                 t,
    #                 additional_args,
    #                 scale=self.transform_scale,
    #                 num_timesteps=self.num_timesteps,
    #                 scale_temporal=False,  # 2D不需要temporal scaling
    #             )
    #             for t in timesteps
    #         ]

    #     progress_wrap = tqdm if progress else (lambda x: x)
            
    #     for i, t in progress_wrap(enumerate(reversed(timesteps))):
    #         # 模型预测速度
    #         # print(t)
    #         if additional_args is not None:
    #             pred = model(z, t, **additional_args)
    #         else:
    #             pred = model(z, t).chunk(2, dim=1)
    #         # print(pred.shape)
    #         # 如果模型返回多个输出，取第一个
    #         if isinstance(pred, tuple):
    #             v_pred = pred[0]
    #         else:
    #             v_pred = pred

    #         # 计算时间步长
    #         dt = timesteps[i] - timesteps[i + 1] if i < len(timesteps) - 1 else timesteps[i]
    #         dt = dt / self.num_timesteps
            
    #         # 更新状态 (2D版本)
    #         z = z + v_pred * dt[:, None, None, None]  # [batch, channel, height, width]

    #     return z
    def sample_simple(
        self,
        model,
        z,
        device,
        additional_args=None,
        progress=True,
    ):
        """
        2D数据，返回时间维度堆叠的结果
        z: [batch_size, channels, height, width] - 初始状态
        返回: [batch_size, channels, time_steps, height, width] - 包含所有中间状态
        """
    
        timesteps = [(1.0 - i / self.num_sampling_steps) * self.num_timesteps 
                    for i in range(self.num_sampling_steps)]
        timesteps = timesteps[1:]
        if self.use_discrete_timesteps:
            timesteps = [int(round(t)) for t in timesteps]

        timesteps = [torch.tensor([t] * z.shape[0], device=device) for t in timesteps]

        if self.use_timestep_transform and additional_args is not None:
            timesteps = [
                timestep_transform(
                    t,
                    additional_args,
                    scale=self.transform_scale,
                    num_timesteps=self.num_timesteps,
                    scale_temporal=False,
                )
                for t in timesteps
            ]

        progress_wrap = tqdm if progress else (lambda x: x)

        # 存储所有中间状态的列表（不包括初始输入z）
        z_trajectory = [z[:,:,0,...]]
        zc = z.clone()
        z = z[:,:,0,...]
        for i, t in progress_wrap(enumerate(reversed(timesteps))):
            print(t)
            # 模型预测速度
            if additional_args is not None:
                pred = model(z, t, **additional_args)
            else:
                pred = model(z, t).chunk(2, dim=1)

            # 如果模型返回多个输出，取第一个
            if isinstance(pred, tuple):
                v_pred = pred[0]
            else:
                v_pred = pred

            # 计算时间步长
            dt = timesteps[i] - timesteps[i + 1] if i < len(timesteps) - 1 else timesteps[i]
            dt = dt / self.num_timesteps

            # 更新状态
            z = z + v_pred * dt[:, None, None, None]+zc[:,:,i,...]

            # 保存当前状态（除了第一次迭代，因为那是输入）
            z_trajectory.append(z.clone())

        # 在时间维度上堆叠所有状态
        # z_trajectory: list of [batch, channels, height, width]
        # 堆叠后: [batch, channels, time_steps, height, width]
        z_stacked = torch.stack(z_trajectory, dim=2)

        return z_stacked
    def sample_with_cfg(
        self,
        model,
        z,
        device,
        additional_args=None,
        guidance_scale=None,
        progress=True,
    ):
        """
        带classifier-free guidance的采样
        """
        if guidance_scale is None:
            guidance_scale = self.cfg_scale

        # 准备时间步
        timesteps = [(1.0 - i / self.num_sampling_steps) * self.num_timesteps 
                    for i in range(self.num_sampling_steps)]
        
        if self.use_discrete_timesteps:
            timesteps = [int(round(t)) for t in timesteps]
        
        timesteps = [torch.tensor([t] * z.shape[0], device=device) for t in timesteps]

        progress_wrap = tqdm if progress else (lambda x: x)

        for i, t in progress_wrap(enumerate(timesteps)):
            # 为CFG准备输入
            z_in = torch.cat([z, z], 0)  # [2*batch, ...]
            t_in = torch.cat([t, t], 0)
            
            # 模型预测
            if additional_args is not None:
                pred = model(z_in, t_in, **additional_args)
            else:
                pred = model(z_in, t_in)
            
            # 处理模型输出
            if isinstance(pred, tuple):
                pred = pred[0]
            
            # 分离条件和无条件预测
            pred_cond, pred_uncond = pred.chunk(2, dim=0)
            
            # Classifier-free guidance
            v_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

            # 计算时间步长
            dt = timesteps[i] - timesteps[i + 1] if i < len(timesteps) - 1 else timesteps[i]
            dt = dt / self.num_timesteps
            
            # 更新状态 
            z = z + v_pred * dt[:, None, None, None]

        return z

    def training_losses(
        self,
        model,
        x_start,
        x_pre,
        model_args=None,
        noise=None,
        weights=None,
        t=None,
        **kwargs,
    ):
        """
        训练损失函数
        """
        return self.scheduler.training_losses(
            model,
            x_start,
            x_pre,
            model_args,
            noise,
            weights,
            t,
            **kwargs,
        )

    def sample_debug(
        self,
        model,
        z,
        device,
        additional_args=None,
        progress=True,
    ):
        """
        调试版本，返回每一步的中间结果
        """
        timesteps = [(1.0 - i / self.num_sampling_steps) * self.num_timesteps 
                    for i in range(self.num_sampling_steps)]
        
        if self.use_discrete_timesteps:
            timesteps = [int(round(t)) for t in timesteps]
        
        timesteps = [torch.tensor([t] * z.shape[0], device=device) for t in timesteps]

        infos = []
        progress_wrap = tqdm if progress else (lambda x: x)
        
        for i, t in progress_wrap(enumerate(timesteps)):
            info = dict(i=i, t=t.cpu().item())
            
            # 模型预测
            if additional_args is not None:
                pred = model(z, t, **additional_args)
            else:
                pred = model(z, t)
            
            if isinstance(pred, tuple):
                v_pred = pred[0]
            else:
                v_pred = pred

            # 更新状态
            dt = timesteps[i] - timesteps[i + 1] if i < len(timesteps) - 1 else timesteps[i]
            dt = dt / self.num_timesteps
            z = z + v_pred * dt[:, None, None, None]

            info["z"] = z.clone()
            info["v_pred"] = v_pred.clone()
            infos.append(info)

        return infos