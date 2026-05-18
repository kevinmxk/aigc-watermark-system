"""
Tree-Ring 频域水印模块
基于傅里叶变换在 latent 空间注入环形频域水印
"""
import torch
import numpy as np
import copy
import argparse
from typing import Tuple, Optional, Dict
from PIL import Image


def circle_mask(size: int = 64, r: int = 10, x_offset: int = 0, y_offset: int = 0) -> np.ndarray:
    """生成环形二值掩码"""
    x0 = y0 = size // 2
    x0 += x_offset
    y0 += y_offset
    y, x = np.ogrid[:size, :size]
    y = y[::-1]
    return ((x - x0) ** 2 + (y - y0) ** 2) <= r ** 2


class TreeRingWatermark:
    """Tree-Ring 频域水印注入与检测"""

    def __init__(self, config: dict, device: str = 'cuda'):
        """
        Args:
            config: Tree-Ring 配置字典
            device: 计算设备
        """
        self.config = config
        self.device = device
        self.w_0 = config.get('w_0', 0.05)
        self.w_radius = config.get('w_radius', 10)
        self.w_seed = config.get('w_seed', 42)
        self.w_channel = config.get('w_channel', -1)

    def get_watermarking_mask(self, latent_shape: torch.Size) -> torch.Tensor:
        """
        生成 Tree-Ring 环形频域掩码

        Args:
            latent_shape: latent 张量形状 (B, C, H, W)

        Returns:
            布尔掩码张量
        """
        mask = torch.zeros(latent_shape, dtype=torch.bool, device=self.device)

        np_mask = circle_mask(latent_shape[-1], r=self.w_radius)
        torch_mask = torch.tensor(np_mask, device=self.device)

        if self.w_channel == -1:
            mask[:, :] = torch_mask
        else:
            mask[:, self.w_channel] = torch_mask

        return mask

    def get_watermarking_pattern(self, latent_shape: torch.Size) -> torch.Tensor:
        """
        生成频域水印图案（固定图案，由 seed 决定）

        Args:
            latent_shape: latent 张量形状

        Returns:
            复数水印图案张量
        """
        # 用固定 seed 生成随机 latent
        rng = torch.Generator(device=self.device)
        rng.manual_seed(self.w_seed)
        gt_init = torch.randn(*latent_shape, generator=rng, device=self.device)

        # 傅里叶变换
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))

        # 环形图案：对每个半径环，填充为该环上某一点的复数值
        gt_patch_tmp = copy.deepcopy(gt_patch)
        for i in range(self.w_radius, 0, -1):
            tmp_mask = circle_mask(gt_init.shape[-1], r=i)
            tmp_mask = torch.tensor(tmp_mask, device=self.device)
            for j in range(gt_patch.shape[1]):
                gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()

        return gt_patch

    def inject_watermark(self, latents: torch.Tensor,
                          watermarking_mask: torch.Tensor,
                          gt_patch: torch.Tensor,
                          intensity: float = None) -> torch.Tensor:
        """
        在 latent 空间注入 Tree-Ring 频域水印

        Args:
            latents: 初始噪声 latent (B, C, H, W)
            watermarking_mask: 环形频域掩码
            gt_patch: 水印图案
            intensity: 水印强度（None 时使用 w_0）

        Returns:
            注入水印后的 latent
        """
        if intensity is None:
            intensity = self.w_0

        latents_fft = torch.fft.fftshift(torch.fft.fft2(latents), dim=(-1, -2))
        latents_fft[watermarking_mask] += intensity * gt_patch[watermarking_mask]
        latents_wm = torch.fft.ifft2(torch.fft.ifftshift(latents_fft, dim=(-1, -2))).real

        return latents_wm

    def inject_and_generate(self, pipe, prompt: str, intensity: float = None,
                             seed: int = None, **gen_kwargs) -> Tuple[torch.Tensor, Image.Image, torch.Tensor]:
        """
        注入水印并生成图像（完整版：返回 latents 和图像）

        Args:
            pipe: InversableStableDiffusionPipeline
            prompt: 文本提示词
            intensity: 水印强度
            seed: 随机种子
            gen_kwargs: 其他生成参数

        Returns:
            (latent_wm, image_wm, watermarking_mask)
        """
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        latent_shape = (1, 4, 64, 64)
        latents = pipe.get_random_latents()

        # 生成水印掩码和图案
        watermarking_mask = self.get_watermarking_mask(latent_shape)
        gt_patch = self.get_watermarking_pattern(latent_shape)

        # 注入水印
        latents_wm = self.inject_watermark(latents.clone(), watermarking_mask, gt_patch, intensity)

        # 生成图像
        with torch.no_grad():
            image_wm = pipe(
                prompt,
                latents=latents_wm,
                **gen_kwargs
            ).images[0]

        return latents_wm, image_wm, watermarking_mask

    def verify_watermark(self, pipe, image: Image.Image,
                          watermarking_mask: torch.Tensor,
                          gt_patch: torch.Tensor) -> Dict:
        """
        验证图像是否包含 Tree-Ring 水印

        Args:
            pipe: InversableStableDiffusionPipeline
            image: 待检测图像
            watermarking_mask: 水印掩码
            gt_patch: 水印图案

        Returns:
            {'detected': bool, 'metric': float}
        """
        # 从图像反推 latent（DDIM inversion 简化版）
        img_np = np.array(image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(self.device)
        mean = torch.tensor([0.5, 0.5, 0.5], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5], device=self.device).view(1, 3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        reversed_latents = pipe.get_image_latents(img_tensor, sample=False)

        # 计算水印指标
        reversed_latents_fft = torch.fft.fftshift(
            torch.fft.fft2(reversed_latents), dim=(-1, -2)
        )
        metric = torch.abs(
            reversed_latents_fft[watermarking_mask] - gt_patch[watermarking_mask]
        ).mean().item()

        # 指标越低 = 水印越强（经验阈值）
        detected = metric < 50.0

        return {
            'detected': detected,
            'metric': round(metric, 4),
        }
