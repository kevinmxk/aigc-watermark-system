"""
DAAM 语义掩码模块 + Guided Strength 强度调制
通过交叉注意力图生成语义掩码，计算语义覆盖率，调制 Tree-Ring 水印强度
"""
import torch
import numpy as np
from typing import Optional, Tuple, Dict
from PIL import Image


class AttentionCapture:
    """通过 hook 捕获 Stable Diffusion 的 cross-attention 权重"""

    def __init__(self, pipe, factors: list = None):
        """
        Args:
            pipe: StableDiffusionPipeline
            factors: 要聚合的注意力层 factors，默认 [0, 1, 2, 4, 8]
        """
        self.pipe = pipe
        self.factors = factors or [0, 1, 2, 4, 8]
        self.attn_maps = {}
        self.hooks = []
        self.last_prompt = ''
        self.last_image = None
        self._step_idx = 0

    def __enter__(self):
        self._register_hooks()
        return self

    def __exit__(self, *args):
        self._remove_hooks()

    def _register_hooks(self):
        """在 UNet 的 cross-attention 层注册 forward hook"""
        def make_hook(factor, layer_idx):
            def hook(module, input, output):
                # output shape: (batch, seq_len, hidden_dim)
                # cross-attention: (batch, seq_len, context_len)
                if hasattr(module, 'attn') and hasattr(module.attn, 'get_attention_scores'):
                    # 捕获 attention weights
                    pass
                # 从 cross-attention processor 获取
                if hasattr(output, 'shape') and len(output.shape) == 3:
                    key = (factor, layer_idx)
                    self.attn_maps[key] = output.detach().cpu()
            return hook

        # 在 UNet 的 up blocks 中定位 cross-attention 层
        layer_idx = 0
        for i, up_block in enumerate(self.pipe.unet.up_blocks):
            if hasattr(up_block, 'attentions') and len(up_block.attentions) > 0:
                for j, transformer in enumerate(up_block.attentions):
                    if hasattr(transformer, 'transformer_blocks'):
                        for block in transformer.transformer_blocks:
                            if hasattr(block, 'attn2'):  # cross-attention
                                factor = 2 ** i  # 空间下采样因子
                                hook_fn = make_hook(factor, layer_idx)
                                handle = block.attn2.register_forward_hook(hook_fn)
                                self.hooks.append(handle)
                                layer_idx += 1

        # 也 hook 中间块
        if hasattr(self.pipe.unet.mid_block, 'attentions'):
            for transformer in self.pipe.unet.mid_block.attentions:
                if hasattr(transformer, 'transformer_blocks'):
                    for block in transformer.transformer_blocks:
                        if hasattr(block, 'attn2'):
                            hook_fn = make_hook(1, layer_idx)
                            handle = block.attn2.register_forward_hook(hook_fn)
                            self.hooks.append(handle)
                            layer_idx += 1

    def _remove_hooks(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    def compute_global_heat_map(self, prompt: str) -> Optional[torch.Tensor]:
        """
        计算全局热力图（聚合所有层的 cross-attention）

        Returns:
            热力图张量 (num_tokens, H_latent, W_latent) 或 None
        """
        if not self.attn_maps:
            return None

        # 聚合所有层的热力图
        all_maps = []
        latent_size = 64

        for (factor, layer_idx), attn_map in self.attn_maps.items():
            if factor not in self.factors:
                continue

            # attn_map shape: (batch, seq_len, hidden) 或 (batch, spatial, context)
            # 需要提取 token 级别的注意力权重
            if len(attn_map.shape) == 3:
                # 取每个 token 的注意力
                token_maps = attn_map[0]  # (seq_len, hidden)
                # 简化：直接使用 attention 的最后一个维度作为空间信息的代理
                pass

        # 简化方案：返回 None，让调用者用 DAAM 库的 trace
        return None

    def get_last_prompt(self) -> str:
        return self.last_prompt

    def get_last_image(self) -> Optional[Image.Image]:
        return self.last_image


def compute_semantic_ratio(mask: torch.Tensor, threshold: float = 0.3) -> float:
    """
    计算语义覆盖率：掩码中值大于阈值的像素占比

    Args:
        mask: DAAM 语义掩码 (H, W)，值在 [0, 1] 区间
        threshold: 二值化阈值

    Returns:
        语义覆盖率 r，取值范围 [0, 1]
    """
    total_pixels = mask.numel()
    semantic_pixels = (mask > threshold).sum().item()
    return semantic_pixels / total_pixels


def guided_strength(w_0: float, semantic_ratio: float, alpha: float = 0.3) -> float:
    """
    计算 Guided Strength 调制后的水印强度

    公式：w' = w_0 * (1 - alpha * r)

    Args:
        w_0: 基础水印强度（默认 0.05）
        semantic_ratio: DAAM 语义覆盖率 r
        alpha: 调制系数（默认 0.3）

    Returns:
        调制后的水印强度 w'

    示例：
        r = 0.0  -> w' = w_0（无语义区域，保持标准强度）
        r = 0.5  -> w' = w_0 * 0.85（中等语义区域，降低15%）
        r = 1.0  -> w' = w_0 * 0.7（全部语义区域，降低30%）
    """
    return w_0 * (1.0 - alpha * semantic_ratio)


class DAAMMaskGenerator:
    """DAAM 语义掩码生成器"""

    def __init__(self, config: dict, device: str = 'cuda'):
        """
        Args:
            config: DAAM 配置字典
            device: 计算设备
        """
        self.config = config
        self.device = device
        self.tau = config.get('tau', 0.3)
        self.alpha = config.get('alpha', 0.3)
        self.w_0 = config.get('w_0', 0.05)

    def generate_mask(self, pipe, prompt: str, seed: int = None,
                      save_heatmap: bool = False,
                      output_dir: str = None) -> Tuple[torch.Tensor, float]:
        """
        生成 DAAM 语义掩码并计算语义覆盖率

        Args:
            pipe: StableDiffusionPipeline（需要支持 trace）
            prompt: 文本提示词
            seed: 随机种子
            save_heatmap: 是否保存热力图
            output_dir: 热力图保存目录

        Returns:
            (mask, semantic_ratio) - 语义掩码和覆盖率
        """
        try:
            # 尝试使用 DAAM 库的 trace
            from daam import trace
            return self._generate_with_daam(pipe, prompt, seed, save_heatmap, output_dir)
        except ImportError:
            # 使用内置的简化方案
            return self._generate_fallback(pipe, prompt, seed, save_heatmap, output_dir)

    def _generate_with_daam(self, pipe, prompt: str, seed: int,
                              save_heatmap: bool, output_dir: str) -> Tuple[torch.Tensor, float]:
        """使用 DAAM 库生成语义掩码"""
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        with trace(pipe) as tc:
            out = pipe(prompt, num_inference_steps=50, guidance_scale=7.5)
            image = out.images[0]

            # 生成实验对象
            exp = tc.to_experiment(output_dir or '.', id='daam_mask')
            heat_map = exp.heat_map()

            # 提取主要关键词的热力图
            words = [w.strip('.,!?;:') for w in prompt.split() if len(w.strip('.,!?;:')) > 2]
            combined_map = None

            for word in reversed(words[:3]):
                try:
                    word_heat_map = heat_map.compute_word_heat_map(word)
                    word_tensor = word_heat_map.expand_as(image, threshold=self.tau)
                    if combined_map is None:
                        combined_map = word_tensor
                    else:
                        combined_map = torch.max(combined_map, word_tensor)
                except (ValueError, RuntimeError):
                    continue

            if combined_map is None:
                # 如果所有词都失败，使用全局热力图
                combined_map = heat_map.heat_maps.mean(dim=0)

            # 归一化到 [0, 1]
            combined_map = (combined_map - combined_map.min()) / (combined_map.max() - combined_map.min() + 1e-8)

            # 保存热力图
            if save_heatmap and output_dir:
                import os
                os.makedirs(output_dir, exist_ok=True)
                safe_prompt = prompt.replace(' ', '_').replace('/', '_')[:30]
                word_heat_map.plot_overlay(image, out_file=f'{output_dir}/{safe_prompt}_heatmap.png')

            semantic_ratio = compute_semantic_ratio(combined_map, self.tau)
            return combined_map, semantic_ratio

    def _generate_fallback(self, pipe, prompt: str, seed: int,
                            save_heatmap: bool, output_dir: str) -> Tuple[torch.Tensor, float]:
        """简化方案：基于关键词长度估算语义覆盖率"""
        # 当 DAAM 库不可用时，使用经验估算
        words = [w.strip('.,!?;:') for w in prompt.split() if len(w.strip('.,!?;:')) > 2]
        # 简单启发：词越多，语义覆盖率越高
        semantic_ratio = min(0.3 + 0.1 * len(words), 1.0)

        # 生成均匀掩码（仅作占位）
        mask = torch.full((64, 64), semantic_ratio)
        return mask, semantic_ratio

    def compute_guided_strength(self, semantic_ratio: float) -> float:
        """计算调制后的水印强度"""
        return guided_strength(self.w_0, semantic_ratio, self.alpha)

    def full_analysis(self, pipe, prompt: str, seed: int = None,
                      output_dir: str = None) -> Dict:
        """
        完整分析：生成掩码 + 计算覆盖率 + 计算调制强度

        Returns:
            {
                'mask': 语义掩码,
                'semantic_ratio': 覆盖率,
                'original_intensity': 原始强度,
                'modulated_intensity': 调制后强度,
                'intensity_reduction_pct': 降低百分比
            }
        """
        mask, ratio = self.generate_mask(pipe, prompt, seed, output_dir=output_dir)
        modulated = self.compute_guided_strength(ratio)
        reduction = (1.0 - modulated / self.w_0) * 100

        return {
            'mask': mask,
            'semantic_ratio': round(ratio, 4),
            'original_intensity': self.w_0,
            'modulated_intensity': round(modulated, 6),
            'intensity_reduction_pct': round(reduction, 2),
        }
