"""
三级水印主管线 - 端到端生成与检测
跨域嵌套（Tree-Ring + DAAM） + 跨域叠加（StegaStamp）
"""
import os
import json
import time
import sys
from typing import Optional, Dict
from PIL import Image
import numpy as np
import torch
import argparse

from .tree_ring import TreeRingWatermark
from .daam_mask import DAAMMaskGenerator
from .stegastamp import StegaStampEncoder, StegaStampDecoder
from ..utils.metrics import MetricsCalculator


class ThreeLayerWatermarkPipeline:
    """
    三级水印生成管线

    流程：
    1. DAAM 语义分析 -> 语义覆盖率 r
    2. Guided Strength 计算 -> 调制强度 w' = w_0 * (1 - alpha * r)
    3. Tree-Ring 频域水印注入 -> 生成带两层嵌套水印的图像
    4. StegaStamp 像素域编码 -> 叠加用户指纹

    使用方式：
        pipeline = ThreeLayerWatermarkPipeline(config, pipe=sd_pipeline)
        result = pipeline.generate(prompt="...", user_id="user_001")
    """

    def __init__(self, config: dict, pipe=None, tree_ring_pipe=None, daam_pipe=None):
        """
        Args:
            config: 完整配置字典
            pipe: Stable Diffusion Pipeline（用于 Tree-Ring + DAAM 共用）
            tree_ring_pipe: Tree-Ring 专用 Pipeline（InversableStableDiffusionPipeline）
            daam_pipe: DAAM 专用 Pipeline（支持 trace）
        """
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.pipe = pipe
        self.tree_ring_pipe = tree_ring_pipe
        self.daam_pipe = daam_pipe

        # 初始化各模块
        tr_config = config.get('tree_ring', {})
        self.tree_ring = TreeRingWatermark(tr_config, self.device)
        self.daam = DAAMMaskGenerator(config.get('daam', tr_config), self.device)

        # StegaStamp（延迟加载）
        ss_config = config.get('stegastamp', {})
        self.ss_encoder = None
        self.ss_decoder = None
        if ss_config.get('encoder_path'):
            self.ss_encoder = StegaStampEncoder(
                encoder_path=ss_config['encoder_path'],
                secret_size=ss_config.get('secret_size', 100),
            )
        if ss_config.get('decoder_path'):
            self.ss_decoder = StegaStampDecoder(
                decoder_path=ss_config['decoder_path'],
                secret_size=ss_config.get('secret_size', 100),
            )

        self.metrics = MetricsCalculator()
        self.user_manager = None  # 延迟初始化

    def generate(self, prompt: str, user_id: str, seed: int = None,
                 output_dir: str = None, save_comparison: bool = False) -> Dict:
        """
        端到端生成：输入提示词 + 用户ID -> 输出带三层水印的图像

        Args:
            prompt: 文本提示词
            user_id: 用户标识字符串
            seed: 随机种子
            output_dir: 输出目录
            save_comparison: 是否保存无水印对比图

        Returns:
            包含生成结果和指标的字典
        """
        t0 = time.time()
        if seed is None:
            seed = self.config.get('generation', {}).get('seed', 42)

        if output_dir is None:
            output_dir = self.config.get('generation', {}).get('output_dir', './output')
        os.makedirs(output_dir, exist_ok=True)

        results = {
            'prompt': prompt,
            'user_id': user_id,
            'seed': seed,
        }

        # ===== 第 1 步：DAAM 语义分析 =====
        print(f"\n[1/4] DAAM 语义分析: '{prompt[:50]}...'")
        daam_result = self.daam.full_analysis(
            pipe=self.daam_pipe or self.pipe,
            prompt=prompt,
            seed=seed,
            output_dir=output_dir,
        )
        semantic_ratio = daam_result['semantic_ratio']
        modulated_intensity = daam_result['modulated_intensity']
        results['semantic_ratio'] = semantic_ratio
        results['modulated_intensity'] = modulated_intensity
        results['daam'] = {
            'semantic_ratio': semantic_ratio,
            'modulated_intensity': modulated_intensity,
            'intensity_reduction_pct': daam_result['intensity_reduction_pct'],
        }
        print(f"  语义覆盖率 r = {semantic_ratio:.4f}")
        print(f"  调制强度 w' = {modulated_intensity:.6f} (降低 {daam_result['intensity_reduction_pct']:.1f}%)")

        # ===== 第 2-3 步：Tree-Ring 水印注入 + 图像生成 =====
        print(f"\n[2/4] Tree-Ring 频域水印注入 + 图像生成...")
        gen_kwargs = {
            'num_images_per_prompt': 1,
            'guidance_scale': self.config.get('model', {}).get('guidance_scale', 7.5),
            'num_inference_steps': self.config.get('model', {}).get('num_inference_steps', 50),
        }

        latents_wm, image_tr, tr_mask = self.tree_ring.inject_and_generate(
            pipe=self.tree_ring_pipe or self.pipe,
            prompt=prompt,
            intensity=modulated_intensity,
            seed=seed,
            **gen_kwargs,
        )

        # 同时生成无水印版本（用于对比）
        print(f"  生成无水印对比图...")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with torch.no_grad():
            image_clean = (self.tree_ring_pipe or self.pipe)(
                prompt,
                **gen_kwargs,
            ).images[0]

        # 计算图像质量指标
        tr_metrics = self.metrics.compute_all(image_clean, image_tr)
        results['tree_ring_metrics'] = tr_metrics
        print(f"  PSNR = {tr_metrics['psnr']:.2f} dB")
        print(f"  SSIM = {tr_metrics['ssim']:.4f}")

        # 保存 Tree-Ring 图像
        safe_prompt = prompt.replace(' ', '_').replace('/', '_')[:40]
        tr_path = os.path.join(output_dir, f"{safe_prompt}_{user_id}_treering.png")
        image_tr.save(tr_path)

        if save_comparison:
            clean_path = os.path.join(output_dir, f"{safe_prompt}_{user_id}_clean.png")
            image_clean.save(clean_path)
            results['clean_image_path'] = clean_path

        # ===== 第 4 步：StegaStamp 用户指纹编码 =====
        print(f"\n[3/4] StegaStamp 用户指纹编码 (user_id={user_id})...")
        if self.ss_encoder is None:
            print("  警告: StegaStamp encoder 未配置，跳过用户层水印")
            final_image = image_tr
            results['stegastamp_metrics'] = {'skipped': True}
        else:
            from ..utils.user_manager import UserIDManager
            if self.user_manager is None:
                self.user_manager = UserIDManager(
                    secret_size=self.config.get('stegastamp', {}).get('secret_size', 100)
                )
            user_secret = self.user_manager.encode(user_id)
            final_image, enc_array = self.ss_encoder.encode(image_tr, user_secret)

            # StegaStamp 指标
            ss_psnr = self.metrics.compute_psnr(image_tr, final_image)
            ss_ssim = self.metrics.compute_ssim(image_tr, final_image)
            results['stegastamp_metrics'] = {
                'psnr': round(ss_psnr, 4),
                'ssim': round(ss_ssim, 6),
                'user_id': user_id,
            }
            print(f"  StegaStamp PSNR = {ss_psnr:.2f} dB")
            print(f"  StegaStamp SSIM = {ss_ssim:.4f}")

        # ===== 保存最终结果 =====
        print(f"\n[4/4] 保存结果...")
        final_path = os.path.join(output_dir, f"{safe_prompt}_{user_id}.png")
        final_image.save(final_path)
        results['output_path'] = final_path

        # 最终图像质量（原始 vs 最终）
        final_metrics = self.metrics.compute_all(image_clean, final_image)
        results['final_metrics'] = final_metrics

        # 保存 metadata
        meta_path = final_path.replace('.png', '_meta.json')
        results['total_time_sec'] = round(time.time() - t0, 2)

        # 移除不能序列化的对象
        serializable = {k: v for k, v in results.items() if k not in ('daam',)}
        serializable['daam'] = results.get('daam', {})
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False, default=str)

        print(f"\n{'='*50}")
        print(f"生成完成！总耗时: {results['total_time_sec']:.1f}s")
        print(f"最终输出: {final_path}")
        print(f"元数据: {meta_path}")
        print(f"{'='*50}")

        return results

    def close(self):
        """释放资源"""
        if self.ss_encoder:
            self.ss_encoder.close()
        if self.ss_decoder:
            self.ss_decoder.close()

    def __del__(self):
        self.close()


class WatermarkDetector:
    """水印检测工具"""

    def __init__(self, config: dict):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        ss_config = config.get('stegastamp', {})
        self.ss_decoder = None
        if ss_config.get('decoder_path'):
            self.ss_decoder = StegaStampDecoder(
                decoder_path=ss_config['decoder_path'],
                secret_size=ss_config.get('secret_size', 100),
            )

    def detect(self, image_path: str, user_id: str = None,
               baseline_image_path: str = None) -> Dict:
        """
        检测图像中的水印

        Args:
            image_path: 待检测图像路径
            user_id: 期望的用户 ID
            baseline_image_path: 无水印基线图像路径

        Returns:
            检测结果字典
        """
        image = Image.open(image_path).convert('RGB')
        results = {}

        # Tree-Ring 检测
        baseline_energy = None
        if baseline_image_path:
            baseline = Image.open(baseline_image_path).convert('L')
            baseline_energy = self.metrics.compute_fft_ring_energy(baseline)[0]

        tr_result = self._detect_tree_ring(image, baseline_energy)
        results['tree_ring'] = tr_result

        # StegaStamp 检测
        if self.ss_decoder is not None:
            ss_result = self._detect_stegastamp(image, user_id)
            results['stegastamp'] = ss_result

        # 综合判断
        tr_ok = results['tree_ring'].get('detected', False)
        ss_ok = results.get('stegastamp', {}).get('bit_accuracy', 0) > 0.95
        results['overall'] = tr_ok and (not results.get('stegastamp') or ss_ok)

        return results

    @property
    def metrics(self):
        from ..utils.metrics import MetricsCalculator
        return MetricsCalculator()

    def _detect_tree_ring(self, image: Image.Image,
                           baseline_energy: float = None) -> Dict:
        from ..utils.metrics import detect_tree_ring
        return detect_tree_ring(image, baseline_energy)

    def _detect_stegastamp(self, image: Image.Image,
                            user_id: str = None) -> Dict:
        if self.ss_decoder is None:
            return {'error': 'StegaStamp decoder not loaded'}

        if user_id is not None:
            from ..utils.user_manager import UserIDManager
            um = UserIDManager()
            expected_secret = um.encode(user_id)
            return self.ss_decoder.verify(image, expected_secret)

        decoded = self.ss_decoder.decode(image)
        return {
            'mean_bit_value': float(np.mean(np.round(decoded))),
        }
