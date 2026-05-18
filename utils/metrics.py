"""
图像质量指标计算 - PSNR, SSIM, FFT 环形能量分析
纯 numpy 实现，不依赖 skimage
"""
import numpy as np
from PIL import Image
from typing import Tuple, Dict


def compute_psnr(img1: Image.Image, img2: Image.Image) -> float:
    """计算两张图像的 PSNR (Peak Signal-to-Noise Ratio)"""
    arr1 = np.array(img1).astype(np.float64)
    arr2 = np.array(img2).astype(np.float64)
    mse = np.mean((arr1 - arr2) ** 2)
    if mse < 1e-10:
        return float('inf')
    max_val = 255.0
    return float(10 * np.log10(max_val ** 2 / mse))


def compute_ssim(img1: Image.Image, img2: Image.Image,
               window_size: int = 11, C1: float = 6.5025, C2: float = 58.5225) -> float:
    """计算 SSIM (Structural Similarity Index) - 灰度版本"""
    arr1 = np.array(img1.convert('L')).astype(np.float64)
    arr2 = np.array(img2.convert('L')).astype(np.float64)

    kernel = np.ones((window_size, window_size), dtype=np.float64) / (window_size ** 2)

    mu1 = _convolve2d(arr1, kernel)
    mu2 = _convolve2d(arr2, kernel)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = _convolve2d(arr1 ** 2, kernel) - mu1_sq
    sigma2_sq = _convolve2d(arr2 ** 2, kernel) - mu2_sq
    sigma12 = _convolve2d(arr1 * arr2, kernel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return float(np.mean(ssim_map))


def _convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2D 卷积（简化版，用于 SSIM 计算）"""
    from scipy.ndimage import convolve
    return convolve(image, kernel, mode='constant', cval=0.0)


def compute_fft_ring_energy(image: Image.Image,
                             inner_radius: int = 80,
                             outer_radius: int = 120) -> Tuple[float, np.ndarray]:
    """
    计算图像灰度图傅里叶变换在环形区域的能量
    用于 Tree-Ring 水印检测

    Returns:
        (ring_energy, fft_2d)
    """
    gray = np.array(image.convert('L')).astype(np.float64)
    fft = np.fft.fft2(gray)
    fft_shifted = np.fft.fftshift(fft)

    h, w = gray.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    ring_mask = (dist >= inner_radius) & (dist <= outer_radius)

    ring_energy = float(np.mean(np.abs(fft_shifted[ring_mask])))
    return ring_energy, fft_shifted


def detect_tree_ring(image: Image.Image,
                     baseline_energy: float = None,
                     inner_radius: int = 80,
                     outer_radius: int = 120,
                     threshold_ratio: float = 0.5) -> Dict:
    """
    检测图像是否包含 Tree-Ring 水印

    Args:
        image: 待检测图像
        baseline_energy: 基准能量（无水印图像），若为 None 则使用默认阈值
        inner_radius: 环形区域内半径（512x512 图像对应 8-12 latent pixel）
        outer_radius: 环形区域外半径
        threshold_ratio: 检测阈值比率

    Returns:
        {'detected': bool, 'ring_energy': float, 'ratio': float}
    """
    ring_energy, _ = compute_fft_ring_energy(image, inner_radius, outer_radius)

    if baseline_energy is not None and baseline_energy > 0:
        ratio = ring_energy / baseline_energy
        detected = ratio > threshold_ratio
    else:
        # 无基线时使用绝对能量阈值（经验值）
        detected = ring_energy > 1000.0
        ratio = None

    return {
        'detected': detected,
        'ring_energy': round(ring_energy, 4),
        'ratio': round(ratio, 4) if ratio is not None else None,
    }


def compute_pixel_diff_pct(img1: Image.Image, img2: Image.Image) -> float:
    """计算两张图像的平均像素差异百分比"""
    arr1 = np.array(img1).astype(np.float64)
    arr2 = np.array(img2).astype(np.float64)
    diff = np.abs(arr1 - arr2)
    return float(np.mean(diff) / 255.0 * 100)


class MetricsCalculator:
    """批量图像质量指标计算器"""

    def __init__(self):
        self.results = []

    def compute_all(self, original: Image.Image, watermarked: Image.Image) -> Dict:
        """计算所有指标"""
        return {
            'psnr': round(compute_psnr(original, watermarked), 4),
            'ssim': round(compute_ssim(original, watermarked), 6),
            'pixel_diff_pct': round(compute_pixel_diff_pct(original, watermarked), 4),
            'fft_ring_energy': round(compute_fft_ring_energy(watermarked)[0], 4),
        }

    def batch_compute(self, pairs: list) -> list:
        """批量计算 [(original, watermarked), ...]"""
        results = []
        for orig, wm in pairs:
            metrics = self.compute_all(orig, wm)
            results.append(metrics)
        return results
