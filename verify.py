"""
系统快速验证脚本
测试所有核心模块是否能正确导入
"""
import sys
import os

print("=" * 60)
print("  AIGC 三级版权溯源系统 - 快速验证")
print("=" * 60)
print()

# 添加项目路径
sys.path.insert(0, os.path.dirname(__file__))

errors = []

# 测试导入 utils
try:
    from utils.metrics import MetricsCalculator, compute_psnr, compute_ssim
    from utils.user_manager import UserIDManager
    print("[OK] utils 模块导入成功")
except Exception as e:
    errors.append(f"utils: {e}")
    print(f"[FAIL] utils: {e}")

# 测试导入 core
try:
    from core.tree_ring import TreeRingWatermark
    from core.daam_mask import DAAMMaskGenerator, guided_strength
    from core.stegastamp import StegaStampEncoder, StegaStampDecoder
    from core.pipeline import ThreeLayerWatermarkPipeline, WatermarkDetector
    print("[OK] core 模块导入成功")
except Exception as e:
    errors.append(f"core: {e}")
    print(f"[FAIL] core: {e}")

# 测试配置加载
try:
    import yaml
    config_path = os.path.join(os.path.dirname(__file__), 'configs', 'default.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print(f"[OK] 配置加载成功 (secret_size={config['stegastamp']['secret_size']})")
except Exception as e:
    errors.append(f"config: {e}")
    print(f"[FAIL] config: {e}")

# 测试 UserIDManager
try:
    um = UserIDManager(secret_size=100)
    user_id = "user_001"
    bits = um.encode(user_id)
    assert len(bits) == 100, "比特长度应为100"
    print(f"[OK] UserIDManager 测试通过 (user_id={user_id} -> {len(bits)}bits)")
except Exception as e:
    errors.append(f"user_manager: {e}")
    print(f"[FAIL] user_manager: {e}")

# 测试 DAAM Guided Strength
try:
    w_0 = 0.05
    r = 0.5
    alpha = 0.3
    w_prime = guided_strength(w_0, r, alpha)
    expected = 0.05 * (1 - 0.3 * 0.5)  # 0.0425
    assert abs(w_prime - expected) < 0.0001, f"计算错误: {w_prime} != {expected}"
    print(f"[OK] Guided Strength 公式测试通过 (w'={w_prime:.6f})")
except Exception as e:
    errors.append(f"guided_strength: {e}")
    print(f"[FAIL] guided_strength: {e}")

# 测试 Tree-Ring
try:
    import torch
    tr = TreeRingWatermark(
        config={
            'w_0': 0.05,
            'w_radius': 10,
            'w_seed': 42
        },
        device='cpu'
    )
    latents = torch.randn(1, 4, 64, 64)
    mask = tr.get_watermarking_mask(latents.shape)
    pattern = tr.get_watermarking_pattern(latents.shape)
    assert mask.shape == latents.shape, "掩码形状错误"
    print(f"[OK] Tree-Ring 测试通过 (shape={latents.shape}, mask={mask.shape})")
except Exception as e:
    errors.append(f"tree_ring: {e}")
    print(f"[FAIL] tree_ring: {e}")

print()
print("=" * 60)
if not errors:
    print("  所有测试通过！系统准备就绪。")
else:
    print(f"  发现 {len(errors)} 个错误，请检查上述输出。")
print("=" * 60)
