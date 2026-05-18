"""
阶段2: Tree-Ring 水印生成
在 'tree-ring' conda 环境中运行 (diffusers 0.11.1)
输入: 提示词、调制强度 w'
输出: 带水印的图像
"""
import argparse
import json
import os
import sys
import torch

# Tree-Ring 环境需要 diffusers 0.11.1
try:
    from diffusers import DDIMScheduler
    from tree_ring_watermark import InversableStableDiffusionPipeline
    TREERING_AVAILABLE = True
except ImportError:
    print("警告: Tree-Ring 或 diffusers 0.11.1 未安装")
    TREERING_AVAILABLE = False


def inject_tree_ring_watermark(prompt: str, intensity: float, seed: int,
                                config: dict) -> dict:
    """
    注入 Tree-Ring 频域水印

    Returns:
        {
            'image_path': 生成的图像路径,
            'latent_path': latent 保存路径,
            'mask_path': 水印掩码路径,
            'intensity_used': 实际使用的水印强度,
        }
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    work_dir = config.get('work_dir', '.')
    user_id = config.get('user_id', 'user_000')

    # 加载模型
    model_path = config.get('model_path',
                            '/root/autodl-tmp/models/stable-diffusion-2-1-base')

    if not TREERING_AVAILABLE:
        # 降级方案：生成占位图像
        from PIL import Image
        img = Image.new('RGB', (512, 512), color=(128, 128, 128))
        img_path = os.path.join(work_dir, f'treering_{user_id}.png')
        img.save(img_path)
        return {
            'image_path': img_path,
            'latent_path': None,
            'mask_path': None,
            'intensity_used': intensity,
            'note': 'Tree-Ring unavailable, using placeholder'
        }

    # 加载 InversableStableDiffusionPipeline
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
    ).to(device)

    # 设置 DDIM scheduler 用于 inversion
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # 设置随机种子
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 生成初始 latent
    latents = pipe.get_random_latents()

    # 注入水印
    from core.tree_ring import TreeRingWatermark
    tree_ring = TreeRingWatermark(
        config={
            'w_0': intensity,
            'w_radius': 10,
            'w_seed': 42,
        },
        device=device
    )

    watermark_mask = tree_ring.get_watermarking_mask(latents.shape)
    gt_patch = tree_ring.get_watermarking_pattern(latents.shape)
    latents_wm = tree_ring.inject_watermark(latents, watermark_mask, gt_patch, intensity)

    # 生成图像
    with torch.no_grad():
        image = pipe(
            prompt,
            latents=latents_wm,
            num_inference_steps=50,
            guidance_scale=7.5,
        ).images[0]

    # 保存结果
    safe_prompt = prompt.replace(' ', '_').replace('/', '_')[:30]
    img_path = os.path.join(work_dir, f'{safe_prompt}_{user_id}_treering.png')
    image.save(img_path)

    # 保存 latent 和 mask（用于验证）
    latent_path = os.path.join(work_dir, f'{safe_prompt}_{user_id}_latent.pt')
    mask_path = os.path.join(work_dir, f'{safe_prompt}_{user_id}_mask.pt')
    torch.save(latents_wm, latent_path)
    torch.save(watermark_mask, mask_path)

    return {
        'image_path': img_path,
        'latent_path': latent_path,
        'mask_path': mask_path,
        'intensity_used': intensity,
        'note': 'Tree-Ring watermark injected successfully'
    }


def main():
    parser = argparse.ArgumentParser(description='Tree-Ring 水印生成阶段')
    parser.add_argument('--input', type=str, required=True, help='输入JSON文件')
    parser.add_argument('--output', type=str, required=True, help='输出JSON文件')
    args = parser.parse_args()

    # 读取输入
    with open(args.input, 'r') as f:
        inputs = json.load(f)

    prompt = inputs.get('prompt', '')
    modulated_intensity = inputs.get('modulated_intensity', 0.05)
    seed = inputs.get('seed', 42)
    user_id = inputs.get('user_id', 'user_000')
    work_dir = inputs.get('work_dir', '.')

    # 配置
    config = {
        'model_path': '/root/autodl-tmp/models/stable-diffusion-2-1-base',
        'work_dir': work_dir,
        'user_id': user_id,
    }

    print(f"[阶段2] Tree-Ring 水印生成")
    print(f"  提示词: {prompt}")
    print(f"  调制强度 w': {modulated_intensity:.6f}")
    print(f"  种子: {seed}")

    # 运行生成
    result = inject_tree_ring_watermark(prompt, modulated_intensity, seed, config)

    print(f"  图像已保存: {result['image_path']}")
    print(f"  Latent 已保存: {result['latent_path']}")

    # 保存输出
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"  结果已保存到: {args.output}")


if __name__ == '__main__':
    main()
