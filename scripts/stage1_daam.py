"""
阶段1: DAAM 语义分析
在 'daam' conda 环境中运行 (diffusers 0.21.2)
输出: 语义覆盖率 r 和调制强度 w'
"""
import argparse
import json
import os
import sys
import torch

# DAAM 环境需要 diffusers 0.21.2
try:
    from diffusers import StableDiffusionPipeline
    import daam
    from daam import trace
    DAAM_AVAILABLE = True
except ImportError:
    print("警告: DAAM 或 diffusers 0.21.2 未安装")
    DAAM_AVAILABLE = False


def compute_semantic_ratio(heat_map, threshold: float = 0.3) -> float:
    """计算语义覆盖率"""
    if isinstance(heat_map, torch.Tensor):
        total_pixels = heat_map.numel()
        semantic_pixels = (heat_map > threshold).sum().item()
    else:
        total_pixels = heat_map.size
        semantic_pixels = (heat_map > threshold).sum()
    return semantic_pixels / total_pixels


def guided_strength(w_0: float, semantic_ratio: float, alpha: float = 0.3) -> float:
    """计算调制后的水印强度: w' = w_0 * (1 - alpha * r)"""
    return w_0 * (1.0 - alpha * semantic_ratio)


def run_daam_analysis(prompt: str, seed: int, config: dict) -> dict:
    """
    运行 DAAM 语义分析

    Returns:
        {
            'semantic_ratio': 语义覆盖率 r,
            'modulated_intensity': 调制后的水印强度 w',
            'original_intensity': 原始强度 w_0,
            'intensity_reduction_pct': 强度降低百分比,
            'mask_path': 语义掩码保存路径,
        }
    """
    if not DAAM_AVAILABLE:
        # 降级方案：返回默认值
        print("DAAM 不可用，使用默认语义覆盖率 0.3")
        w_0 = config.get('w_0', 0.05)
        alpha = config.get('alpha', 0.3)
        semantic_ratio = 0.3  # 默认中等覆盖率
        modulated = guided_strength(w_0, semantic_ratio, alpha)

        return {
            'semantic_ratio': semantic_ratio,
            'modulated_intensity': round(modulated, 6),
            'original_intensity': w_0,
            'intensity_reduction_pct': round((1 - modulated / w_0) * 100, 2),
            'mask_path': None,
            'note': 'DAAM unavailable, using default ratio'
        }

    # DAAM 分析流程
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载模型
    model_path = config.get('model_path', 'stabilityai/stable-diffusion-2-1-base')
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
    ).to(device)

    # 设置随机种子
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 使用 DAAM trace 捕获注意力
    with trace(pipe) as tc:
        out = pipe(prompt, num_inference_steps=50, guidance_scale=7.5)
        image = out.images[0]

        # 生成热力图
        exp = tc.to_experiment('.', id='daam_analysis')
        heat_map = exp.heat_map()

        # 提取关键词的热力图
        words = [w.strip('.,!?;:') for w in prompt.split() if len(w.strip('.,!?;:')) > 2]
        combined_map = None

        for word in reversed(words[:3]):  # 取前3个关键词
            try:
                word_heat_map = heat_map.compute_word_heat_map(word)
                word_tensor = word_heat_map.expand_as(image, threshold=config.get('tau', 0.3))
                if combined_map is None:
                    combined_map = word_tensor
                else:
                    combined_map = torch.max(combined_map, word_tensor)
            except (ValueError, RuntimeError):
                continue

        if combined_map is None:
            combined_map = heat_map.heat_maps.mean(dim=0)

        # 归一化
        combined_map = (combined_map - combined_map.min()) / (combined_map.max() - combined_map.min() + 1e-8)

        # 计算语义覆盖率
        semantic_ratio = compute_semantic_ratio(combined_map, config.get('tau', 0.3))

    # 计算调制强度
    w_0 = config.get('w_0', 0.05)
    alpha = config.get('alpha', 0.3)
    modulated = guided_strength(w_0, semantic_ratio, alpha)

    # 保存掩码
    mask_path = os.path.join(config.get('work_dir', '.'), 'daam_mask.pt')
    torch.save(combined_map, mask_path)

    return {
        'semantic_ratio': round(semantic_ratio, 4),
        'modulated_intensity': round(modulated, 6),
        'original_intensity': w_0,
        'intensity_reduction_pct': round((1 - modulated / w_0) * 100, 2),
        'mask_path': mask_path,
    }


def main():
    parser = argparse.ArgumentParser(description='DAAM 语义分析阶段')
    parser.add_argument('--input', type=str, required=True, help='输入JSON文件')
    parser.add_argument('--output', type=str, required=True, help='输出JSON文件')
    args = parser.parse_args()

    # 读取输入
    with open(args.input, 'r') as f:
        inputs = json.load(f)

    prompt = inputs.get('prompt', '')
    seed = inputs.get('seed', 42)
    work_dir = inputs.get('work_dir', '.')

    # 配置
    config = {
        'w_0': 0.05,
        'alpha': 0.3,
        'tau': 0.3,
        'work_dir': work_dir,
    }

    print(f"[阶段1] DAAM 语义分析")
    print(f"  提示词: {prompt}")
    print(f"  种子: {seed}")

    # 运行分析
    result = run_daam_analysis(prompt, seed, config)

    print(f"  语义覆盖率 r: {result['semantic_ratio']:.4f}")
    print(f"  调制强度 w': {result['modulated_intensity']:.6f}")
    print(f"  强度降低: {result['intensity_reduction_pct']:.1f}%")

    # 保存输出
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"  结果已保存到: {args.output}")


if __name__ == '__main__':
    main()
