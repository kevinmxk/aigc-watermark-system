"""
批量生成脚本
从提示词文件读取，批量生成带三层水印的图像
"""
import os
import sys
import json
import time
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.pipeline import ThreeLayerWatermarkPipeline


def batch_generate(config: dict,
                   prompts: list,
                   user_id_prefix: str = 'user_',
                   output_dir: str = './output',
                   start_seed: int = 42):
    """
    批量生成带三层水印的图像

    Args:
        config: 配置字典
        prompts: 提示词列表
        user_id_prefix: 用户 ID 前缀
        output_dir: 输出目录
        start_seed: 起始随机种子
    """
    os.makedirs(output_dir, exist_ok=True)
    pipeline = ThreeLayerWatermarkPipeline(config)

    print(f"\n{'='*60}")
    print(f"  批量生成: {len(prompts)} 张图像")
    print(f"  输出目录: {output_dir}")
    print(f"{'='*60}\n")

    all_results = []
    t_start = time.time()

    for i, prompt in enumerate(prompts):
        user_id = f"{user_id_prefix}{i:03d}"
        seed = start_seed + i

        print(f"[{i+1}/{len(prompts)}] '{prompt[:50]}...' (seed={seed})")

        result = pipeline.generate(
            prompt=prompt,
            user_id=user_id,
            seed=seed,
            output_dir=output_dir,
        )
        all_results.append(result)

    t_total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  批量生成完成!")
    print(f"  总耗时: {t_total:.1f}s, 平均: {t_total/len(prompts):.1f}s/张")
    print(f"  输出目录: {output_dir}")
    print(f"{'='*60}")

    # 保存批量元数据
    meta_path = os.path.join(output_dir, 'batch_meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n元数据已保存到: {meta_path}")

    return all_results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='批量生成带三层水印的图像')
    parser.add_argument('--prompts', type=str, required=True,
                        help='提示词文件路径（每行一个）')
    parser.add_argument('--output-dir', type=str, default='./output',
                        help='输出目录')
    parser.add_argument('--user-prefix', type=str, default='user_',
                        help='用户 ID 前缀')
    parser.add_argument('--seed', type=int, default=42, help='起始随机种子')
    parser.add_argument('--config', type=str, default=None, help='配置文件')

    args = parser.parse_args()

    # 加载配置
    config_path = args.config or os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'configs', 'default.yaml'
    )
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if args.output_dir:
        config['generation']['output_dir'] = args.output_dir

    # 读取提示词
    with open(args.prompts, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]

    batch_generate(
        config=config,
        prompts=prompts,
        user_id_prefix=args.user_prefix,
        output_dir=args.output_dir,
        start_seed=args.seed,
    )
