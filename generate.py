"""
AIGC 三级版权溯源系统 - 主入口
用户输入提示词 + 用户ID，自动生成带三层水印的图像
"""
import argparse
import os
import sys
import yaml
import json
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(__file__))

from core.pipeline import ThreeLayerWatermarkPipeline


def load_config(config_path: str = None) -> dict:
    """加载配置文件"""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'configs', 'default.yaml')

    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description='AIGC 三级版权溯源系统 - 端到端水印生成',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 生成带水印的图像
  python generate.py --prompt "a cat sitting on a windowsill" --user-id user_001

  # 生成对比图（有/无水印）
  python generate.py --prompt "a beautiful sunset" --user-id user_002 --save-comparison

  # 批量生成（从文件读取提示词）
  python generate.py --prompts prompts.txt --user-id-prefix brand_ --output-dir ./results

  # 指定自定义配置
  python generate.py --prompt "..." --user-id user_001 --config my_config.yaml
        """
    )

    # 必需参数
    parser.add_argument('--prompt', type=str, help='文本提示词')
    parser.add_argument('--prompts', type=str, help='提示词文件路径（每行一个提示词）')

    # 用户标识
    parser.add_argument('--user-id', type=str, help='用户标识（如 user_001）')
    parser.add_argument('--user-id-prefix', type=str, default='user_',
                        help='批量生成时的用户 ID 前缀')

    # 生成参数
    parser.add_argument('--seed', type=int, default=None, help='随机种子')
    parser.add_argument('--output-dir', type=str, default=None, help='输出目录')
    parser.add_argument('--save-comparison', action='store_true',
                        help='是否保存无水印对比图')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')

    # 模型参数
    parser.add_argument('--model-path', type=str, default=None,
                        help='模型路径（覆盖配置文件中的设置）')
    parser.add_argument('--w-0', type=float, default=None, help='基础水印强度 w_0')
    parser.add_argument('--alpha', type=float, default=None, help='Guided Strength 调制系数')
    parser.add_argument('--steagstamp-encoder', type=str, default=None,
                        help='StegaStamp Encoder SavedModel 路径')
    parser.add_argument('--steagstamp-decoder', type=str, default=None,
                        help='StegaStamp Decoder SavedModel 路径')

    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 命令行参数覆盖配置文件
    if args.model_path:
        config['model']['base_model'] = args.model_path
    if args.w_0 is not None:
        config['tree_ring']['w_0'] = args.w_0
    if args.alpha is not None:
        config['daam']['alpha'] = args.alpha
    if args.steagstamp_encoder:
        config['stegastamp']['encoder_path'] = args.steagstamp_encoder
    if args.steagstamp_decoder:
        config['stegastamp']['decoder_path'] = args.steagstamp_decoder
    if args.output_dir:
        config['generation']['output_dir'] = args.output_dir

    # 验证输入
    if not args.prompt and not args.prompts:
        parser.error('请提供 --prompt 或 --prompts')

    # 初始化管线
    print("=" * 60)
    print("  AIGC 三级版权溯源系统")
    print("  Tree-Ring + DAAM Guided Strength + StegaStamp")
    print("=" * 60)
    print()

    pipeline = ThreeLayerWatermarkPipeline(config)

    # 单张生成
    if args.prompt:
        user_id = args.user_id or 'user_001'
        print(f"[生成] 提示词: {args.prompt}")
        print(f"       用户ID: {user_id}")
        print(f"       种子: {args.seed or config['generation']['seed']}")
        print()

        result = pipeline.generate(
            prompt=args.prompt,
            user_id=user_id,
            seed=args.seed,
            output_dir=args.output_dir,
            save_comparison=args.save_comparison,
        )

        print(f"\n结果已保存到: {result['output_path']}")

        # 保存 metadata
        metadata_path = result['output_path'].replace('.png', '_meta.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        print(f"元数据已保存到: {metadata_path}")

    # 批量生成
    elif args.prompts:
        with open(args.prompts, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]

        print(f"[批量生成] 共 {len(prompts)} 张图像")
        output_dir = args.output_dir or config['generation']['output_dir']
        os.makedirs(output_dir, exist_ok=True)

        all_results = []
        for i, prompt in enumerate(prompts):
            user_id = f"{args.user_id_prefix}{i:03d}"
            print(f"\n[{i+1}/{len(prompts)}] {prompt}")

            result = pipeline.generate(
                prompt=prompt,
                user_id=user_id,
                seed=args.seed + i if args.seed else None,
                output_dir=output_dir,
            )
            all_results.append(result)

        # 保存批量结果
        batch_meta_path = os.path.join(output_dir, 'batch_results.json')
        with open(batch_meta_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n批量结果已保存到: {batch_meta_path}")


if __name__ == '__main__':
    main()
