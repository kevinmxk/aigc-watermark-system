"""
AIGC 三级版权溯源系统 - 水印检测工具
检测图像中的 Tree-Ring 频域水印和 StegaStamp 用户指纹
"""
import argparse
import os
import sys
import json
import yaml

sys.path.insert(0, os.path.dirname(__file__))

from core.pipeline import WatermarkDetector


def load_config(config_path: str = None) -> dict:
    """加载配置文件"""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'configs', 'default.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description='AIGC 三级版权溯源系统 - 水印检测',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 检测单张图像
  python detect.py --image output/cat_user_001.png --user-id user_001

  # 检测多张图像
  python detect.py --image-dir ./output --user-id user_001

  # 带基线对比
  python detect.py --image output/cat.png --user-id user_001 --baseline output/cat_no_wm.png

  # 指定配置文件
  python detect.py --image output/cat.png --config my_config.yaml
        """
    )

    parser.add_argument('--image', type=str, help='待检测图像路径')
    parser.add_argument('--image-dir', type=str, help='待检测图像目录')
    parser.add_argument('--user-id', type=str, default=None,
                        help='期望的用户 ID（用于验证）')
    parser.add_argument('--baseline', type=str, default=None,
                        help='无水印基线图像路径')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--output', type=str, default=None,
                        help='检测结果输出路径（JSON）')

    args = parser.parse_args()

    if not args.image and not args.image_dir:
        parser.error('请提供 --image 或 --image-dir')

    config = load_config(args.config)
    detector = WatermarkDetector(config)

    images = []
    if args.image:
        images = [args.image]
    elif args.image_dir:
        images = [
            os.path.join(args.image_dir, f)
            for f in os.listdir(args.image_dir)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ]
        images.sort()

    print("=" * 60)
    print("  AIGC 三级版权溯源系统 - 水印检测")
    print("=" * 60)
    print()

    all_results = []

    for img_path in images:
        print(f"[检测] {img_path}")
        result = detector.detect(
            image_path=img_path,
            user_id=args.user_id,
            baseline_image_path=args.baseline,
        )

        # 打印结果
        if 'tree_ring' in result:
            tr = result['tree_ring']
            status = 'DETECTED' if tr.get('detected') else 'NOT DETECTED'
            print(f"  Tree-Ring: {status} (energy={tr.get('ring_energy', 'N/A')})")

        if 'stegastamp' in result:
            ss = result['stegastamp']
            if 'bit_accuracy' in ss:
                print(f"  StegaStamp: Bit Acc={ss['bit_accuracy']*100:.2f}%, "
                      f"String Acc={'PASS' if ss.get('string_accuracy') else 'FAIL'}")

        overall = 'PASS' if result.get('overall') else 'FAIL'
        print(f"  综合判断: {overall}")
        print()

        result['image_path'] = img_path
        all_results.append(result)

    # 保存结果
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"检测结果已保存到: {args.output}")

    # 统计
    detected_count = sum(1 for r in all_results if r.get('overall'))
    print(f"\n总计: {len(images)} 张图像, {detected_count} 张通过检测 "
          f"({detected_count/len(images)*100:.1f}%)")


if __name__ == '__main__':
    main()
