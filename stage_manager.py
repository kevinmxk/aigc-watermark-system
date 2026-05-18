"""
阶段化执行架构 - 解决三层环境冲突问题

三层水印无法在单一 Python 进程中运行，因为：
- Tree-Ring 需要 diffusers 0.11.1 (InversableStableDiffusionPipeline)
- DAAM 需要 diffusers 0.21.2 (trace 支持)
- StegaStamp 需要 TensorFlow 1.15.5

解决方案：
1. 分阶段运行，每阶段使用独立环境
2. 中间结果通过文件传递
3. 提供统一的协调脚本

流程：
    [阶段1: DAAM分析] -> 保存语义掩码和调制强度
        ↓
    [阶段2: Tree-Ring生成] -> 读取调制强度，生成图像
        ↓
    [阶段3: StegaStamp编码] -> 读取图像，叠加用户指纹
"""
import os
import json
import subprocess
import sys
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class StageConfig:
    """阶段配置"""
    name: str
    conda_env: str
    script: str
    input_keys: List[str]
    output_keys: List[str]


class StageManager:
    """
    阶段管理器 - 协调三个不兼容环境的执行

    使用 subprocess + conda run 在独立环境中运行每个阶段
    """

    STAGES = {
        'daam': StageConfig(
            name='DAAM语义分析',
            conda_env='daam',
            script='scripts/stage1_daam.py',
            input_keys=['prompt', 'seed'],
            output_keys=['semantic_ratio', 'modulated_intensity', 'daam_mask_path']
        ),
        'tree_ring': StageConfig(
            name='Tree-Ring水印生成',
            conda_env='tree-ring',
            script='scripts/stage2_treering.py',
            input_keys=['prompt', 'modulated_intensity', 'seed', 'user_id'],
            output_keys=['image_path', 'latent_path', 'mask_path']
        ),
        'stegastamp': StageConfig(
            name='StegaStamp用户指纹',
            conda_env='stegastamp',
            script='scripts/stage3_stegastamp.py',
            input_keys=['image_path', 'user_id'],
            output_keys=['final_image_path', 'bit_accuracy']
        ),
    }

    def __init__(self, work_dir: str = './work'):
        self.work_dir = work_dir
        os.makedirs(work_dir, exist_ok=True)
        self.state_file = os.path.join(work_dir, 'pipeline_state.json')
        self.state = self._load_state()

    def _load_state(self) -> Dict:
        """加载管道状态"""
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_state(self):
        """保存管道状态"""
        self.state['last_update'] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def run_stage(self, stage_name: str, inputs: Dict) -> Dict:
        """
        运行单个阶段

        Args:
            stage_name: 阶段名称 ('daam', 'tree_ring', 'stegastamp')
            inputs: 输入参数字典

        Returns:
            阶段输出结果
        """
        config = self.STAGES[stage_name]

        print(f"\n{'='*60}")
        print(f"  阶段: {config.name}")
        print(f"  环境: {config.conda_env}")
        print(f"{'='*60}")

        # 准备输入文件
        stage_input_file = os.path.join(self.work_dir, f'{stage_name}_input.json')
        with open(stage_input_file, 'w') as f:
            json.dump(inputs, f, indent=2)

        # 准备输出文件路径
        stage_output_file = os.path.join(self.work_dir, f'{stage_name}_output.json')

        # 构建命令
        cmd = [
            'conda', 'run', '-n', config.conda_env,
            'python', config.script,
            '--input', stage_input_file,
            '--output', stage_output_file,
        ]

        print(f"  执行: {' '.join(cmd)}")
        print(f"  工作目录: {self.work_dir}")

        # 执行阶段
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10分钟超时
                cwd=os.path.dirname(os.path.dirname(__file__))
            )

            if result.returncode != 0:
                print(f"  ERROR: {result.stderr}")
                raise RuntimeError(f"Stage {stage_name} failed: {result.stderr}")

            print(f"  STDOUT: {result.stdout}")

            # 读取输出
            if os.path.exists(stage_output_file):
                with open(stage_output_file, 'r') as f:
                    outputs = json.load(f)

                # 更新状态
                self.state[stage_name] = {
                    'status': 'completed',
                    'outputs': outputs,
                    'timestamp': datetime.now().isoformat()
                }
                self._save_state()

                return outputs
            else:
                raise RuntimeError(f"Stage {stage_name} did not produce output")

        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Stage {stage_name} timed out after 600s")

    def run_full_pipeline(self, prompt: str, user_id: str, seed: int = 42) -> Dict:
        """
        运行完整的三阶段流水线

        Args:
            prompt: 文本提示词
            user_id: 用户ID
            seed: 随机种子

        Returns:
            完整结果字典
        """
        results = {}

        # 阶段1: DAAM语义分析
        daam_inputs = {
            'prompt': prompt,
            'seed': seed,
            'work_dir': self.work_dir,
        }
        daam_outputs = self.run_stage('daam', daam_inputs)
        results['daam'] = daam_outputs

        # 阶段2: Tree-Ring水印生成
        tree_ring_inputs = {
            'prompt': prompt,
            'modulated_intensity': daam_outputs['modulated_intensity'],
            'seed': seed,
            'user_id': user_id,
            'work_dir': self.work_dir,
        }
        tree_outputs = self.run_stage('tree_ring', tree_ring_inputs)
        results['tree_ring'] = tree_outputs

        # 阶段3: StegaStamp用户指纹
        stega_inputs = {
            'image_path': tree_outputs['image_path'],
            'user_id': user_id,
            'work_dir': self.work_dir,
        }
        stega_outputs = self.run_stage('stegastamp', stega_inputs)
        results['stegastamp'] = stega_outputs

        # 保存最终结果
        final_result = {
            'prompt': prompt,
            'user_id': user_id,
            'seed': seed,
            'final_image': stega_outputs['final_image_path'],
            'stages': results,
            'timestamp': datetime.now().isoformat(),
        }

        final_path = os.path.join(self.work_dir, 'final_result.json')
        with open(final_path, 'w') as f:
            json.dump(final_result, f, indent=2)

        return final_result


def main():
    """CLI入口"""
    import argparse

    parser = argparse.ArgumentParser(
        description='AIGC 三级版权溯源系统 - 阶段化执行器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 完整流水线
  python stage_manager.py --prompt "a cat" --user-id user_001

  # 运行单个阶段
  python stage_manager.py --stage daam --prompt "a cat"

  # 从指定状态继续
  python stage_manager.py --resume --prompt "a cat" --user-id user_001
        """
    )

    parser.add_argument('--prompt', type=str, required=True, help='文本提示词')
    parser.add_argument('--user-id', type=str, default='user_001', help='用户ID')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--stage', type=str, choices=['daam', 'tree_ring', 'stegastamp'],
                        help='仅运行指定阶段')
    parser.add_argument('--work-dir', type=str, default='./work',
                        help='工作目录')
    parser.add_argument('--resume', action='store_true',
                        help='从上次状态继续')

    args = parser.parse_args()

    manager = StageManager(work_dir=args.work_dir)

    if args.stage:
        # 运行单个阶段
        inputs = {
            'prompt': args.prompt,
            'seed': args.seed,
            'work_dir': args.work_dir,
        }
        if args.user_id:
            inputs['user_id'] = args.user_id

        outputs = manager.run_stage(args.stage, inputs)
        print(f"\n阶段 {args.stage} 完成:")
        print(json.dumps(outputs, indent=2))
    else:
        # 运行完整流水线
        result = manager.run_full_pipeline(
            prompt=args.prompt,
            user_id=args.user_id,
            seed=args.seed
        )
        print(f"\n{'='*60}")
        print("  完整流水线完成!")
        print(f"{'='*60}")
        print(f"  最终图像: {result['final_image']}")
        print(f"  语义覆盖率: {result['stages']['daam']['semantic_ratio']:.4f}")
        print(f"  调制强度: {result['stages']['daam']['modulated_intensity']:.6f}")
        print(f"  StegaStamp 准确率: {result['stages']['stegastamp'].get('bit_accuracy', 'N/A')}")


if __name__ == '__main__':
    main()
