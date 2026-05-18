# AIGC 三级版权溯源系统

基于跨域嵌套与语义感知掩码的 AIGC 多级版权溯源架构。

用户输入提示词 + 用户ID → 自动生成带三层水印的图像。

## 三层架构

| 层级 | 技术 | 作用域 | 核心机制 |
|------|------|--------|---------|
| 品牌层 | Tree-Ring | 频域（latent傅里叶） | 环形频域水印，全局保护 |
| 验证层 | DAAM Guided Strength | 空间语义域 | 语义覆盖率 r 调制 w' = w₀ × (1 - α × r) |
| 用户层 | StegaStamp | 像素域 | 100比特用户ID，后处理叠加 |

**设计要点**：前两层**跨域嵌套**（结构耦合），第三层**跨域叠加**（后处理不干扰前两层）

## 系统架构

```
输入文本提示词 p + 用户ID
        ↓
[1] DAAM 语义分析 ───────→ 生成语义掩码 M，计算覆盖率 r
        ↓
[2] Guided Strength ─────→ w' = w₀ × (1 - α × r)，α=0.3
        ↓
[3] Tree-Ring 注入 ──────→ 在初始噪声 latent 傅里叶空间注入频域水印
        ↓
[4] Stable Diffusion ────→ 生成带两层嵌套水印的图像
        ↓
[5] StegaStamp 编码 ─────→ 像素域嵌入用户 ID（100比特）
        ↓
输出带三级水印的最终图像
```

## 快速开始

### 环境准备

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置模型路径
# 编辑 configs/default.yaml，设置模型路径
```

### 单张生成

```bash
python generate.py --prompt "a cat sitting on a windowsill" --user-id user_001
```

### 生成对比图

```bash
python generate.py --prompt "a beautiful sunset" --user-id user_002 --save-comparison
```

### 批量生成

```bash
# 创建提示词文件（每行一个提示词）
echo "a cat on a windowsill" > prompts.txt
echo "a beautiful sunset" >> prompts.txt
echo "a red apple on a wooden table" >> prompts.txt

# 批量生成
python scripts/batch_generate.py --prompts prompts.txt --output-dir ./results
```

### 水印检测

```bash
# 检测单张图像
python detect.py --image output/cat_user_001.png --user-id user_001

# 检测整个目录
python detect.py --image-dir ./output
```

## 项目结构

```
aigc-watermark-system/
├── core/
│   ├── pipeline.py          # 三级水印主管线
│   ├── tree_ring.py         # Tree-Ring 频域水印模块
│   ├── daam_mask.py         # DAAM 语义掩码 + Guided Strength
│   └── stegastamp.py        # StegaStamp 用户指纹模块
├── utils/
│   ├── metrics.py           # PSNR/SSIM/FFT检测指标
│   └── user_manager.py      # 用户ID管理
├── configs/
│   └── default.yaml         # 默认配置
├── scripts/
│   └── batch_generate.py    # 批量生成
├── generate.py              # 主入口
├── detect.py                # 水印检测
├── requirements.txt
└── README.md
```

## 核心 API

```python
from core.pipeline import ThreeLayerWatermarkPipeline, WatermarkDetector
import yaml

# 加载配置
with open('configs/default.yaml') as f:
    config = yaml.safe_load(f)

# 生成
pipeline = ThreeLayerWatermarkPipeline(config)
result = pipeline.generate(
    prompt="a cat sitting on a windowsill",
    user_id="user_001",
    seed=42,
    save_comparison=True,
)
print(result['metrics'])  # PSNR, SSIM, FFT等
print(result['semantic_ratio'])  # DAAM 语义覆盖率
print(result['modulated_intensity'])  # 调制后的水印强度

# 检测
detector = WatermarkDetector(config)
result = detector.detect('output/cat_user_001.png', user_id='user_001')
print(result['tree_ring'])   # Tree-Ring 检测结果
print(result['stegastamp'])  # StegaStamp 检测结果
```

## Guided Strength 公式

```
w' = w₀ × (1 - α × r)
```

- **w₀** = 0.05：基础水印强度
- **α** = 0.3：调制系数（经验值，通过消融实验确定）
- **r** = DAAM语义覆盖率：掩码中值 > τ=0.3 的像素占比

| 语义覆盖率 r | 调制后强度 w' | 降低幅度 |
|-------------|--------------|---------|
| 0%（无覆盖）| 0.050 | 0% |
| 25%（低覆盖）| 0.046 | 7.6% |
| 50%（中覆盖）| 0.043 | 15% |
| 75%（高覆盖）| 0.039 | 22.5% |
| 100%（全覆盖）| 0.035 | 30% |

## 部署到 AutoDL

1. 将整个 `aigc-watermark-system/` 目录上传到 `/root/autodl-tmp/`
2. 确认 `configs/default.yaml` 中的模型路径正确
3. 确认 StegaStamp SavedModel 路径正确
4. 在 `tree-ring` conda 环境中运行：
   ```bash
   conda activate tree-ring
   cd /root/autodl-tmp/aigc-watermark-system
   python generate.py --prompt "..." --user-id user_001
   ```

## 引用

```bibtex
@mastersthesis{huang2026aigc,
  title={基于跨域嵌套与语义感知掩码的AIGC多级版权溯源架构研究},
  author={黄锴},
  school={暨南大学},
  year={2026}
}
```
