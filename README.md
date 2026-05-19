# AIGC 三级版权溯源系统

基于跨域嵌套与语义感知掩码的 AIGC 多级版权溯源架构。

用户输入提示词 + 用户ID → 自动生成带三层水印的图像。

---

## 项目声明

| 项目 | 内容 |
|------|------|
| **项目名称** | AIGC 三级版权溯源系统 |
| **项目作者** | Huang Kai（黄锴） |
| **作者单位** | 暨南大学 · 网络空间安全学院 |
| **学号** | 2022102917 |
| **指导教师** | 吴小天 |
| **配套论文** | 《基于跨域嵌套与语义感知掩码的AIGC多级版权溯源架构研究》 |
| **开源协议** | MIT License |

### 技术栈

| 分类 | 详情 |
|------|------|
| **开发语言** | Python 3.7 – 3.9 |
| **深度学习框架** | PyTorch 1.13+ / 2.8+, TensorFlow 1.15.5 |
| **扩散模型** | Stable Diffusion 2.1 Base |
| **Web 框架** | Streamlit 1.28+ |

### 核心技术

| 技术 | 论文 | 作用域 |
|------|------|--------|
| **Tree-Ring Watermarks** | NeurIPS 2023 | 频域（latent 傅里叶） |
| **DAAM Guided Strength** | ACL 2023 | 空间语义域 |
| **StegaStamp** | CVPR 2020 | 像素域 |
| **BCH + CRC 纠错** | — | 编码层 |

---

## 三层架构

| 层级 | 技术 | 作用域 | 核心机制 |
|------|------|--------|---------|
| 品牌层 | Tree-Ring | 频域（latent傅里叶） | 环形频域水印，全局保护 |
| 验证层 | DAAM Guided Strength | 空间语义域 | 语义覆盖率 r 调制 w' = w₀ × (1 - α × r) |
| 用户层 | StegaStamp | 像素域 | 100比特用户ID，后处理叠加 |

**设计要点**：前两层**跨域嵌套**（结构耦合），第三层**跨域叠加**（后处理不干扰前两层）

## 环境兼容性说明

**重要**：三层架构依赖**不兼容**的软件环境，无法在同一个 Python 进程中运行：

| 层级 | 环境 | 关键依赖 | 冲突原因 |
|------|------|---------|---------|
| 验证层 (DAAM) | `daam` | diffusers 0.21.2 | DAAM trace 需要 diffusers 0.21+ |
| 品牌层 (Tree-Ring) | `tree-ring` | diffusers 0.11.1 | InversableStableDiffusionPipeline 需 0.11 |
| 用户层 (StegaStamp) | `stegastamp` | TensorFlow 1.15.5 | TF 1.x 与 PyTorch 内存冲突 |

**解决方案**：使用**分阶段执行架构**，通过 `subprocess` + `conda run` 在独立环境中运行每个阶段，中间结果通过 JSON 文件传递。

## 系统架构

```
输入文本提示词 p + 用户ID
        ↓
[阶段1: DAAM语义分析] ───→ 生成语义掩码 M，计算覆盖率 r
        ↓ (JSON文件传递)
[阶段2: Tree-Ring生成] ──→ 注入频域水印，生成图像
        ↓ (图像文件传递)
[阶段3: StegaStamp编码] ─→ 叠加用户指纹
        ↓
输出带三级水印的最终图像
```

## 快速开始

### 1. 环境准备（三环境安装）

```bash
# ===== 环境1: DAAM (diffusers 0.21.2) =====
conda create -n daam python=3.8
conda activate daam
pip install -r requirements-daam.txt

# ===== 环境2: Tree-Ring (diffusers 0.11.1) =====
conda create -n tree-ring python=3.8
conda activate tree-ring
pip install -r requirements-treering.txt

# Tree-Ring 水印库 (需从源码安装)
git clone https://github.com/Tree-Ring/tree-ring-watermark.git
cd tree-ring-watermark
pip install -e .

# ===== 环境3: StegaStamp (TensorFlow 1.15.5) =====
conda create -n stegastamp python=3.7
conda activate stegastamp

# 安装 CUDA 10.0 (TF 1.15 需要)
conda install cudatoolkit=10.0 cudnn=7.6

pip install -r requirements-stegastamp.txt

# StegaStamp 模型
# 下载 StegaStamp SavedModel 到 saved_models/stegastamp_140k/
```

### 2. 使用阶段管理器运行（推荐）

```bash
# 完整三阶段流水线
python stage_manager.py --prompt "a cat sitting on a windowsill" --user-id user_001

# 指定工作目录
python stage_manager.py --prompt "a beautiful sunset" --user-id user_002 --work-dir ./work

# 运行单个阶段
python stage_manager.py --stage daam --prompt "a cat"
python stage_manager.py --stage tree_ring --prompt "a cat" (需要阶段1的输出)
```

### 3. 使用传统方式（单环境运行）

如果只想测试某一层的功能（跳过其他层）：

```bash
# 仅生成 Tree-Ring 水印（跳过 DAAM 和 StegaStamp）
conda activate tree-ring
python generate.py --prompt "a cat" --user-id user_001 --skip-daam --skip-stegastamp

# 仅添加 StegaStamp（对已生成图像）
conda activate stegastamp
python scripts/stage3_stegastamp.py --input stage_input.json --output stage_output.json
```

### 4. 批量生成

```bash
# 创建提示词文件
cat > prompts.txt << EOF
a cat on a windowsill
a beautiful sunset
a red apple on a wooden table
EOF

# 批量运行
python scripts/batch_generate.py --prompts prompts.txt --output-dir ./results
```

### 5. 水印检测

```bash
# 检测单张图像
python detect.py --image output/cat_user_001.png --user-id user_001

# 检测整个目录
python detect.py --image-dir ./output
```

## 项目结构

```
aigc-watermark-system/
├── stage_manager.py              # 阶段管理器（解决三环境冲突）
├── generate.py                   # 主生成入口（单环境模式）
├── detect.py                     # 水印检测入口
├── verify.py                     # 系统验证脚本
├── web_ui.py                     # Streamlit Web 界面
├── start_web.bat                 # Web 界面启动脚本（Windows）
├── start_web.sh                  # Web 界面启动脚本（Linux/Mac）
│
├── configs/
│   └── default.yaml              # 默认配置（模型路径、水印参数）
│
├── core/                         # 核心模块
│   ├── __init__.py
│   ├── pipeline.py               # 三级水印主管线 + 检测器
│   ├── tree_ring.py              # Tree-Ring 频域水印模块
│   ├── daam_mask.py              # DAAM 语义掩码 + Guided Strength
│   └── stegastamp.py             # StegaStamp 编/解码器
│
├── utils/                        # 工具模块
│   ├── __init__.py
│   ├── metrics.py                # PSNR / SSIM / FFT 检测指标
│   ├── user_manager.py           # 用户ID管理（Exact/Hamming/BCH_CRC 三种模式）
│   ├── bch_codec.py              # BCH 纠错码编解码
│   └── crc.py                    # CRC-8/16/32 循环冗余校验
│
├── scripts/                      # 分阶段执行脚本
│   ├── stage1_daam.py            # 阶段1: DAAM 语义分析 (daam 环境)
│   ├── stage2_treering.py        # 阶段2: Tree-Ring 水印注入 (tree-ring 环境)
│   ├── stage3_stegastamp.py      # 阶段3: StegaStamp 编码 (stegastamp 环境)
│   ├── batch_generate.py         # 批量生成脚本
│   └── demo_bch_crc.py           # BCH+CRC 纠错方案演示
│
├── docs/                         # 方案文档
│   ├── stegastamp_hamming_scheme.md    # 汉明距离验证方案
│   ├── stegastamp_bch_crc_plan.md      # BCH+CRC 纠错方案
│   └── WEB_UI.md                       # Web 前端使用说明
│
├── requirements.txt              # 基础依赖说明
├── requirements-daam.txt         # DAAM 环境依赖
├── requirements-treering.txt     # Tree-Ring 环境依赖
└── requirements-stegastamp.txt   # StegaStamp 环境依赖
```

## 核心 API

### 阶段管理器方式（解决环境冲突）

```python
from stage_manager import StageManager

# 创建阶段管理器
manager = StageManager(work_dir='./work')

# 运行完整流水线
result = manager.run_full_pipeline(
    prompt="a cat sitting on a windowsill",
    user_id="user_001",
    seed=42
)

print(f"最终图像: {result['final_image']}")
print(f"语义覆盖率: {result['stages']['daam']['semantic_ratio']:.4f}")
print(f"调制强度: {result['stages']['daam']['modulated_intensity']:.6f}")
print(f"StegaStamp准确率: {result['stages']['stegastamp']['bit_accuracy']:.4f}")
```

### 单环境方式（传统用法）

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
)
print(result['semantic_ratio'])
print(result['modulated_intensity'])

# 检测
detector = WatermarkDetector(config)
result = detector.detect('output/cat_user_001.png', user_id='user_001')
print(result['tree_ring'])
print(result['stegastamp'])
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

1. **上传代码**
   ```bash
   # 将整个 aigc-watermark-system/ 上传到 /root/autodl-tmp/
   rsync -avz ./aigc-watermark-system/ root@your-autodl:/root/autodl-tmp/aigc-watermark-system/
   ```

2. **创建三个 conda 环境**（按上面的步骤）

3. **确认模型路径**（编辑 `configs/default.yaml`）
   ```yaml
   model:
     base_model: "/root/autodl-tmp/models/stable-diffusion-2-1-base"
   ```

4. **运行阶段管理器**
   ```bash
   cd /root/autodl-tmp/aigc-watermark-system
   python stage_manager.py --prompt "..." --user-id user_001
   ```

## 故障排除

### 环境冲突问题
- **错误**: `ImportError: cannot import name 'InversableStableDiffusionPipeline'`
- **原因**: 在错误的 conda 环境中运行
- **解决**: 确保使用 `tree-ring` 环境

### CUDA 版本问题
- **错误**: `CUDA_ERROR_NO_DEVICE` 或 `cuDNN version mismatch`
- **原因**: TF 1.15 需要 CUDA 10.0，PyTorch 2.0+ 需要 CUDA 11.7+
- **解决**: 使用分阶段执行，每个环境独立配置 CUDA

### 显存不足
- **错误**: `CUDA out of memory`
- **解决**: 减小 batch size，或使用 `torch.cuda.empty_cache()`

## 引用

```bibtex
@mastersthesis{huang2026aigc,
  title={基于跨域嵌套与语义感知掩码的AIGC多级版权溯源架构研究},
  author={黄锴},
  school={暨南大学},
  year={2026}
}
```
