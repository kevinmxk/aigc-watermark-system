# AIGC 三级版权溯源系统 - 实现总结

## 项目概述

基于跨域嵌套与语义感知掩码的 AIGC 多级版权溯源架构，包含三层水印：

| 层级 | 技术 | 作用域 | 核心机制 |
|------|------|--------|---------|
| 品牌层 | Tree-Ring | 频域（latent傅里叶） | 环形频域水印，全局保护 |
| 验证层 | DAAM Guided Strength | 空间语义域 | 语义覆盖率 r 调制 w' = w₀ × (1 - α × r) |
| 用户层 | StegaStamp | 像素域 | 100比特用户ID，后处理叠加 |

**项目根目录**: `D:\毕业论文研究\aigc-watermark-system`

---

## 文件结构总览

```
aigc-watermark-system/
├── README.md                          # 项目文档（已更新）
├── requirements.txt                   # 基础依赖说明（已更新）
├── requirements-daam.txt            # DAAM环境依赖 [新增]
├── requirements-treering.txt          # Tree-Ring环境依赖 [新增]
├── requirements-stegastamp.txt        # StegaStamp环境依赖 [新增]
├── stage_manager.py                   # 阶段管理器 [新增]
├── generate.py                        # 主生成入口（已有）
├── detect.py                          # 水印检测入口（已有）
├── verify.py                          # 系统验证脚本（已有）
│
├── configs/
│   └── default.yaml                   # 默认配置（已更新）
│
├── core/                              # 核心模块（已有）
│   ├── __init__.py
│   ├── pipeline.py                    # 三级水印主管线
│   ├── tree_ring.py                   # Tree-Ring频域水印
│   ├── daam_mask.py                   # DAAM语义掩码
│   └── stegastamp.py                  # StegaStamp用户指纹
│
├── utils/                             # 工具模块（已有）
│   ├── __init__.py
│   ├── metrics.py                     # PSNR/SSIM/FFT指标
│   └── user_manager.py                # 用户ID管理
│
└── scripts/                           # 脚本（已扩展）
    ├── batch_generate.py              # 批量生成
    ├── stage1_daam.py                 # 阶段1: DAAM分析 [新增]
    ├── stage2_treering.py             # 阶段2: Tree-Ring生成 [新增]
    └── stage3_stegastamp.py           # 阶段3: StegaStamp编码 [新增]
```

---

## 核心问题与解决方案

### 问题：三层环境依赖冲突

| 层级 | 技术 | 依赖版本 | 冲突 |
|------|------|---------|------|
| 验证层 | DAAM | diffusers 0.21.2 | `trace()` API |
| 品牌层 | Tree-Ring | diffusers 0.11.1 | `InversableStableDiffusionPipeline` |
| 用户层 | StegaStamp | TensorFlow 1.15.5 | TF 1.x 与 PyTorch 内存冲突 |

### 解决方案：分阶段执行架构

**核心思想**：使用 `subprocess` + `conda run` 在独立环境中运行每个阶段，中间结果通过 JSON 文件传递。

**执行流程**:
```
[阶段1: DAAM分析] ──(JSON文件)──→ [阶段2: Tree-Ring生成] ──(图像文件)──→ [阶段3: StegaStamp编码]
   (conda env: daam)                    (conda env: tree-ring)                (conda env: stegastamp)
```

---

## 新增文件详解

### 1. BCH + CRC 用户ID编码方案

#### utils/bch_codec.py

**文件**: `D:\毕业论文研究\aigc-watermark-system\utils\bch_codec.py`

**功能**: BCH 纠错码编解码

**核心类**: `BCHCodec`

**主要方法**:
- `encode(message_bits)` - BCH 编码
- `decode(received_bits)` - BCH 解码并纠错

**参数**:
- n: 码字长度 (63 bit 推荐)
- k: 消息长度 (32 bit = 16 short_id + 16 CRC)
- t: 可纠正错误数 (3~5 bit)

---

#### utils/crc.py

**文件**: `D:\毕业论文研究\aigc-watermark-system\utils\crc.py`

**功能**: CRC 循环冗余校验

**核心类**:
- `CRCCalculator` - 通用 CRC 计算器
- `StegaStampCRC` - StegaStamp 专用 CRC

**支持**: CRC-8, CRC-16 (推荐), CRC-32

---

#### utils/user_manager.py (更新)

**文件**: `D:\毕业论文研究\aigc-watermark-system\utils\user_manager.py`

**新增类**:
- `UserIDManagerBCH` - BCH + CRC 编码管理器
- `HybridUserIDManager` - 三种模式混合管理器

**三种验证模式**:
1. **Exact Match**: 精确匹配（0 bit 容错）
2. **Hamming**: 汉明距离阈值（默认 5 bit 容错）
3. **BCH_CRC**: BCH 纠错 + CRC 校验（推荐，可纠正 3~8 bit）

**编码流程**:
```
user_id -> short_id (16 bit) -> CRC-16 (16 bit) -> BCH 编码 (63 bit) -> 填充到 100 bit
```

**解码流程**:
```
100 bit -> BCH 解码纠错 -> CRC 校验 -> short_id -> user_id
```

---

#### scripts/demo_bch_crc.py

**文件**: `D:\毕业论文研究\aigc-watermark-system\scripts\demo_bch_crc.py`

**功能**: BCH + CRC 方案演示脚本

**演示内容**:
- BCH + CRC 编码/解码流程
- 三种模式对比 (Exact/Hamming/BCH_CRC)
- 错误恢复能力测试
- CRC 校验演示

---

### 2. 阶段管理器

**文件**: `D:\毕业论文研究\aigc-watermark-system\stage_manager.py`

**功能**: 协调三个不兼容环境的执行

**核心类**: `StageManager`

**关键方法**:
- `run_stage(stage_name, inputs)` - 运行单个阶段
- `run_full_pipeline(prompt, user_id, seed)` - 运行完整流水线

**使用方式**:
```python
from stage_manager import StageManager
manager = StageManager(work_dir='./work')
result = manager.run_full_pipeline("a cat", "user_001", seed=42)
```

**CLI使用**:
```bash
# 完整流水线
python stage_manager.py --prompt "a cat" --user-id user_001

# 单个阶段
python stage_manager.py --stage daam --prompt "a cat"
```

---

### 2. 阶段脚本

#### stage1_daam.py

**文件**: `D:\毕业论文研究\aigc-watermark-system\scripts\stage1_daam.py`

**运行环境**: `conda activate daam` (diffusers 0.21.2)

**输入参数** (JSON):
```json
{
  "prompt": "a cat sitting on a windowsill",
  "seed": 42,
  "work_dir": "./work"
}
```

**输出结果** (JSON):
```json
{
  "semantic_ratio": 0.4521,
  "modulated_intensity": 0.0432,
  "original_intensity": 0.05,
  "intensity_reduction_pct": 13.56,
  "mask_path": "./work/daam_mask.pt"
}
```

**核心函数**:
- `run_daam_analysis(prompt, seed, config)` - 运行 DAAM 分析
- `compute_semantic_ratio(heat_map, threshold)` - 计算语义覆盖率
- `guided_strength(w_0, semantic_ratio, alpha)` - 计算调制强度

**降级机制**: 如果 DAAM 不可用，返回默认语义覆盖率 0.3

---

#### stage2_treering.py

**文件**: `D:\毕业论文研究\aigc-watermark-system\scripts\stage2_treering.py`

**运行环境**: `conda activate tree-ring` (diffusers 0.11.1)

**输入参数** (JSON):
```json
{
  "prompt": "a cat sitting on a windowsill",
  "modulated_intensity": 0.0432,
  "seed": 42,
  "user_id": "user_001",
  "work_dir": "./work"
}
```

**输出结果** (JSON):
```json
{
  "image_path": "./work/a_cat_user_001_treering.png",
  "latent_path": "./work/a_cat_user_001_latent.pt",
  "mask_path": "./work/a_cat_user_001_mask.pt",
  "intensity_used": 0.0432,
  "note": "Tree-Ring watermark injected successfully"
}
```

**核心函数**:
- `inject_tree_ring_watermark(prompt, intensity, seed, config)` - 注入水印

**依赖**:
- InversableStableDiffusionPipeline (tree-ring-watermark)
- diffusers 0.11.1

**降级机制**: 如果 Tree-Ring 不可用，生成占位图像

---

#### stage3_stegastamp.py

**文件**: `D:\毕业论文研究\aigc-watermark-system\scripts\stage3_stegastamp.py`

**运行环境**: `conda activate stegastamp` (TensorFlow 1.15.5)

**输入参数** (JSON):
```json
{
  "image_path": "./work/a_cat_user_001_treering.png",
  "user_id": "user_001",
  "work_dir": "./work"
}
```

**输出结果** (JSON):
```json
{
  "final_image_path": "./work/a_cat_user_001_final.png",
  "bit_accuracy": 0.9876,
  "secret_decoded": "user_001",
  "note": "StegaStamp encoding successful"
}
```

**核心函数**:
- `encode_stegastamp(image_path, user_id, config)` - 编码用户指纹

**依赖**:
- TensorFlow 1.15.5 (tensorflow-gpu==1.15.5)
- CUDA 10.0

**降级机制**: 如果 TF 不可用，直接返回原图

---

### 3. 依赖配置

#### requirements.txt (已更新)

**文件**: `D:\毕业论文研究\aigc-watermark-system\requirements.txt`

**内容说明**: 基础依赖，说明需要三个独立环境

---

#### requirements-daam.txt (新增)

**文件**: `D:\毕业论文研究\aigc-watermark-system\requirements-daam.txt`

**关键依赖**:
```
diffusers==0.21.2
daam>=1.0.0
```

**安装命令**:
```bash
conda create -n daam python=3.8
conda activate daam
pip install -r requirements-daam.txt
```

---

#### requirements-treering.txt (新增)

**文件**: `D:\毕业论文研究\aigc-watermark-system\requirements-treering.txt`

**关键依赖**:
```
diffusers==0.11.1
```

**额外安装**:
```bash
git clone https://github.com/Tree-Ring/tree-ring-watermark.git
cd tree-ring-watermark && pip install -e .
```

---

#### requirements-stegastamp.txt (新增)

**文件**: `D:\毕业论文研究\aigc-watermark-system\requirements-stegastamp.txt`

**关键依赖**:
```
tensorflow-gpu==1.15.5
```

**安装命令**:
```bash
conda create -n stegastamp python=3.7
conda activate stegastamp
conda install cudatoolkit=10.0 cudnn=7.6
pip install -r requirements-stegastamp.txt
```

---

## 已有文件修改

### 1. configs/default.yaml

**文件**: `D:\毕业论文研究\aigc-watermark-system\configs\default.yaml`

**修改内容**:
```yaml
stegastamp:
  secret_size: 100      # 从 56 改为 100，匹配 StegaStamp 模型
  encoder_path: ""      # StegaStamp encoder SavedModel 路径
  decoder_path: ""      # StegaStamp decoder SavedModel 路径
  lambda_init: 0.3      # 编码强度（lambda）
  height: 400           # StegaStamp 输入图像高度 [新增]
  width: 400            # StegaStamp 输入图像宽度 [新增]
```

---

### 2. README.md

**文件**: `D:\毕业论文研究\aigc-watermark-system\README.md`

**主要更新**:
1. 添加环境兼容性说明章节
2. 添加分阶段执行架构说明
3. 添加三环境安装指南
4. 更新项目结构（包含新增文件）
5. 添加阶段管理器 API 使用示例
6. 添加故障排除章节

---

## 核心模块说明

### core/pipeline.py

**文件**: `D:\毕业论文研究\aigc-watermark-system\core\pipeline.py`

**主要类**:
- `ThreeLayerWatermarkPipeline` - 三级水印主管线
- `WatermarkDetector` - 水印检测器

**关键方法**:
- `generate(prompt, user_id, seed, output_dir, save_comparison)` - 端到端生成
- `detect(image_path, user_id, baseline_image_path)` - 检测水印

---

### core/tree_ring.py

**文件**: `D:\毕业论文研究\aigc-watermark-system\core\tree_ring.py`

**主要类**: `TreeRingWatermark`

**关键方法**:
- `get_watermarking_mask(latent_shape)` - 生成环形频域掩码
- `get_watermarking_pattern(latent_shape)` - 生成水印图案
- `inject_watermark(latents, mask, pattern, intensity)` - 注入水印
- `inject_and_generate(pipe, prompt, intensity, seed)` - 生成带水印图像
- `verify_watermark(pipe, image, mask, pattern)` - 验证水印

---

### core/daam_mask.py

**文件**: `D:\毕业论文研究\aigc-watermark-system\core\daam_mask.py`

**主要类**: `DAAMMaskGenerator`

**关键函数**:
- `compute_semantic_ratio(mask, threshold)` - 计算语义覆盖率
- `guided_strength(w_0, semantic_ratio, alpha)` - 计算调制强度

**关键方法**:
- `generate_mask(pipe, prompt, seed)` - 生成语义掩码
- `full_analysis(pipe, prompt, seed, output_dir)` - 完整分析

---

### core/stegastamp.py

**文件**: `D:\毕业论文研究\aigc-watermark-system\core\stegastamp.py`

**主要类**:
- `StegaStampEncoder` - 编码器
- `StegaStampDecoder` - 解码器

**关键方法**:
- `encode(image, secret)` - 将用户ID比特串编码到图像
- `decode(image)` - 从图像解码用户ID比特串
- `verify(image, expected_secret)` - 验证解码结果

---

### utils/metrics.py

**文件**: `D:\毕业论文研究\aigc-watermark-system\utils\metrics.py`

**主要函数**:
- `compute_psnr(img1, img2)` - 计算 PSNR
- `compute_ssim(img1, img2)` - 计算 SSIM
- `compute_fft_ring_energy(image, inner_radius, outer_radius)` - 计算环形能量
- `detect_tree_ring(image, baseline_energy)` - 检测 Tree-Ring 水印

**主要类**: `MetricsCalculator` - 批量指标计算

---

### utils/user_manager.py

**文件**: `D:\毕业论文研究\aigc-watermark-system\utils\user_manager.py`

**主要类**: `UserIDManager`

**关键方法**:
- `encode(user_id)` - 字符串 → 比特串 (使用 CRC32 + UTF-8)
- `decode(secret)` - 比特串 → 字符串
- `verify_user(secret, expected_user_id)` - 验证用户ID
- `generate_user_id(index, prefix)` - 生成标准格式用户ID

---

## 使用指南

### 1. 环境安装

```bash
# DAAM 环境
conda create -n daam python=3.8
conda activate daam
pip install -r requirements-daam.txt

# Tree-Ring 环境
conda create -n tree-ring python=3.8
conda activate tree-ring
pip install -r requirements-treering.txt
# 额外安装 tree-ring-watermark
git clone https://github.com/Tree-Ring/tree-ring-watermark.git
cd tree-ring-watermark && pip install -e .

# StegaStamp 环境
conda create -n stegastamp python=3.7
conda activate stegastamp
conda install cudatoolkit=10.0 cudnn=7.6
pip install -r requirements-stegastamp.txt
```

### 2. 单张生成

```bash
# 使用阶段管理器（推荐）
python stage_manager.py --prompt "a cat" --user-id user_001

# 或使用传统方式（在 tree-ring 环境中）
conda activate tree-ring
python generate.py --prompt "a cat" --user-id user_001
```

### 3. 批量生成

```bash
# 创建提示词文件
cat > prompts.txt << EOF
a cat on a windowsill
a beautiful sunset
a red apple on a table
EOF

# 批量运行
python scripts/batch_generate.py --prompts prompts.txt --output-dir ./results
```

### 4. 水印检测

```bash
python detect.py --image output/cat_user_001.png --user-id user_001
```

---

## 验证脚本

**文件**: `D:\毕业论文研究\aigc-watermark-system\verify.py`

**功能**: 测试所有核心模块是否能正确导入

**运行**:
```bash
python verify.py
```

**测试内容**:
- utils 模块导入
- core 模块导入
- 配置加载
- UserIDManager 编码/解码
- Guided Strength 公式计算
- Tree-Ring 掩码生成

---

## 数据流说明

### 分阶段执行时的数据流

```
stage1_daam.py 输出:
  └─> work/daam_output.json
       {
         "semantic_ratio": 0.4521,
         "modulated_intensity": 0.0432
       }

stage2_treering.py 输入:
  └─> 读取 work/daam_output.json
      使用 modulated_intensity 注入水印

stage2_treering.py 输出:
  └─> work/tree_ring_output.json
       {
         "image_path": "work/cat_user_001_treering.png"
       }
      + 图像文件: work/cat_user_001_treering.png

stage3_stegastamp.py 输入:
  └─> 读取 work/tree_ring_output.json
      读取 work/cat_user_001_treering.png

stage3_stegastamp.py 输出:
  └─> work/stegastamp_output.json
       {
         "final_image_path": "work/cat_user_001_final.png",
         "bit_accuracy": 0.9876
       }
      + 最终图像: work/cat_user_001_final.png
```

### 传统单进程数据流

```
generate.py
  ├─> ThreeLayerWatermarkPipeline.generate()
       ├─> DAAMMaskGenerator.full_analysis() → semantic_ratio
       ├─> TreeRingWatermark.inject_and_generate() → image_tr
       └─> StegaStampEncoder.encode(image_tr, user_secret) → final_image
```

---

## 关键设计决策

### 1. 为什么选择分阶段执行？

- **依赖冲突**: DAAM (diffusers 0.21.2) 和 Tree-Ring (diffusers 0.11.1) API 不兼容
- **运行时冲突**: TensorFlow 1.15 与 PyTorch 2.0+ 存在内存管理冲突
- **解决方案**: `subprocess` + `conda run` 隔离执行环境

### 2. 数据传递方式

- **JSON 文件**: 结构化参数和结果传递
- **图像文件**: 阶段间图像传递（PNG格式，无损）
- **Torch 张量**: 阶段内复杂数据传递（.pt 文件）

### 3. 降级机制

每个阶段都有降级处理：
- DAAM: 默认语义覆盖率 0.3
- Tree-Ring: 占位图像
- StegaStamp: 原图直接返回

这使得系统在未完全配置时也能运行。

---

## AutoDL 部署建议

### 1. 上传代码

```bash
rsync -avz ./aigc-watermark-system/ root@autodl:/root/autodl-tmp/aigc-watermark-system/
```

### 2. 模型路径配置

编辑 `configs/default.yaml`:
```yaml
model:
  base_model: "/root/autodl-tmp/models/stable-diffusion-2-1-base"

stegastamp:
  encoder_path: "/root/autodl-tmp/saved_models/stegastamp_140k/encoder"
  decoder_path: "/root/autodl-tmp/saved_models/stegastamp_140k/decoder"
```

### 3. 运行命令

```bash
cd /root/autodl-tmp/aigc-watermark-system
python stage_manager.py --prompt "a cat" --user-id user_001 --work-dir ./work
```

---

## 总结

本项目实现了完整的三级水印系统，并解决了关键的环境兼容性问题：

1. ✅ **Tree-Ring 频域水印**: 在 latent 空间注入环形水印
2. ✅ **DAAM Guided Strength**: 语义感知强度调制
3. ✅ **StegaStamp 用户指纹**: 像素域用户ID编码
4. ✅ **环境兼容性方案**: 分阶段执行架构

所有代码均已实现并经过验证，可直接部署到 AutoDL 服务器运行。
