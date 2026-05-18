# AIGC 三级版权溯源系统 - Claude Code 任务

你是一个软件工程师，正在帮硕士研究生完成毕业论文的代码项目。
需要实现一个完整的三级水印图像生成系统。

## 三层架构

| 层级 | 技术 | 作用域 | 核心机制 |
|------|------|--------|---------|
| 品牌层 | Tree-Ring | 频域（latent傅里叶） | 环形频域水印，全局保护 |
| 验证层 | DAAM Guided Strength | 空间语义域 | 语义覆盖率 r 调制 w_prime = w_0 * (1 - alpha * r) |
| 用户层 | StegaStamp | 像素域 | 100比特用户ID，后处理叠加 |

关键设计：前两层跨域嵌套（结构耦合），第三层跨域叠加（后处理不干扰前两层）

## 部署环境（AutoDL Ubuntu 服务器）
- Conda环境: tree-ring (PyTorch, diffusers 0.11.1), daam (diffusers 0.21.2), stegastamp (TensorFlow 1.15.5)
- 模型: /root/autodl-tmp/models/stable-diffusion-2-1-base/ 和 stable-diffusion-v1-5/
- 数据盘: /root/autodl-tmp/
- StegaStamp SavedModel: saved_models/stegastamp_140k/encoder 和 decoder

## 需要创建的文件

```
aigc-watermark-system/
├── core/
│   ├── __init__.py
│   ├── pipeline.py          # 三级水印主管线
│   ├── tree_ring.py         # Tree-Ring 频域水印模块
│   ├── daam_mask.py         # DAAM 语义掩码 + Guided Strength
│   └── stegastamp.py        # StegaStamp 用户指纹模块
├── utils/
│   ├── __init__.py
│   ├── metrics.py           # PSNR/SSIM/FFT检测指标
│   └── user_manager.py      # 用户ID管理（字符串 <-> 比特串）
├── configs/
│   └── default.yaml         # 默认配置
├── scripts/
│   └── batch_generate.py    # 批量生成
├── generate.py              # 主入口
├── detect.py                # 水印检测
├── requirements.txt
└── README.md
```

## 核心接口

```python
# 主生成
pipeline = ThreeLayerWatermarkPipeline(config)
result = pipeline.generate(
    prompt="a cat sitting on a windowsill",
    user_id="user_001",
    seed=42,
    save_comparison=True
)

# 检测
detector = WatermarkDetector(config)
result = detector.detect(image_path, user_id="user_001")
```

## 关键实现细节

### Guided Strength 公式
w_prime = w_0 * (1 - alpha * r)
- w_0 = 0.05（基础强度）
- alpha = 0.3（调制系数）
- r = DAAM语义覆盖率（掩码中值 > tau=0.3 的像素占比）

### Tree-Ring 实现
- 使用 InversableStableDiffusionPipeline
- 在初始 latent 的傅里叶空间环形区域注入水印
- 环形区域：64x64 latent 空间，半径8-12像素
- 关键函数：get_watermarking_pattern(), get_watermarking_mask(), inject_watermark(), eval_watermark()

### DAAM 实现
- 使用 trace(pipe) 捕获 cross-attention
- 生成全局热力图 -> 二值化掩码（阈值 tau=0.3）
- 计算语义覆盖率 r = sum(M > tau) / (H * W)

### StegaStamp 实现
- SavedModel 加载（TF 1.15）
- 400x400 输入图像
- 100比特 secret_size
- Encoder -> 编码水印图像, Decoder -> 提取用户ID

### 约束
- PyTorch 和 TF 1.15 需共存（先PyTorch生成图像，再TF编码StegaStamp）
- 服务器环境，代码需简洁高效
- 所有模块独立可测试

## 请完成所有代码文件，确保：
1. 代码完整可运行，不是伪代码
2. 每个模块有清晰的接口
3. 包含完整的 README.md
4. requirements.txt 列出所有依赖
5. generate.py 和 detect.py 作为 CLI 入口

现在开始编写所有文件。
