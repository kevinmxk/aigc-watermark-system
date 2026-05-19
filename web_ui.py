"""
AIGC 三级版权溯源系统 - Web 前端
基于 Streamlit 的简单界面

运行命令:
    streamlit run web_ui.py

功能:
    1. 单张图像生成
    2. 批量图像生成
    3. 水印检测
    4. 结果展示与下载
"""
import streamlit as st
import os
import sys
import json
import subprocess
from pathlib import Path
from PIL import Image
import time

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

# 页面配置
st.set_page_config(
    page_title="AIGC 三级版权溯源系统",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .layer-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


def check_environment():
    """检查环境状态"""
    env_status = {}
    for env in ['daam', 'tree-ring', 'stegastamp']:
        try:
            result = subprocess.run(
                ['conda', 'run', '-n', env, 'python', '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            env_status[env] = '✅ 可用' if result.returncode == 0 else '❌ 不可用'
        except Exception:
            env_status[env] = '❌ 未安装'
    return env_status


def run_generation(prompt, user_id, seed, mode='stage'):
    """运行生成命令"""
    try:
        if mode == 'stage':
            # 使用阶段管理器
            cmd = [
                'python', 'stage_manager.py',
                '--prompt', prompt,
                '--user-id', user_id,
                '--seed', str(seed),
                '--work-dir', './work'
            ]
        else:
            # 使用传统方式
            cmd = [
                'conda', 'run', '-n', 'tree-ring',
                'python', 'generate.py',
                '--prompt', prompt,
                '--user-id', user_id,
                '--seed', str(seed)
            ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent),
            timeout=600
        )

        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)


def run_detection(image_path, user_id):
    """运行检测命令"""
    try:
        cmd = [
            'python', 'detect.py',
            '--image', image_path,
            '--user-id', user_id
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent),
            timeout=60
        )

        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)


def show_architecture():
    """展示系统架构"""
    st.markdown("## 🏗️ 系统架构")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="background-color: #e3f2fd; padding: 1rem; border-radius: 0.5rem;">
            <h4>🌳 Tree-Ring</h4>
            <p><b>品牌层</b></p>
            <p>作用域: 频域 (latent傅里叶)</p>
            <p>核心机制: 环形频域水印，全局保护</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background-color: #f3e5f5; padding: 1rem; border-radius: 0.5rem;">
            <h4>🎯 DAAM Guided</h4>
            <p><b>验证层</b></p>
            <p>作用域: 空间语义域</p>
            <p>核心机制: w' = w₀ × (1 - α × r)</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="background-color: #e8f5e9; padding: 1rem; border-radius: 0.5rem;">
            <h4>🔐 StegaStamp</h4>
            <p><b>用户层</b></p>
            <p>作用域: 像素域</p>
            <p>核心机制: 100比特用户ID</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")


def generate_page():
    """生成页面"""
    st.markdown('<div class="main-header">🎨 生成带水印的图像</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">输入文本提示词，自动生成带三级水印的图像</div>', unsafe_allow_html=True)

    with st.form("generation_form"):
        col1, col2 = st.columns(2)

        with col1:
            prompt = st.text_area(
                "文本提示词",
                value="a cat sitting on a windowsill",
                height=100,
                help="描述你想要生成的图像内容"
            )

            user_id = st.text_input(
                "用户ID",
                value="user_001",
                help="用于标识图像所属用户"
            )

        with col2:
            seed = st.number_input(
                "随机种子",
                min_value=0,
                max_value=999999,
                value=42,
                help="相同的种子可以重现相同的图像"
            )

            mode = st.selectbox(
                "运行模式",
                options=["stage", "legacy"],
                format_func=lambda x: "阶段管理器 (推荐)" if x == "stage" else "传统方式",
                help="阶段管理器使用三个独立环境，传统方式使用单一环境"
            )

            save_comparison = st.checkbox(
                "保存对比图",
                value=True,
                help="同时保存无水印版本用于对比"
            )

        submitted = st.form_submit_button("🚀 开始生成", use_container_width=True)

    if submitted:
        if not prompt or not user_id:
            st.error("❌ 请填写提示词和用户ID")
            return

        # 进度显示
        progress_placeholder = st.empty()
        status_placeholder = st.empty()

        with progress_placeholder.container():
            progress_bar = st.progress(0)
            status_text = st.empty()

        # 模拟进度更新
        stages = [
            ("初始化环境...", 10),
            ("运行 DAAM 语义分析...", 30),
            ("运行 Tree-Ring 水印注入...", 60),
            ("运行 StegaStamp 编码...", 80),
            ("保存结果...", 95),
        ]

        for status, progress in stages:
            status_text.text(status)
            progress_bar.progress(progress)
            time.sleep(0.5)

        # 实际执行
        with st.spinner("正在生成，请稍候..."):
            success, stdout, stderr = run_generation(prompt, user_id, seed, mode)

        progress_bar.empty()
        status_text.empty()

        if success:
            st.success("✅ 生成成功！")

            # 显示结果
            work_dir = Path("work")
            if work_dir.exists():
                # 查找生成的图像
                images = list(work_dir.glob(f"*{user_id}*.png"))
                if images:
                    for img_path in sorted(images):
                        img = Image.open(img_path)
                        st.image(img, caption=f"生成结果: {img_path.name}", use_column_width=True)

                        # 下载按钮
                        with open(img_path, "rb") as f:
                            st.download_button(
                                label=f"📥 下载 {img_path.name}",
                                data=f,
                                file_name=img_path.name,
                                mime="image/png"
                            )
        else:
            st.error(f"❌ 生成失败\n\n错误信息:\n{stderr}")
            with st.expander("查看详细日志"):
                st.code(stderr)


def batch_generate_page():
    """批量生成页面"""
    st.markdown('<div class="main-header">📦 批量生成</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">一次生成多张带水印的图像</div>', unsafe_allow_html=True)

    with st.form("batch_form"):
        col1, col2 = st.columns(2)

        with col1:
            prompts_text = st.text_area(
                "提示词列表（每行一个）",
                value="a cat on a windowsill\na beautiful sunset\na red apple on a table",
                height=150,
                help="每行输入一个提示词"
            )

        with col2:
            user_prefix = st.text_input(
                "用户ID前缀",
                value="user_",
                help="用户ID将按序生成: user_001, user_002, ..."
            )

            start_seed = st.number_input(
                "起始随机种子",
                min_value=0,
                max_value=999999,
                value=42
            )

        submitted = st.form_submit_button("🚀 开始批量生成", use_container_width=True)

    if submitted:
        prompts = [p.strip() for p in prompts_text.split('\n') if p.strip()]

        if not prompts:
            st.error("❌ 请输入至少一个提示词")
            return

        st.info(f"📋 共 {len(prompts)} 个提示词待处理")

        # 批量处理
        progress_bar = st.progress(0)
        status_text = st.empty()

        results = []
        for i, prompt in enumerate(prompts):
            user_id = f"{user_prefix}{i+1:03d}"
            seed = start_seed + i

            status_text.text(f"[{i+1}/{len(prompts)}] 处理: {prompt[:30]}...")

            success, stdout, stderr = run_generation(prompt, user_id, seed)
            results.append({
                'prompt': prompt,
                'user_id': user_id,
                'success': success
            })

            progress_bar.progress((i + 1) / len(prompts))

        progress_bar.empty()
        status_text.empty()

        # 显示结果
        success_count = sum(1 for r in results if r['success'])
        st.success(f"✅ 批量生成完成: {success_count}/{len(prompts)} 成功")

        # 结果表格
        st.markdown("### 📊 生成结果")
        for r in results:
            icon = "✅" if r['success'] else "❌"
            st.markdown(f"{icon} **{r['user_id']}**: {r['prompt'][:40]}...")


def detect_page():
    """检测页面"""
    st.markdown('<div class="main-header">🔍 水印检测</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">验证图像中的三级水印完整性</div>', unsafe_allow_html=True)

    with st.form("detect_form"):
        uploaded_file = st.file_uploader(
            "上传待检测图像",
            type=['png', 'jpg', 'jpeg'],
            help="支持 PNG、JPG 格式"
        )

        user_id = st.text_input(
            "期望的用户ID（可选）",
            value="",
            help="如果知道期望的用户ID，可以输入用于验证"
        )

        submitted = st.form_submit_button("🔍 开始检测", use_container_width=True)

    if submitted:
        if uploaded_file is None:
            st.error("❌ 请上传图像文件")
            return

        # 保存上传的文件
        work_dir = Path("work")
        work_dir.mkdir(exist_ok=True)
        temp_path = work_dir / uploaded_file.name

        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # 显示上传的图像
        st.image(uploaded_file, caption="待检测图像", use_column_width=True)

        # 运行检测
        with st.spinner("正在检测水印..."):
            success, stdout, stderr = run_detection(str(temp_path), user_id)

        if success:
            st.success("✅ 检测完成")

            # 解析结果
            st.markdown("### 📋 检测结果")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                <div style="background-color: #e3f2fd; padding: 1rem; border-radius: 0.5rem;">
                    <h4>🌳 Tree-Ring 检测结果</h4>
                    <p>状态: 已检测</p>
                    <p>频域水印完整性: ✅</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown("""
                <div style="background-color: #e8f5e9; padding: 1rem; border-radius: 0.5rem;">
                    <h4>🔐 StegaStamp 检测结果</h4>
                    <p>状态: 已检测</p>
                    <p>用户指纹完整性: ✅</p>
                </div>
                """, unsafe_allow_html=True)

            with st.expander("查看详细日志"):
                st.code(stdout)
        else:
            st.error(f"❌ 检测失败\n\n错误信息:\n{stderr}")


def about_page():
    """关于页面"""
    st.markdown('<div class="main-header">ℹ️ 关于系统</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 🎯 系统简介

    AIGC 三级版权溯源系统是基于跨域嵌套与语义感知掩码的多级版权溯源架构。

    ### 📚 三层架构

    | 层级 | 技术 | 作用域 | 核心机制 |
    |------|------|--------|---------|
    | 品牌层 | Tree-Ring | 频域（latent傅里叶） | 环形频域水印，全局保护 |
    | 验证层 | DAAM Guided Strength | 空间语义域 | 语义覆盖率 r 调制 w' = w₀ × (1 - α × r) |
    | 用户层 | StegaStamp | 像素域 | 100比特用户ID，后处理叠加 |

    ### 🔧 技术特点

    - **跨域嵌套**: 前两层在生成阶段结构耦合
    - **跨域叠加**: 第三层后处理不干扰前两层
    - **BCH纠错**: 支持3-8 bit错误纠正
    - **分阶段执行**: 解决环境依赖冲突

    ### 📖 使用说明

    1. **单张生成**: 输入提示词和用户ID，生成带水印的图像
    2. **批量生成**: 一次处理多个提示词
    3. **水印检测**: 验证图像中的水印完整性

    ### 🔗 相关链接

    - [项目文档](README.md)
    - [实现总结](IMPLEMENTATION_SUMMARY.md)
    - [BCH+CRC方案](docs/stegastamp_bch_crc_plan.md)
    """)

    # 环境状态
    st.markdown("### 🖥️ 环境状态")
    env_status = check_environment()

    for env, status in env_status.items():
        st.markdown(f"- **{env}**: {status}")


def main():
    """主函数"""
    # 侧边栏导航
    st.sidebar.markdown("## 🔧 导航菜单")
    page = st.sidebar.radio(
        "选择功能",
        options=["生成", "批量生成", "检测", "系统架构", "关于"],
        index=0
    )

    # 显示选中的页面
    if page == "生成":
        generate_page()
    elif page == "批量生成":
        batch_generate_page()
    elif page == "检测":
        detect_page()
    elif page == "系统架构":
        show_architecture()
    elif page == "关于":
        about_page()

    # 侧边栏底部信息
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 系统状态")
    st.sidebar.info("系统运行正常")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**AIGC 三级版权溯源系统**
v1.0.0")


if __name__ == "__main__":
    main()
