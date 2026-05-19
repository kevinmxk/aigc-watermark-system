#!/bin/bash
# 启动 AIGC 三级版权溯源系统 Web 界面

echo "=========================================="
echo "  AIGC 三级版权溯源系统 - Web 界面"
echo "=========================================="
echo ""

# 检查 Streamlit 是否安装
if ! python -c "import streamlit" 2>/dev/null; then
    echo "正在安装 Streamlit..."
    pip install streamlit
fi

echo "启动 Web 界面..."
echo "请在浏览器中访问: http://localhost:8501"
echo ""

streamlit run web_ui.py
