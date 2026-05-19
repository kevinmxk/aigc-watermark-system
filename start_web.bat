@echo off
REM 启动 AIGC 三级版权溯源系统 Web 界面

echo ==========================================
echo   AIGC 三级版权溯源系统 - Web 界面
echo ==========================================
echo.

REM 检查 Streamlit 是否安装
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo 正在安装 Streamlit...
    pip install streamlit -i https://pypi.tuna.tsinghua.edu.cn/simple
)

echo 启动 Web 界面...
echo 请在浏览器中访问: http://localhost:8501
echo.

streamlit run web_ui.py

pause
