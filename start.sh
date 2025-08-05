#!/bin/bash

# 脚本开始执行时打印一条日志，方便调试
echo "Starting the Autoaim Service Script..."

# --- 关键步骤一: 加载root用户的系统级环境变量 ---
# 这等同于您'su'后获得的环境，为NPU/AI库提供路径 (如LD_LIBRARY_PATH)
# 这一步对于模型加载至关重要！
if [ -f /root/.bashrc ]; then
    source /root/.bashrc
    echo "Sourced /root/.bashrc"
elif [ -f /root/.profile ]; then
    source /root/.profile
    echo "Sourced /root/.profile"
else
    echo "Warning: Could not find .bashrc or .profile for root."
fi

# --- 关键步骤二: 激活Python虚拟环境 ---
# 这完全复刻了您的 "source ./activate" 操作
echo "Activating Python virtual environment..."
source /home/HwHiAiUser/PyCharmMiscProject/.venv1/bin/activate

# --- 关键步骤三: 切换到项目的工作目录 ---
# 确保脚本能找到所有相关文件
cd /home/HwHiAiUser/PyCharmMiscProject
echo "Changed directory to $(pwd)"

# --- 最后一步: 运行您的Python主程序 ---
# 因为虚拟环境已激活, 直接调用'python'即可
# systemd会捕获所有print()输出
echo "Starting the main Python script (main.py)..."
python serial_listener.py
