# RM-Hunter: 基于昇腾 Atlas 200I DK 的 RoboMaster 智能自瞄系统

![RoboMaster](https://img.shields.io/badge/RoboMaster-EP/Core-blue.svg)![Hardware](https://img.shields.io/badge/Hardware-Atlas%20200I%20DK-orange.svg)![Framework](https://img.shields.io/badge/Framework-PyTorch/MindSpore-brightgreen.svg)![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📖 项目简介

**RM-Hunter** 是一个专为 RoboMaster 机器人设计的、运行在华为昇腾 Atlas 200I DK A2 开发者套件上的高级自主瞄准解决方案。本项目通过串口接收指令，能够实时检测并攻击指定颜色的装甲板，实现了从目标检测、追踪、筛选到云台控制的全流程闭环。

该系统不仅集成了基于YOLO的深度学习检测模型，还通过卡尔曼滤波器（Kalman Filter）对目标进行稳定追踪和运动预测，并设计了一套智能评分系统来选择最优攻击目标。

## ✨ 核心特性

- **硬件平台**: 完全基于华为**昇腾 Atlas 200I DK A2** NPU进行加速，性能强大。
- **指令式启动**: 通过**串口通信**接收外部指令（如来自裁判系统或遥控器），启动并切换攻击任务（攻击红方或蓝方）。
- **实时目标检测**: 使用高性能的 `.om` 格式AI模型，实时检测画面中的装甲板。
- **智能追踪与预测**:
    - **卡尔曼滤波器(Kalman Filter)**: 对目标位置进行平滑滤波和速度估计，有效抑制画面抖动，实现稳定追踪。
    - **运动预测**: 根据目标速度预测下一时刻的位置，提前量瞄准，提高命中率。
- **颜色动态识别**: 在锁定目标后，通过HSV色彩空间分析，动态确认装甲板的**红/蓝**属性，避免误识别。
- **高级目标选择**:
    - **综合评分系统**: 结合目标的**距离、角度、面积**三大要素，为每个潜在目标计算得分。
    - **粘性锁定**: 优先锁定当前目标，只有当新目标的评分显著高于当前目标时才进行切换，避免云台频繁晃动。
- **精准云台控制**:
    - **PID + 前馈控制**: 结合PID反馈控制和基于目标速度的前馈控制，使云台能够快速、精准地跟上目标。
- **健壮的状态机**: 拥有**巡逻(IDLE)**、**搜索(SEARCHING)**、**追踪(TRACKING)**和**惯性追踪(COASTING)**等清晰的状态，使机器人在不同场景下（如目标丢失）都能做出合理响应。

## 🔧 项目结构

以下是项目的核心文件结构和说明：

```
.
├── 16.om                   # 核心物体检测模型文件
├── labels.txt              # 模型对应的标签文件
├── start.sh                # (核心) 主启动脚本
├── serial_listener.py      # (核心) 串口监听与任务分发服务
├── main_shoot_blue.py      # (核心) 攻击蓝色目标的任务脚本
├── main_shoot_red.py       # (核心) 攻击红色目标的任务脚本
├── autoaim.py              # (核心) 自瞄系统主模块，封装了所有功能接口
├── detector.py             # (核心) 检测与追踪器，负责目标处理、选择和绘制
├── config.py               # (核心) 全局配置文件(PID参数、路径、阈值等)
├── tracker.py              # (核心) 追踪器类，定义了卡尔曼滤波器和目标状态
├── utils.py                # 视觉和通用工具函数
├── det_utils.py            # 底层检测工具函数 (如NMS)
├── Calcu_distance_api.py   # 距离估算模块
└── GetZDgree_api.py        # 角度估算模块
```

## 🚀 安装与部署

### 1. 硬件要求

- **计算平台**: 华为昇腾 Atlas 200I DK A2
- **机器人**: 大疆 RoboMaster EP 或 EP Core
- **摄像头**: 标准USB摄像头

### 2. 环境准备

- 确保您的 Atlas 200I DK 已正确安装昇腾的**CANN工具包**，并已配置好系统环境变量。
- 本项目推荐在 Python 虚拟环境中运行。

### 3. 依赖安装

创建一个 `requirements.txt` 文件，内容如下：

```txt
numpy
opencv-python
pyserial
scipy
```

然后执行以下命令安装依赖：

```bash
# 激活你的Python虚拟环境
# source /path/to/your/venv/bin/activate

pip install -r requirements.txt
```

### 4. 模型与配置

- 将 `16.om` 模型文件和 `labels.txt` 标签文件放置在项目根目录。
- **(重要)** 根据您的实际情况修改 `config.py` 文件中的参数，例如：
    - `DEFAULT_MODEL_PATH` 和 `DEFAULT_LABEL_PATH`: 模型和标签文件的路径。
    - `GIMBAL_YAW_KP_INIT`, `GIMBAL_PITCH_KP_INIT` 等: 云台PID参数，需要根据机器人实际情况进行调试。
    - `PIXEL_HEIGHT_AT_CALIBRATION_DISTANCE`, `CALIBRATION_DISTANCE_METERS`: 用于距离估算的标定参数。

### 5. 启动程序

1.  **重命名启动脚本**:
    ```bash
    mv start.sh.txt start.sh
    ```
2.  **授予执行权限**:
    ```bash
    chmod +x start.sh
    ```
3.  **以管理员权限运行**:
    由于脚本需要操作底层硬件和服务，必须使用 `sudo` 运行。
    ```bash
    sudo ./start.sh
    ```
    脚本将自动激活虚拟环境，加载环境变量，并启动 `serial_listener.py` 服务。

## 🎮 使用说明

1.  **连接串口**: 将您的串口设备（如单片机、USB转TTL模块）连接到 Atlas 板的 `/dev/ttyAMA0` 接口。串口配置为 **115200波特率, 8-N-1**。
2.  **发送指令**:
    - **攻击蓝色方**: 通过串口发送字符串 "**1**"。
    - **攻击红色方**: 通过串口发送字符串 "**2**"。
3.  **观察机器人**: 程序启动后，机器人将进入**巡逻**状态。当接收到有效指令后，它会立即开始执行指定的攻击任务，自动寻找并射击目标。任务完成后，程序将自动退出。

## 💡 未来工作

- [ ] **集成底盘控制**: 实现基于目标位置的底盘跟随或躲避逻辑。
- [ ] **弹道补偿**: 加入更精确的弹道模型，根据目标距离补偿子弹重力下坠。
- [ ] **多目标协同**: 在多机器人场景下，实现目标分配与协同攻击。
- [ ] **Web监控界面**: 开发一个简单的Web界面，用于实时查看摄像头画面、目标信息和系统状态。

## 🙏 致谢

本项目的状态机和多线程控制架构的设计灵感，来源于作者的另一个开源项目：[FLL-AquaHunter](https://github.com/BlueDarkUP/FLL-AquaHunter)。该项目是为 FIRST LEGO League (FLL) 机器人竞赛设计的，其中模块化的任务管理和事件驱动的思想在 RM-Hunter 中得到了继承和发展，以适应更复杂的实时对抗场景。

## 🤝 贡献

欢迎对本项目感兴趣的开发者提交 Pull Request 或在 Issues 中报告问题和提出建议。您的贡献将使这个项目变得更好！

## 📄 许可证

本项目采用 [MIT License](https://opensource.org/licenses/MIT) 开源。