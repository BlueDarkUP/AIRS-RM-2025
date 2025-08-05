import numpy as np

# --- 追踪与开火逻辑 ---
STABILITY_THRESHOLD = 5         # 连续识别N帧后视为稳定目标
FIRE_COOLDOWN_SECONDS = 0.13       # 开火冷却时间(秒)
DEFAULT_EXPOSURE_FACTOR = 1.0   # 默认曝光增益

# --- 卡尔曼滤波器参数 ---
KF_DT = 1.0                       # 时间步长
KF_PROCESS_NOISE_STD = 2.0        # 过程噪声标准差
KF_MEASUREMENT_NOISE_STD = 5.0    # 测量噪声标准差

# --- 追踪器参数 ---
IOU_MATCHING_THRESHOLD = 0.3      # 追踪匹配的IOU阈值
MAX_FRAMES_SINCE_UPDATE = 12      # 目标丢失后保留的最大帧数
MIN_HITS_TO_ACTIVATE = 3          # 目标连续识别N次后激活追踪
AIM_PREDICTION_FRAMES = 8        # 向前预测N帧进行瞄准

# --- 类别ID设置 ---
AUTO_AIM_CLASSES = {0, 2}               # 需要自动瞄准的类别ID (例如: 红/蓝装甲板)
TARGET_TRACKING_MODEL_CLASSES = {0, 2}  # 需要进行追踪的类别ID

# --- 距离估算参数 ---
PIXEL_HEIGHT_AT_CALIBRATION_DISTANCE = 45 # 标定距离下目标像素高度
CALIBRATION_DISTANCE_METERS = 0.3         # 标定距离(米)

# --- 预测逻辑参数 ---
MIN_VELOCITY_FOR_PREDICTION_SQ = 0.5    # 启用运动预测的最小速度(平方值)，低于此值不预判，防抖动

# --- 目标选择权重 ---
NEW_TARGET_SCORE_PREFERENCE = 0.35      # 新目标得分高出此比例才切换

# 目标评分权重 (总和建议为1)
WEIGHT_DISTANCE = 0.4   # 距离权重
WEIGHT_ANGLE = 0.3      # 角度权重
WEIGHT_SIZE = 0.3       # 尺寸权重

# 评分归一化参数
MAX_NORMALIZATION_DISTANCE_METERS = 5.0 # 最大归一化距离(米)
MIN_NORMALIZATION_DISTANCE_METERS = 0.1 # 最小归一化距离(米)

# (以下参数已移至 detector.py 中定义)
# MAX_NORMALIZATION_ANGLE_PIXELS = 0.0
# MAX_NORMALIZATION_AREA_PIXELS = 0.0

# --- 云台PID参数 (Yaw 和 Pitch 轴) ---
# Yaw 轴
GIMBAL_YAW_KP_INIT = 300
GIMBAL_YAW_KI_INIT = 3
GIMBAL_YAW_KD_INIT = 400
GIMBAL_YAW_KF_INIT = 350                  # Yaw轴前馈增益
GIMBAL_YAW_ALPHA_D_FILTER_INIT = 0        # 微分项滤波系数
GIMBAL_YAW_INTEGRAL_LIMIT_INIT = 50000     # 积分限幅

# Pitch 轴
GIMBAL_PITCH_KP_INIT = 230
GIMBAL_PITCH_KI_INIT = 5
GIMBAL_PITCH_KD_INIT = 400
GIMBAL_PITCH_KF_INIT = 150                # Pitch轴前馈增益
GIMBAL_PITCH_ALPHA_D_FILTER_INIT = 0      # 微分项滤波系数
GIMBAL_PITCH_INTEGRAL_LIMIT_INIT = 50000   # 积分限幅

# --- 颜色检测(HSV) ---
def define_color_ranges_hsv():
    """定义红色和蓝色的HSV范围"""
    # 红色 (因环绕需两个范围)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    # 蓝色
    lower_blue = np.array([100, 100, 50])
    upper_blue = np.array([130, 255, 255])
    return {
        "red": [(lower_red1, upper_red1), (lower_red2, upper_red2)],
        "blue": [(lower_blue, upper_blue)]
    }

COLOR_RANGES_HSV = define_color_ranges_hsv()

# --- 模型与推理配置 ---
DEFAULT_MODEL_PATH = "16.om"
DEFAULT_LABEL_PATH = 'labels.txt'
DEFAULT_INFER_CONFIG = {'conf_thres': 0.35, 'iou_thres': 0.45, 'input_shape': [640, 640]}

# --- 目标ID白名单 ---
# 若非空，则只攻击列表内的ID。例: {0, 1} 表示只攻击0号和1号目标。
# 设置为 set() 则禁用白名单，攻击任何ID的目标。
TARGET_ID_WHITELIST = {2} # 在此设置要攻击的特定ID RedArmor2  BlueArmor0