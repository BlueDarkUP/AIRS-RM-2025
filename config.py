import numpy as np

# Kalman Filter Parameters
KF_DT = 1.0  # Time step
KF_PROCESS_NOISE_STD = 2.0  # Standard deviation of process noise
KF_MEASUREMENT_NOISE_STD = 5.0  # Standard deviation of measurement noise

# Tracking Parameters
IOU_MATCHING_THRESHOLD = 0.3  # IOU threshold for matching tracks with detections
MAX_FRAMES_SINCE_UPDATE = 12# Max frames to keep a track without updates
MIN_HITS_TO_ACTIVATE = 3  # Min hits to activate a track (make it non-tentative)
AIM_PREDICTION_FRAMES = 8  # Number of frames to predict aim point ahead

# Class IDs for Auto Aiming and specific tracking
AUTO_AIM_CLASSES = {0, 2}  # Class IDs that trigger aiming logic (RedArmor, BlueArmor)
TARGET_TRACKING_MODEL_CLASSES = {0, 2} # Model class IDs to initiate tracking for (RedArmor, BlueArmor)

# Distance Estimation Parameters
PIXEL_HEIGHT_AT_CALIBRATION_DISTANCE = 45  # Pixel height of a known object at a known distance
CALIBRATION_DISTANCE_METERS = 0.3  # The known distance in meters

# Kalman Filter Parameters
MIN_VELOCITY_FOR_PREDICTION_SQ = 0.5 # 目标速度平方的最小阈值。低于此阈值，预瞄点将直接指向目标当前中心，
                                    # 避免在目标静止或慢速移动时出现不必要的预测跳动。
                                    # (例如，0.5表示速度大小约0.7像素/帧或像素/秒)

# Target Selection Weights and Thresholds
NEW_TARGET_SCORE_PREFERENCE = 0.35 # 新目标的分数必须高于当前锁定目标分数的35%才进行切换

# Scoring Weights (summing to 1 for clarity in normalization)
# These weights determine the relative importance of each factor in the overall score.
WEIGHT_DISTANCE = 0.4   # 距离权重排位第二
WEIGHT_ANGLE = 0.3      # 角度偏移权重排第三
WEIGHT_SIZE = 0.3       # 装甲板大小权重排第四

# Normalization constants for scoring
# These values define the range over which scores are normalized from 0 to 1.
# Values outside this range will typically result in a 0 or 1 score after clamping.
MAX_NORMALIZATION_DISTANCE_METERS = 5.0 # Distance at which score becomes 0 (e.g., too far)
MIN_NORMALIZATION_DISTANCE_METERS = 0.1 # Distance at which score becomes 1 (e.g., very close)
                                        # Use a small non-zero value to avoid division by zero or extreme scores at 0m.

# 这两个常量现在在 detector.py 中作为固定的归一化值定义，不再在此处动态设置。
# MAX_NORMALIZATION_ANGLE_PIXELS = 0.0 # Max possible pixel distance from center to corner
# MAX_NORMALIZATION_AREA_PIXELS = 0.0  # Max possible area (frame_width * frame_height)

# Gimbal PID Constants (Separate for Yaw and Pitch)
# These initial values are used for trackbar default positions.
# They can be tuned via the UI trackbars during runtime.
GIMBAL_YAW_KP_INIT = 220
GIMBAL_YAW_KI_INIT = 0.1
GIMBAL_YAW_KD_INIT = 85
GIMBAL_YAW_ALPHA_D_FILTER_INIT = 0 # Alpha for derivative filter
GIMBAL_YAW_INTEGRAL_LIMIT_INIT = 5000 # Limit for integral windup

GIMBAL_PITCH_KP_INIT = 120
GIMBAL_PITCH_KI_INIT = 0.1
GIMBAL_PITCH_KD_INIT = 60
GIMBAL_PITCH_ALPHA_D_FILTER_INIT = 0 # Alpha for derivative filter
GIMBAL_PITCH_INTEGRAL_LIMIT_INIT = 5000 # Limit for integral windup

# Color Detection (HSV Ranges)
def define_color_ranges_hsv():
    """Defines HSV color ranges for red and blue."""
    # Red color (two ranges because red wraps around the 0/180 mark in HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    # Blue color
    lower_blue = np.array([100, 100, 50])
    upper_blue = np.array([130, 255, 255])
    return {
        "red": [(lower_red1, upper_red1), (lower_red2, upper_red2)],
        "blue": [(lower_blue, upper_blue)]
    }

COLOR_RANGES_HSV = define_color_ranges_hsv()

# Model and UI configuration (can be overridden in main.py if needed)
DEFAULT_MODEL_PATH = "./16.om"
DEFAULT_LABEL_PATH = './labels.txt'
DEFAULT_INFER_CONFIG = {'conf_thres': 0.35, 'iou_thres': 0.45, 'input_shape': [640, 640]}
DEFAULT_EXPOSURE_FACTOR = 1.0