# config.py
import numpy as np

# Kalman Filter Parameters
KF_DT = 1.0  # Time step
KF_PROCESS_NOISE_STD = 2.0  # Standard deviation of process noise
KF_MEASUREMENT_NOISE_STD = 5.0  # Standard deviation of measurement noise

# Tracking Parameters
IOU_MATCHING_THRESHOLD = 0.3  # IOU threshold for matching tracks with detections
MAX_FRAMES_SINCE_UPDATE = 30  # Max frames to keep a track without updates
MIN_HITS_TO_ACTIVATE = 3  # Min hits to activate a track (make it non-tentative)
AIM_PREDICTION_FRAMES = 7  # Number of frames to predict aim point ahead

# Class IDs for Auto Aiming and specific tracking
AUTO_AIM_CLASSES = {0, 2}  # Class IDs that trigger aiming logic
TARGET_TRACKING_MODEL_CLASSES = {0, 2} # Model class IDs to initiate tracking for

# Distance Estimation Parameters
PIXEL_HEIGHT_AT_CALIBRATION_DISTANCE = 45  # Pixel height of a known object at a known distance
CALIBRATION_DISTANCE_METERS = 0.3  # The known distance in meters

# Color Detection (HSV Ranges)
def define_color_ranges_hsv():
    """Defines HSV color ranges for red and blue."""
    # Red color
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