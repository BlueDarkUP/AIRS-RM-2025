# utils.py
import cv2
import numpy as np
from det_utils import letterbox # Assuming letterbox is in det_utils

def calculate_iou(box1, box2):
    """
    Calculates Intersection over Union (IoU) between two bounding boxes.
    Boxes are in [x1, y1, x2, y2] format.
    """
    x1_i = max(box1[0], box2[0])
    y1_i = max(box1[1], box2[1])
    x2_i = min(box1[2], box2[2])
    y2_i = min(box1[3], box2[3])

    inter_area = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
    if inter_area == 0:
        return 0.0

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def check_dominant_color_in_roi(roi_bgr, target_color_name, color_ranges, min_pixel_percentage=0.1):
    """
    Checks if a target color is dominant in the Region of Interest (ROI).
    """
    if roi_bgr is None or roi_bgr.size == 0 or roi_bgr.shape[0] < 5 or roi_bgr.shape[1] < 5:
        return False # ROI too small or invalid

    hsv_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    total_mask = np.zeros(hsv_roi.shape[:2], dtype="uint8")

    for lower, upper in color_ranges[target_color_name]:
        mask_part = cv2.inRange(hsv_roi, lower, upper)
        total_mask = cv2.bitwise_or(total_mask, mask_part)

    color_pixel_count = cv2.countNonZero(total_mask)
    total_pixels_in_roi = roi_bgr.shape[0] * roi_bgr.shape[1]

    if total_pixels_in_roi == 0: return False
    percentage = color_pixel_count / total_pixels_in_roi
    return percentage >= min_pixel_percentage


def adjust_exposure_hsv(image, factor):
    """
    Adjusts image exposure using HSV color space.
    Factor > 1 increases exposure, Factor < 1 decreases exposure.
    """
    if factor == 1.0: return image # No change needed

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)

    v_float = v.astype(np.float32)
    v_adjusted_float = np.clip(v_float * factor, 0, 255) # Apply factor and clip to 0-255
    v_adjusted = v_adjusted_float.astype(np.uint8)

    final_hsv = cv2.merge((h, s, v_adjusted))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)


def preprocess_image(image, cfg, bgr2rgb=True):
    """
    Preprocesses an image for model inference.
    Uses letterbox resizing.
    """
    img, scale_ratio, pad_size = letterbox(image, new_shape=cfg['input_shape'])
    if bgr2rgb:
        img = img[:, :, ::-1]  # BGR to RGB
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = np.ascontiguousarray(img, dtype=np.float32)
    return img, scale_ratio, pad_size


def get_labels_from_txt(path):
    """
    Loads class labels from a text file.
    Each line in the file is a label, and its index is the class ID.
    """
    labels_dict = dict()
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for cat_id, label in enumerate(f.readlines()):
                labels_dict[cat_id] = label.strip()
    except FileNotFoundError:
        print(f"Error: Label file not found at {path}")
        return None
    except Exception as e:
        print(f"Error reading label file {path}: {e}")
        return None
    return labels_dict