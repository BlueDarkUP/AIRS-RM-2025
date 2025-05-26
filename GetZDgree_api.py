import math


def calculate_object_rotation(bbox_width: float, bbox_height: float) -> float:
    epsilon = 1e-6  # 用于处理接近零的维度

    is_width_effectively_zero = bbox_width < epsilon
    is_height_effectively_zero = bbox_height < epsilon

    if is_width_effectively_zero and is_height_effectively_zero:
        print("宽高度都接近于零。")
        return -1.0  # 无效输入
    if is_width_effectively_zero:
        return 90.0
    if is_height_effectively_zero:
        return 0.0
    object_AR_L_div_S = 5.0 / 2.8
    if bbox_width >= bbox_height:
        bbox_long_side = bbox_width
        bbox_short_side = bbox_height
    else:
        bbox_long_side = bbox_height
        bbox_short_side = bbox_width
    bbox_AR_long_div_short = bbox_long_side / bbox_short_side
    numerator = object_AR_L_div_S - bbox_AR_long_div_short
    denominator = (bbox_AR_long_div_short * object_AR_L_div_S) - 1.0
    if abs(numerator) < epsilon:
        alpha_rad = 0.0
    else:
        if abs(denominator) < epsilon:
            print("分母接近零")
            alpha_rad = math.atan(numerator / denominator)
        else:
            alpha_rad = math.atan(numerator / denominator)

    alpha_deg = math.degrees(alpha_rad)
    alpha_deg = max(0.0, min(alpha_deg, 45.0))
    if bbox_width >= bbox_height:
        final_rotation_angle_deg = alpha_deg
    else:
        final_rotation_angle_deg = 90.0 - alpha_deg
    final_rotation_angle_deg = max(0.0, min(final_rotation_angle_deg, 90.0))

    return final_rotation_angle_deg
