# detector.py
import cv2
import numpy as np
import torch
from time import time
from scipy.optimize import linear_sum_assignment

# Assuming these are provided external utilities/APIs
# 你需要确保这些文件在你的项目路径中
from det_utils import nms, scale_coords
from GetZDgree_api import calculate_object_rotation
from Calcu_distance_api import estimate_distance

# Project-specific imports
import config as cfg
from utils import calculate_iou, preprocess_image, check_dominant_color_in_roi
from tracker import Track  # Assuming Track class is in tracker.py

g_next_track_id = 0
g_locked_target_track_id = None  # Global variable to store the track_id of the currently locked target

# --- 过滤阈值 ---
# 定义置信度阈值：低于此值的检测结果将不被处理和显示
CONFIDENCE_THRESHOLD_FOR_DISPLAY = 0.7
# 定义角度阈值：绝对值大于此值的目标将不被显示（即使被追踪）
ANGLE_THRESHOLD_DEGREES_FOR_DISPLAY = 70.0
# 定义距离阈值：小于此值的目标将不被显示
DISTANCE_THRESHOLD_FOR_DISPLAY = 0.45  # meters

# --- 归一化坐标系下的常量 ---
# 最大归一化距离：从画面中心(0.5, 0.5)到任意一个角的欧几里得距离
# 在0-1归一化坐标系中，中心为(0.5,0.5)，角点如(0,0), (0,1), (1,0), (1,1)。
# 距离中心最远的点是四个角点，例如(0,0)到(0.5,0.5)的距离是sqrt((0.5-0)^2 + (0.5-0)^2) = sqrt(0.25+0.25) = sqrt(0.5)
MAX_NORMALIZED_CENTER_DIST = np.sqrt(0.5 ** 2 + 0.5 ** 2)  # 約 0.707
# 最大归一化面积：在0-1归一化坐标系中，最大面积为 1 * 1 = 1.0
MAX_NORMALIZED_AREA = 1.0


def calculate_track_score(track, frame_width, frame_height):
    """
    Calculates a combined score for a track based on multiple criteria.
    Scores are normalized between 0 and 1 before weighting.
    Uses the new normalized (0-1) coordinate system with top-left origin.
    """
    # 将目标的像素中心坐标转换为0-1归一化坐标
    norm_center_x = track.center_x / frame_width
    norm_center_y = track.center_y / frame_height

    # 1. Distance Score (higher is better, closer is better)
    dist_score = 0.0
    if track.estimated_distance is not None:
        # Normalize: 1.0 at MIN_NORMALIZATION_DISTANCE_METERS, 0.0 at MAX_NORMALIZATION_DISTANCE_METERS
        if cfg.MAX_NORMALIZATION_DISTANCE_METERS > cfg.MIN_NORMALIZATION_DISTANCE_METERS:
            dist_score = 1.0 - (track.estimated_distance - cfg.MIN_NORMALIZATION_DISTANCE_METERS) / \
                         (cfg.MAX_NORMALIZATION_DISTANCE_METERS - cfg.MIN_NORMALIZATION_DISTANCE_METERS)
            dist_score = np.clip(dist_score, 0.0, 1.0)  # Clamp between 0 and 1
        elif track.estimated_distance <= cfg.MIN_NORMALIZATION_DISTANCE_METERS:
            dist_score = 1.0  # Very close targets get max score
        # Else (if track.estimated_distance > MAX_NORMALIZATION_DISTANCE_METERS), dist_score remains 0.0

    # 2. Angle Offset Score (higher is better, closer to normalized center (0.5,0.5) is better)
    angle_score = 0.0
    if MAX_NORMALIZED_CENTER_DIST > 0:  # 避免除以零
        # 计算目标中心到画面归一化中心(0.5, 0.5)的欧几里得距离
        center_dist_in_norm_coords = np.sqrt((norm_center_x - 0.5) ** 2 + (norm_center_y - 0.5) ** 2)
        angle_score = 1.0 - (center_dist_in_norm_coords / MAX_NORMALIZED_CENTER_DIST)
        angle_score = np.clip(angle_score, 0.0, 1.0)  # 钳制在0到1之间

    # 3. Size Score (higher is better, larger area is better)
    # 计算归一化后的目标框宽度、高度和面积
    norm_track_width = track.width / frame_width
    norm_track_height = track.height / frame_height
    norm_track_area = norm_track_width * norm_track_height

    size_score = 0.0
    if MAX_NORMALIZED_AREA > 0:  # 避免除以零 (MAX_NORMALIZED_AREA 恒为 1.0)
        size_score = norm_track_area / MAX_NORMALIZED_AREA  # 等同于直接使用 norm_track_area
        size_score = np.clip(size_score, 0.0, 1.0)  # 钳制在0到1之间

    # Combined weighted score
    combined_score = (cfg.WEIGHT_DISTANCE * dist_score +
                      cfg.WEIGHT_ANGLE * angle_score +
                      cfg.WEIGHT_SIZE * size_score)

    # Store individual and combined score in the track object for debugging/reference
    track.last_dist_score = dist_score
    track.last_angle_score = angle_score
    track.last_size_score = size_score
    track.score = combined_score  # Update track's score attribute

    return combined_score


def select_best_target(active_tracks, current_locked_track_id, frame_width, frame_height):
    """
    Selects the best target from active tracks based on weighted criteria and stickiness.
    Returns the track_id of the selected target.
    """
    best_candidate_id = None
    best_candidate_score = -1.0
    locked_track_obj = None  # Will store the actual track object if it's currently locked
    locked_track_score_unmodified = -1.0  # Score of the locked target without any stickiness bonus

    # Define the pool of candidates for selection based on whether a target is currently locked
    candidates_for_selection = []
    if current_locked_track_id is None:
        # If no target is locked, consider ALL valid (non-gimbal) tracks, even tentative ones.
        # This speeds up acquisition of a new target.
        candidates_for_selection = [t for t in active_tracks if t.class_id in cfg.AUTO_AIM_CLASSES]
    else:
        # If a target IS locked, only consider *confirmed* tracks for potential switching.
        # This ensures stability once a target is locked.
        candidates_for_selection = [t for t in active_tracks if
                                    not t.is_tentative and t.class_id in cfg.AUTO_AIM_CLASSES]

    if not candidates_for_selection:
        return None  # No eligible targets to select from

    # Step 1: Calculate scores for all candidates and find the overall best
    for track in candidates_for_selection:
        # 调用更新后的 calculate_track_score，传递帧的宽度和高度
        score = calculate_track_score(track, frame_width, frame_height)

        # Identify the currently locked target if it's among the candidates
        if track.track_id == current_locked_track_id:
            locked_track_obj = track
            locked_track_score_unmodified = score

        # Find the overall best scoring candidate (without considering stickiness yet)
        if score > best_candidate_score:
            best_candidate_score = score
            best_candidate_id = track.track_id

    # Step 2: Apply stickiness logic if there's a previously locked target
    if locked_track_obj is not None:
        # If the best candidate found is already the locked target, stick with it
        if best_candidate_id == locked_track_obj.track_id:
            return locked_track_obj.track_id

        # If there's a new best candidate, check if its score is significantly higher
        # The condition: new_best_score > locked_score_unmodified * (1 + NEW_TARGET_SCORE_PREFERENCE)
        if best_candidate_score > locked_track_score_unmodified * (1 + cfg.NEW_TARGET_SCORE_PREFERENCE):
            # Switch to the new best candidate if it meets the preference threshold
            return best_candidate_id
        else:
            # Otherwise, stick with the current locked target
            return locked_track_obj.track_id

    # Step 3: If no target was previously locked, or the locked target was lost/ineligible,
    # simply return the highest scoring eligible candidate from the current `candidates_for_selection` pool.
    return best_candidate_id


def process_frame_with_tracking(model, frame_raw_bgr, frame_for_detection_bgr, active_tracks, labels_dict, infer_cfg):
    """
    处理单个帧：检测对象、更新轨迹并准备显示。
    返回: processed_display_frame, active_tracks, locked_target_info (一个包含目标信息的字典, 或 None)
    """
    global g_next_track_id, g_locked_target_track_id
    display_frame = frame_for_detection_bgr.copy()
    raw_img_h, raw_img_w = frame_for_detection_bgr.shape[:2]

    # 初始化新的返回变量，如果未锁定目标则为None
    locked_target_info_for_return = None

    # 1. Preprocess image for detection
    img_processed, scale_ratio, pad_size = preprocess_image(frame_for_detection_bgr, infer_cfg)
    img_processed_normalized = img_processed / 255.0

    # 2. Model Inference
    t_start_infer = time()
    raw_model_outputs = model.infer([img_processed_normalized])
    t_end_infer = time()
    infer_time = t_end_infer - t_start_infer

    # 3. Post-process Detections and apply Filters
    current_detections_for_tracking = []
    if raw_model_outputs and raw_model_outputs[0] is not None:
        output_np = raw_model_outputs[0]
        output_for_nms = torch.from_numpy(output_np)
        detections_from_nms_list = nms(output_for_nms, conf_thres=infer_cfg["conf_thres"],
                                       iou_thres=infer_cfg["iou_thres"])
        if detections_from_nms_list and detections_from_nms_list[0] is not None and detections_from_nms_list[
            0].numel() > 0:
            pred_all_tensor = detections_from_nms_list[0]
            pred_all_np = pred_all_tensor.cpu().numpy()
            if pred_all_np.shape[0] > 0:
                scale_coords(infer_cfg['input_shape'], pred_all_np[:, :4], (raw_img_h, raw_img_w),
                             ratio_pad=(scale_ratio, pad_size))
                for det in pred_all_np:
                    class_id = int(det[5])
                    confidence = float(det[4])
                    if class_id not in [1, 3] and confidence >= CONFIDENCE_THRESHOLD_FOR_DISPLAY:
                        current_detections_for_tracking.append(det)

    # 4. Track Prediction and Matching
    for track in active_tracks:
        track.predict_kf()

    matched_indices = []
    unmatched_detections_indices = list(range(len(current_detections_for_tracking)))
    unmatched_tracks_indices = list(range(len(active_tracks)))
    if len(current_detections_for_tracking) > 0 and len(active_tracks) > 0:
        iou_matrix = np.zeros((len(current_detections_for_tracking), len(active_tracks)), dtype=np.float32)
        for d, det_data in enumerate(current_detections_for_tracking):
            det_box = det_data[:4]
            for t_idx, track_obj in enumerate(active_tracks):
                track_box_pred = track_obj.get_current_bbox_for_display()
                iou_matrix[d, t_idx] = calculate_iou(det_box, track_box_pred)

        cost_matrix = 1 - iou_matrix
        det_indices_matched, track_indices_matched = linear_sum_assignment(cost_matrix)
        for d_idx, t_idx in zip(det_indices_matched, track_indices_matched):
            if iou_matrix[d_idx, t_idx] >= cfg.IOU_MATCHING_THRESHOLD:
                matched_indices.append((d_idx, t_idx))
                if d_idx in unmatched_detections_indices: unmatched_detections_indices.remove(d_idx)
                if t_idx in unmatched_tracks_indices: unmatched_tracks_indices.remove(t_idx)

    # 5. Update Matched Tracks
    for d_idx, t_idx in matched_indices:
        detection_data = current_detections_for_tracking[d_idx]
        active_tracks[t_idx].update_kf(detection_data[:4], int(detection_data[5]), float(detection_data[4]))

    # 6. Create New Tracks for Unmatched Detections
    for d_idx in unmatched_detections_indices:
        detection_data = current_detections_for_tracking[d_idx]
        original_model_class_id = int(detection_data[5])
        if original_model_class_id in cfg.TARGET_TRACKING_MODEL_CLASSES:
            new_track = Track(detection_data[:4], original_model_class_id, float(detection_data[4]), g_next_track_id)
            g_next_track_id += 1
            active_tracks.append(new_track)

    # 7. Remove Lost Tracks
    surviving_tracks = []
    for track_obj in active_tracks:
        if not track_obj.is_lost:
            surviving_tracks.append(track_obj)
        else:
            if track_obj.track_id == g_locked_target_track_id:
                g_locked_target_track_id = None
    active_tracks[:] = surviving_tracks

    # --- Target Selection Logic ---
    g_locked_target_track_id = select_best_target(active_tracks, g_locked_target_track_id, raw_img_w, raw_img_h)

    # 8. Draw Tracks and Information on Display Frame
    num_drawn_this_frame = 0

    for track in active_tracks:
        if track.is_tentative or track.class_id not in cfg.AUTO_AIM_CLASSES:
            continue

        current_bbox_xyxy = track.get_current_bbox_for_display()
        x1_f, y1_f, x2_f, y2_f = current_bbox_xyxy
        x1, y1, x2, y2 = map(int, current_bbox_xyxy)
        bbox_w_for_display = int(x2_f - x1_f)
        bbox_h_for_display = int(y2_f - y1_f)

        # ROI for color check on the raw (unexposed) frame
        roi_x1, roi_y1 = max(0, x1), max(0, y1)
        roi_x2, roi_y2 = min(raw_img_w, x2), min(raw_img_h, y2)
        final_class_id_for_track = track.model_class_id
        color_suffix_for_track = ""
        if roi_x2 > roi_x1 and roi_y2 > roi_y1:
            roi_on_raw_frame = frame_raw_bgr[roi_y1:roi_y2, roi_x1:roi_x2]
            contains_red = check_dominant_color_in_roi(roi_on_raw_frame, "red", cfg.COLOR_RANGES_HSV)
            contains_blue = check_dominant_color_in_roi(roi_on_raw_frame, "blue", cfg.COLOR_RANGES_HSV)
            if contains_red and contains_blue:
                color_suffix_for_track = "(R&B)"
            elif track.model_class_id in cfg.AUTO_AIM_CLASSES:
                if contains_red:
                    final_class_id_for_track = 0;
                    color_suffix_for_track = "(R)"
                elif contains_blue:
                    final_class_id_for_track = 2;
                    color_suffix_for_track = "(B)"
                else:
                    color_suffix_for_track = "(No R/B)"
        else:
            color_suffix_for_track = "(Inv.ROI)"

        track.class_id = final_class_id_for_track
        track.color_detection_suffix = color_suffix_for_track
        track.display_label_name = labels_dict.get(final_class_id_for_track, f"ID:{final_class_id_for_track}")

        track.calculate_aim_point()
        track.rotation_angle = calculate_object_rotation(track.width,
                                                         track.height) if track.width > 0 and track.height > 0 else None
        if track.rotation_angle is not None and abs(track.rotation_angle) > ANGLE_THRESHOLD_DEGREES_FOR_DISPLAY:
            continue

        track.estimated_distance = estimate_distance(track.height, cfg.PIXEL_HEIGHT_AT_CALIBRATION_DISTANCE,
                                                     cfg.CALIBRATION_DISTANCE_METERS) if track.height > 0 else None
        if track.estimated_distance is not None and track.estimated_distance < DISTANCE_THRESHOLD_FOR_DISPLAY:
            continue

        # --- Drawing based on target selection ---
        bbox_color = track.track_color
        bbox_thickness = 2
        aim_marker_color = (0, 0, 255)

        if track.track_id == g_locked_target_track_id:
            bbox_color = (0, 0, 255)
            bbox_thickness = 4
            aim_marker_color = (0, 255, 255)

            # --- 主要修改区域：组装新的 locked_target_info 字典 ---
            if track.predicted_aim_point:
                # 1. 获取并归一化预测的瞄准点 (黄色十字)
                aim_x_pixel, aim_y_pixel = track.predicted_aim_point
                aim_x_clamped = np.clip(aim_x_pixel, 0, raw_img_w - 1)
                aim_y_clamped = np.clip(aim_y_pixel, 0, raw_img_h - 1)
                normalized_aim_point = (aim_x_clamped / raw_img_w, aim_y_clamped / raw_img_h)

                # 2. 获取并归一化检测框的中心点
                center_x_pixel, center_y_pixel = track.center_x, track.center_y
                center_x_clamped = np.clip(center_x_pixel, 0, raw_img_w - 1)
                center_y_clamped = np.clip(center_y_pixel, 0, raw_img_h - 1)
                normalized_center_point = (center_x_clamped / raw_img_w, center_y_clamped / raw_img_h)

                # 3. 组装成字典用于返回
                locked_target_info_for_return = {
                    'id': track.track_id,
                    'aim_point_norm': normalized_aim_point,
                    'center_point_norm': normalized_center_point
                }
            # --- 修改结束 ---

        # 绘制
        disp_x1, disp_y1, disp_x2, disp_y2 = x1, y1, x2, y2
        if not (disp_x1 < disp_x2 and disp_y1 < disp_y2): continue

        num_drawn_this_frame += 1
        cv2.rectangle(display_frame, (disp_x1, disp_y1), (disp_x2, disp_y2), bbox_color, bbox_thickness)

        label_text = (f"TID:{track.track_id} {track.display_label_name}{track.color_detection_suffix} "
                      f"C:{track.conf:.2f} S:{track.score:.2f}")
        if track.rotation_angle is not None: label_text += f" Ang:{track.rotation_angle:.1f}°"
        if track.estimated_distance is not None: label_text += f" Dist:{track.estimated_distance:.2f}m"
        text_y_pos = disp_y1 - 10 if disp_y1 > 20 else disp_y1 + 20
        cv2.putText(display_frame, label_text, (disp_x1, text_y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, bbox_color, 1,
                    cv2.LINE_AA)

        if track.predicted_aim_point:
            aim_x, aim_y = map(int, track.predicted_aim_point)
            aim_x_clamped = np.clip(aim_x, 0, display_frame.shape[1] - 1)
            aim_y_clamped = np.clip(aim_y, 0, display_frame.shape[0] - 1)
            bbox_center_x, bbox_center_y = int((disp_x1 + disp_x2) / 2), int((disp_y1 + disp_y2) / 2)
            cv2.circle(display_frame, (bbox_center_x, bbox_center_y), 4, (0, 255, 255), -1)
            cv2.drawMarker(display_frame, (aim_x_clamped, aim_y_clamped), aim_marker_color,
                           markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)

    # 10. Display FPS and track count
    fps = 1.0 / infer_time if infer_time > 0 else 0
    cv2.putText(display_frame, f"FPS: {fps:.1f} Tracks: {len(active_tracks)} Drawn: {num_drawn_this_frame}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    if g_locked_target_track_id is not None:
        cv2.putText(display_frame, f"LOCKED: TID:{g_locked_target_track_id}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    # 返回处理后的帧、更新后的轨迹列表和锁定目标的信息
    return display_frame, active_tracks, locked_target_info_for_return