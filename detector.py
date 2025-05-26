# detector.py
import cv2
import numpy as np
import torch
from time import time
from scipy.optimize import linear_sum_assignment

# Assuming these are provided external utilities/APIs
from det_utils import nms, scale_coords
from GetZDgree_api import calculate_object_rotation
from Calcu_distance_api import estimate_distance

# Project-specific imports
import config as cfg  # Use alias for brevity
from utils import calculate_iou, preprocess_image, check_dominant_color_in_roi
from tracker import Track

g_next_track_id = 0  # Global for assigning unique track IDs


def process_frame_with_tracking(model, frame_raw_bgr, frame_for_detection_bgr, active_tracks, labels_dict, infer_cfg):
    """
    Processes a single frame: detects objects, updates tracks, and prepares display.
    """
    global g_next_track_id
    display_frame = frame_for_detection_bgr.copy()
    raw_img_h, raw_img_w = frame_for_detection_bgr.shape[:2]

    # 1. Preprocess image for detection
    img_processed, scale_ratio, pad_size = preprocess_image(frame_for_detection_bgr, infer_cfg)
    img_processed_normalized = img_processed / 255.0

    # 2. Model Inference
    t_start_infer = time()
    raw_model_outputs = model.infer([img_processed_normalized])
    t_end_infer = time()
    infer_time = t_end_infer - t_start_infer

    # 3. Post-process Detections
    current_detections = []  # List to store [x1,y1,x2,y2, conf, class_id]
    if raw_model_outputs and raw_model_outputs[0] is not None:
        output_np = raw_model_outputs[0]
        output_for_nms = torch.from_numpy(output_np)  # NMS function might expect a tensor

        # Apply NMS
        detections_from_nms_list = nms(output_for_nms, conf_thres=infer_cfg["conf_thres"],
                                       iou_thres=infer_cfg["iou_thres"])

        if detections_from_nms_list and detections_from_nms_list[0] is not None and detections_from_nms_list[
            0].numel() > 0:
            pred_all_tensor = detections_from_nms_list[0]
            pred_all_np = pred_all_tensor.cpu().numpy()  # [x1,y1,x2,y2, conf, class_id]

            if pred_all_np.shape[0] > 0:
                # Scale coordinates back to original image size
                scale_coords(infer_cfg['input_shape'], pred_all_np[:, :4], (raw_img_h, raw_img_w),
                             ratio_pad=(scale_ratio, pad_size))
                for det in pred_all_np:
                    current_detections.append(det)  # det is [x1,y1,x2,y2, conf, class_id]

    # 4. Track Prediction and Matching
    predicted_track_bboxes = []
    for track in active_tracks:
        predicted_track_bboxes.append(track.predict_kf())  # Kalman Filter predict step

    # Match detections with tracks using IoU and Hungarian algorithm
    matched_indices = []
    unmatched_detections_indices = list(range(len(current_detections)))
    unmatched_tracks_indices = list(range(len(active_tracks)))

    if len(current_detections) > 0 and len(active_tracks) > 0:
        iou_matrix = np.zeros((len(current_detections), len(active_tracks)), dtype=np.float32)
        for d, det_data in enumerate(current_detections):
            det_box = det_data[:4]  # x1,y1,x2,y2
            for t_idx, track_box_pred in enumerate(predicted_track_bboxes):
                iou_matrix[d, t_idx] = calculate_iou(det_box, track_box_pred)

        cost_matrix = 1 - iou_matrix  # Hungarian algorithm minimizes cost
        det_indices_matched, track_indices_matched = linear_sum_assignment(cost_matrix)

        for d_idx, t_idx in zip(det_indices_matched, track_indices_matched):
            if iou_matrix[d_idx, t_idx] >= cfg.IOU_MATCHING_THRESHOLD:
                matched_indices.append((d_idx, t_idx))
                if d_idx in unmatched_detections_indices: unmatched_detections_indices.remove(d_idx)
                if t_idx in unmatched_tracks_indices: unmatched_tracks_indices.remove(t_idx)

    # 5. Update Matched Tracks
    for d_idx, t_idx in matched_indices:
        detection_data = current_detections[d_idx]
        active_tracks[t_idx].update_kf(detection_data[:4], int(detection_data[5]), float(detection_data[4]))

    # 6. Create New Tracks for Unmatched Detections
    newly_created_tracks_this_frame = []
    for d_idx in unmatched_detections_indices:
        detection_data = current_detections[d_idx]
        original_model_class_id = int(detection_data[5])
        # Only create tracks for specified classes
        if original_model_class_id in cfg.TARGET_TRACKING_MODEL_CLASSES:
            new_track = Track(detection_data[:4], original_model_class_id, float(detection_data[4]), g_next_track_id)
            g_next_track_id += 1
            active_tracks.append(new_track)
            newly_created_tracks_this_frame.append(new_track)

    # 7. (Optional) Handle potential track splitting: if new tracks are created, old tracks that weren't updated might be due to splits.
    # This logic is a bit aggressive and might need tuning or removal depending on behavior.
    tracks_to_delete_due_to_split_like_behavior = []
    if len(newly_created_tracks_this_frame) > 0:
        for track_obj in active_tracks:
            # If a track was not newly created and did not get an update (time_since_update > 0 means it only predicted)
            if track_obj not in newly_created_tracks_this_frame and track_obj.time_since_update > 0:
                # This is a heuristic: if new tracks appeared and this old one didn't match,
                # it might be because the object split or changed appearance drastically.
                # Consider removing it to prefer the new, more confident detections.
                # This is more relevant if MAX_FRAMES_SINCE_UPDATE is high.
                # For now, let's comment out the aggressive deletion logic for stability,
                # lost tracks will be handled by track.is_lost.
                # tracks_to_delete_due_to_split_like_behavior.append(track_obj)
                pass

    if tracks_to_delete_due_to_split_like_behavior:
        active_tracks[:] = [t for t in active_tracks if t not in tracks_to_delete_due_to_split_like_behavior]

    # 8. Remove Lost Tracks and Prepare for Display
    surviving_tracks = []
    for track_obj in active_tracks:
        if not track_obj.is_lost:
            surviving_tracks.append(track_obj)
    active_tracks[:] = surviving_tracks  # Update active_tracks in place

    # 9. Draw Tracks and Information on Display Frame
    num_drawn_this_frame = 0
    for track in active_tracks:
        if track.is_tentative:  # Don't draw tentative tracks
            continue

        current_bbox_state_floats = track.get_current_bbox_from_state()
        x1_f, y1_f, x2_f, y2_f = current_bbox_state_floats
        x1, y1, x2, y2 = map(int, current_bbox_state_floats)

        bbox_w_for_display = int(x2_f - x1_f)
        bbox_h_for_display = int(y2_f - y1_f)

        # ROI for color check on the raw (unexposed) frame
        roi_x1 = max(0, x1);
        roi_y1 = max(0, y1)
        roi_x2 = min(raw_img_w, x2);
        roi_y2 = min(raw_img_h, y2)

        current_model_class_id = track.model_class_id
        final_class_id_for_track = current_model_class_id
        color_suffix_for_track = ""

        if roi_x2 > roi_x1 and roi_y2 > roi_y1:  # Valid ROI
            roi_on_raw_frame = frame_raw_bgr[roi_y1:roi_y2, roi_x1:roi_x2]
            contains_red = check_dominant_color_in_roi(roi_on_raw_frame, "red", cfg.COLOR_RANGES_HSV)
            contains_blue = check_dominant_color_in_roi(roi_on_raw_frame, "blue", cfg.COLOR_RANGES_HSV)

            if contains_red and contains_blue:
                color_suffix_for_track = "(R&B)"  # Or handle as a special class if needed
            elif current_model_class_id in cfg.AUTO_AIM_CLASSES:  # Only refine class for target objects
                if contains_red:
                    final_class_id_for_track = 0  # Assuming class 0 is Red Armor
                    color_suffix_for_track = "(R)"
                elif contains_blue:
                    final_class_id_for_track = 2  # Assuming class 2 is Blue Armor
                    color_suffix_for_track = "(B)"
                else:
                    color_suffix_for_track = "(No R/B)"
        else:
            color_suffix_for_track = "(Inv.ROI)"

        track.class_id = final_class_id_for_track  # Update track's class_id based on color
        track.color_detection_suffix = color_suffix_for_track
        track.display_label_name = labels_dict.get(final_class_id_for_track, f"ID:{final_class_id_for_track}")

        # Calculate aim point
        track.calculate_aim_point()

        # Calculate rotation angle
        track.rotation_angle = None
        if track.class_id in cfg.AUTO_AIM_CLASSES:  # Only for auto-aim targets
            bbox_w_f = x2_f - x1_f
            bbox_h_f = y2_f - y1_f
            if bbox_w_f > 0 and bbox_h_f > 0:
                try:
                    track.rotation_angle = calculate_object_rotation(bbox_w_f, bbox_h_f)
                except Exception as e:
                    # print(f"Rotation calculation error: {e}")
                    pass

        # Estimate distance
        track.estimated_distance = None
        if track.class_id in cfg.AUTO_AIM_CLASSES and bbox_h_for_display > 0:
            try:
                track.estimated_distance = estimate_distance(
                    bbox_h_for_display,
                    cfg.PIXEL_HEIGHT_AT_CALIBRATION_DISTANCE,
                    cfg.CALIBRATION_DISTANCE_METERS
                )
            except Exception as e:
                # print(f"Distance estimation error: {e}")
                pass

        # Draw bounding box
        disp_x1 = max(0, x1);
        disp_y1 = max(0, y1)
        disp_x2 = min(display_frame.shape[1], x2);
        disp_y2 = min(display_frame.shape[0], y2)
        if not (disp_x1 < disp_x2 and disp_y1 < disp_y2): continue  # Skip if bbox is outside frame

        num_drawn_this_frame += 1
        cv2.rectangle(display_frame, (disp_x1, disp_y1), (disp_x2, disp_y2), track.track_color, 2)

        # Prepare and draw label
        label_text = f"TID:{track.track_id} {track.display_label_name}{track.color_detection_suffix} C:{track.conf:.2f}"
        if track.rotation_angle is not None:
            label_text += f" Ang:{track.rotation_angle:.1f}°"
        if track.estimated_distance is not None:
            label_text += f" Dist:{track.estimated_distance:.2f}m"
        label_text += f" W:{bbox_w_for_display} H:{bbox_h_for_display}"

        text_y_pos = disp_y1 - 10 if disp_y1 - 10 > 10 else disp_y1 + 20
        cv2.putText(display_frame, label_text, (disp_x1, text_y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, track.track_color, 1, cv2.LINE_AA)

        # Draw predicted aim point if available
        if track.predicted_aim_point:
            aim_x, aim_y = track.predicted_aim_point
            # Clamp aim point to be within frame boundaries
            aim_x_clamped = np.clip(aim_x, 0, display_frame.shape[1] - 1)
            aim_y_clamped = np.clip(aim_y, 0, display_frame.shape[0] - 1)

            # Draw circle at bbox center and crosshair at aim point
            bbox_center_x = int((disp_x1 + disp_x2) / 2)
            bbox_center_y = int((disp_y1 + disp_y2) / 2)
            cv2.circle(display_frame, (bbox_center_x, bbox_center_y), 4, (0, 255, 255), -1)  # Yellow dot for center
            cv2.drawMarker(display_frame, (aim_x_clamped, aim_y_clamped), (0, 0, 255),
                           markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)  # Red cross for aim point

    # 10. Display FPS and track count
    fps = 1.0 / infer_time if infer_time > 0 else 0
    cv2.putText(display_frame, f"FPS: {fps:.1f} Tracks: {len(active_tracks)} Drawn: {num_drawn_this_frame}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    # Display status messages
    total_detections_this_frame = len(current_detections)
    if num_drawn_this_frame == 0 and total_detections_this_frame == 0:
        cv2.putText(display_frame, "No Detections", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2,
                    cv2.LINE_AA)
    elif num_drawn_this_frame == 0 and total_detections_this_frame > 0:
        tentative_target_tracks = sum(
            1 for t in active_tracks if t.is_tentative and t.model_class_id in cfg.TARGET_TRACKING_MODEL_CLASSES)
        if tentative_target_tracks > 0:
            cv2.putText(display_frame, "Tentative Tracks...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 128, 0), 2,
                        cv2.LINE_AA)
        elif len(
                active_tracks) == 0 and total_detections_this_frame > 0:  # Detections but no active tracks (e.g. non-target classes)
            cv2.putText(display_frame, "Non-target Dets", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2,
                        cv2.LINE_AA)

    return display_frame, active_tracks  # Return updated tracks list