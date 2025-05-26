# main.py
import cv2
import numpy as np  # Though not directly used, good to have if extending
from ais_bench.infer.interface import InferSession

# Project-specific imports
import config as cfg
from utils import get_labels_from_txt, adjust_exposure_hsv
from detector import process_frame_with_tracking


def main():
    print("程序开始...")

    # --- Configuration ---
    model_path = cfg.DEFAULT_MODEL_PATH
    label_path = cfg.DEFAULT_LABEL_PATH
    infer_config = cfg.DEFAULT_INFER_CONFIG
    exposure_factor = cfg.DEFAULT_EXPOSURE_FACTOR

    # --- Load Model ---
    try:
        print(f"尝试加载FP16模型: {model_path}")
        model = InferSession(0, model_path)  # Assuming device_id 0
        print("InferSession 初始化完成。")
    except Exception as e:
        print(f"初始化 InferSession 时发生严重错误: {e}")
        return  # Exit if model fails to load

    # --- Load Labels ---
    print("加载标签文件...")
    labels_dict = get_labels_from_txt(label_path)
    if labels_dict is None:
        print("加载标签文件失败，程序退出。")
        return
    print(f"成功加载 {len(labels_dict)} 个标签: {labels_dict}")
    print(f"使用推理配置: {infer_config}")
    print(f"初始曝光调整因子: {exposure_factor:.2f}")

    # --- Initialize Camera ---
    print("初始化摄像头...")
    cap = cv2.VideoCapture(0)  # Use 0 for default camera
    if not cap.isOpened():
        print("错误: 无法打开摄像头。")
        return

    # --- Setup Window and State Variables ---
    window_name = 'Object Tracking & Aiming (Kalman Filter)'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    is_paused = False
    active_tracks_list = []  # Stores active Track objects

    # Buffers for paused state
    paused_frame_raw = None
    paused_frame_exposed = None
    processed_display_frame = None  # Stores the last processed frame for display when paused

    try:
        while True:
            frame_raw_current_loop = None
            frame_exposed_current_loop = None

            if not is_paused:
                ret, frame_raw_cam = cap.read()
                if not ret:
                    print("错误: 无法从摄像头捕获帧。视频结束或摄像头断开?")
                    if paused_frame_raw is None:  # If never captured a frame, break
                        break
                        # If already paused and stream ends, stay paused on last good frame
                    frame_raw_current_loop = paused_frame_raw
                    frame_exposed_current_loop = paused_frame_exposed
                    is_paused = True  # Force pause if stream ends
                else:
                    frame_raw_current_loop = frame_raw_cam.copy()
                    paused_frame_raw = frame_raw_current_loop.copy()  # Cache for pause/exposure change

                # Apply exposure adjustment
                frame_exposed_current_loop = adjust_exposure_hsv(frame_raw_current_loop, exposure_factor)
                paused_frame_exposed = frame_exposed_current_loop.copy()  # Cache exposed version

                # Process the frame
                processed_display_frame, active_tracks_list = process_frame_with_tracking(
                    model,
                    frame_raw_current_loop,  # For color ROI checks
                    frame_exposed_current_loop,  # For detection
                    active_tracks_list,
                    labels_dict,
                    infer_config
                )
            else:  # When paused
                if paused_frame_raw is None:  # Should not happen if logic is correct
                    print("Paused but no frame available. Resuming.")
                    is_paused = False
                    continue
                # Use cached frames when paused
                frame_raw_current_loop = paused_frame_raw
                frame_exposed_current_loop = paused_frame_exposed
                # No need to re-process unless exposure changes (handled by key press)

            # --- Display ---
            current_display = None
            if processed_display_frame is not None:
                current_display = processed_display_frame.copy()  # Work on a copy for adding text
                # Add exposure info
                cv2.putText(current_display, f"Exp: {exposure_factor:.1f}",
                            (current_display.shape[1] - 120, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
                if is_paused:
                    cv2.putText(current_display, "PAUSED", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow(window_name, current_display)
            elif frame_exposed_current_loop is not None:  # Fallback if processed_display_frame is somehow None
                cv2.imshow(window_name, frame_exposed_current_loop)
            elif frame_raw_current_loop is not None:  # Fallback further
                cv2.imshow(window_name, frame_raw_current_loop)

            # --- Key Handling ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("'q' pressed, exiting.")
                break
            elif key == ord('p'):
                is_paused = not is_paused
                if not is_paused:
                    print("Resumed.")
                else:
                    print("Paused.")

            # Exposure adjustment keys
            elif key == ord('+') or key == ord('='):
                exposure_factor = round(min(exposure_factor + 0.1, 3.0), 1)
                print(f"曝光因子增加到: {exposure_factor:.1f}")
                if paused_frame_raw is not None:  # Re-process if a frame is available
                    frame_exposed_current_loop = adjust_exposure_hsv(paused_frame_raw, exposure_factor)
                    if not is_paused: paused_frame_exposed = frame_exposed_current_loop.copy()  # Update cache if not paused

                    # Re-process with new exposure
                    processed_display_frame, active_tracks_list = process_frame_with_tracking(
                        model, paused_frame_raw, frame_exposed_current_loop,
                        active_tracks_list, labels_dict, infer_config
                    )

            elif key == ord('-') or key == ord('_'):
                exposure_factor = round(max(exposure_factor - 0.1, 0.1), 1)
                print(f"曝光因子减少到: {exposure_factor:.1f}")
                if paused_frame_raw is not None:
                    frame_exposed_current_loop = adjust_exposure_hsv(paused_frame_raw, exposure_factor)
                    if not is_paused: paused_frame_exposed = frame_exposed_current_loop.copy()

                    processed_display_frame, active_tracks_list = process_frame_with_tracking(
                        model, paused_frame_raw, frame_exposed_current_loop,
                        active_tracks_list, labels_dict, infer_config
                    )

    finally:
        print("释放资源...")
        if cap: cap.release()
        cv2.destroyAllWindows()
        print('程序结束.')


if __name__ == "__main__":
    main()