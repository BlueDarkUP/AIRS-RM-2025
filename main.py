import cv2
import numpy as np
import time
from ais_bench.infer.interface import InferSession
from robomaster import robot  # 引入Robomaster SDK

# Project-specific imports
import config as cfg
from utils import get_labels_from_txt, adjust_exposure_hsv
from detector import process_frame_with_tracking


# --- Callback function for Trackbars ---
def on_trackbar_change(val):
    pass


def main():
    print("程序开始...")

    # --- Configuration ---
    model_path = cfg.DEFAULT_MODEL_PATH
    label_path = cfg.DEFAULT_LABEL_PATH
    infer_config = cfg.DEFAULT_INFER_CONFIG
    exposure_factor = cfg.DEFAULT_EXPOSURE_FACTOR

    # --- Robomaster Initialization ---
    ep_robot = None
    try:
        print("初始化Robomaster...")
        ep_robot = robot.Robot()
        ep_robot.initialize(conn_type="rndis")
        print(f"尝试连接Robomaster")
        ep_vision = ep_robot.vision
        ep_camera = ep_robot.camera
        ep_gimbal = ep_robot.gimbal
        ep_blaster = ep_robot.blaster
        print("Robomaster云台复位...")
        ep_gimbal.recenter(pitch_speed=200, yaw_speed=200).wait_for_completed()
        print("Robomaster初始化完成。")
        time.sleep(1)
    except Exception as e:
        print(f"初始化Robomaster时发生错误: {e}")
        if ep_robot: ep_robot.close()
        return

    # --- Load Model & Labels ---
    try:
        print(f"尝试加载FP16模型: {model_path}")
        model = InferSession(0, model_path)
        print("InferSession 初始化完成。")
    except Exception as e:
        print(f"初始化 InferSession 时发生严重错误: {e}")
        if ep_robot: ep_robot.close()
        return

    print("加载标签文件...")
    labels_dict = get_labels_from_txt(label_path)
    if labels_dict is None:
        print("加载标签文件失败，程序退出。")
        if ep_robot: ep_robot.close()
        return
    print(f"成功加载 {len(labels_dict)} 个标签: {labels_dict}")

    # --- Initialize Camera ---
    print("初始化摄像头...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误: 无法打开摄像头。")
        if ep_robot: ep_robot.close()
        return

    # --- Setup Windows & State Variables ---
    window_name = 'Object Tracking & Aiming (Kalman Filter)'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    # --- Create PID Control Window and Trackbars ---
    pid_window_name = 'PID Control'
    cv2.namedWindow(pid_window_name)
    p_max, i_max, d_max, alpha_d_filter_max, integral_limit_max = 500, 100, 200, 100, 5000
    initial_p_yaw, initial_i_yaw, initial_d_yaw, initial_alpha_d_filter_yaw, initial_integral_limit_yaw = cfg.GIMBAL_YAW_KP_INIT, int(
        cfg.GIMBAL_YAW_KI_INIT * 100), cfg.GIMBAL_YAW_KD_INIT, int(cfg.GIMBAL_YAW_ALPHA_D_FILTER_INIT * 100), int(
        cfg.GIMBAL_YAW_INTEGRAL_LIMIT_INIT * 10)
    initial_p_pitch, initial_i_pitch, initial_d_pitch, initial_alpha_d_filter_pitch, initial_integral_limit_pitch = cfg.GIMBAL_PITCH_KP_INIT, int(
        cfg.GIMBAL_PITCH_KI_INIT * 100), cfg.GIMBAL_PITCH_KD_INIT, int(cfg.GIMBAL_PITCH_ALPHA_D_FILTER_INIT * 100), int(
        cfg.GIMBAL_PITCH_INTEGRAL_LIMIT_INIT * 10)
    cv2.createTrackbar('P_Yaw', pid_window_name, initial_p_yaw, p_max, on_trackbar_change)
    cv2.createTrackbar('I_Yaw (x0.01)', pid_window_name, initial_i_yaw, i_max, on_trackbar_change)
    cv2.createTrackbar('D_Yaw', pid_window_name, initial_d_yaw, d_max, on_trackbar_change)
    cv2.createTrackbar('Alpha_D_Yaw (x0.01)', pid_window_name, initial_alpha_d_filter_yaw, alpha_d_filter_max,
                       on_trackbar_change)
    cv2.createTrackbar('Integral_Limit_Yaw (x0.1)', pid_window_name, initial_integral_limit_yaw, integral_limit_max,
                       on_trackbar_change)
    cv2.createTrackbar('P_Pitch', pid_window_name, initial_p_pitch, p_max, on_trackbar_change)
    cv2.createTrackbar('I_Pitch (x0.01)', pid_window_name, initial_i_pitch, i_max, on_trackbar_change)
    cv2.createTrackbar('D_Pitch', pid_window_name, initial_d_pitch, d_max, on_trackbar_change)
    cv2.createTrackbar('Alpha_D_Pitch (x0.01)', pid_window_name, initial_alpha_d_filter_pitch, alpha_d_filter_max,
                       on_trackbar_change)
    cv2.createTrackbar('Integral_Limit_Pitch (x0.1)', pid_window_name, initial_integral_limit_pitch, integral_limit_max,
                       on_trackbar_change)

    # --- Loop State Variables ---
    is_paused = False
    active_tracks_list = []
    paused_frame_raw, paused_frame_exposed, processed_display_frame = None, None, None
    prev_time = time.time()

    # --- PID State Variables ---
    prev_err_x_yaw, prev_err_y_pitch = 0.0, 0.0
    accumulate_err_x_yaw, accumulate_err_y_pitch = 0.0, 0.0
    filtered_der_err_x_yaw, filtered_der_err_y_pitch = 0.0, 0.0

    # <<< 新增：追踪稳定性相关的状态变量 >>>
    locked_target_id = None  # 当前锁定目标的ID
    locked_target_frame_count = 0  # 当前目标被连续追踪的帧数
    STABILITY_THRESHOLD = 12  # 稳定追踪阈值（帧），超过此值才使用预测点

    try:
        while True:
            current_time = time.time()
            frame_raw_current_loop, frame_exposed_current_loop = None, None
            locked_target_info = None  # <<< 修改：使用新的信息字典

            # --- Read PID parameters from Trackbars ---
            p_yaw = cv2.getTrackbarPos('P_Yaw', pid_window_name)
            i_yaw = cv2.getTrackbarPos('I_Yaw (x0.01)', pid_window_name) / 100.0
            d_yaw = cv2.getTrackbarPos('D_Yaw', pid_window_name)
            alpha_d_filter_yaw = cv2.getTrackbarPos('Alpha_D_Yaw (x0.01)', pid_window_name) / 100.0
            integral_limit_yaw = cv2.getTrackbarPos('Integral_Limit_Yaw (x0.1)', pid_window_name) / 10.0
            p_pitch = cv2.getTrackbarPos('P_Pitch', pid_window_name)
            i_pitch = cv2.getTrackbarPos('I_Pitch (x0.01)', pid_window_name) / 100.0
            d_pitch = cv2.getTrackbarPos('D_Pitch', pid_window_name)
            alpha_d_filter_pitch = cv2.getTrackbarPos('Alpha_D_Pitch (x0.01)', pid_window_name) / 100.0
            integral_limit_pitch = cv2.getTrackbarPos('Integral_Limit_Pitch (x0.1)', pid_window_name) / 10.0

            # --- Frame Acquisition and Processing ---
            if not is_paused:
                ret, frame_raw_cam = cap.read()
                if not ret:
                    print("错误: 无法从摄像头捕获帧。")
                    if paused_frame_raw is None: break
                    frame_raw_current_loop = paused_frame_raw
                    frame_exposed_current_loop = paused_frame_exposed
                    is_paused = True
                else:
                    frame_raw_current_loop = frame_raw_cam.copy()
                    paused_frame_raw = frame_raw_current_loop.copy()

                if frame_raw_current_loop is not None:
                    frame_exposed_current_loop = adjust_exposure_hsv(frame_raw_current_loop, exposure_factor)
                    paused_frame_exposed = frame_exposed_current_loop.copy()
            else:
                if paused_frame_raw is None:
                    is_paused = False
                    continue
                frame_raw_current_loop = paused_frame_raw
                frame_exposed_current_loop = paused_frame_exposed

            if frame_exposed_current_loop is not None:
                # <<< 修改：接收新的返回格式 locked_target_info >>>
                processed_display_frame, active_tracks_list, locked_target_info = process_frame_with_tracking(
                    model, frame_raw_current_loop, frame_exposed_current_loop,
                    active_tracks_list, labels_dict, infer_config
                )

            # --- PID Control Logic with Stability Check ---
            dt = current_time - prev_time
            if dt < 1e-6: dt = 1e-6

            pid_target_point = None
            tracking_mode_text = "状态: 未锁定"

            if locked_target_info:
                current_target_id = locked_target_info['id']

                # <<< 新增：更新连续追踪帧数 >>>
                if current_target_id == locked_target_id:
                    locked_target_frame_count += 1
                else:
                    # 发现新目标或切换目标，重置计数器
                    locked_target_id = current_target_id
                    locked_target_frame_count = 1

                # <<< 新增：根据追踪稳定性选择目标点 >>>
                if locked_target_frame_count < STABILITY_THRESHOLD:
                    # 追踪不稳定，使用检测框中心
                    pid_target_point = locked_target_info['center_point_norm']
                    tracking_mode_text = f"状态: 稳定中 ({locked_target_frame_count}/{STABILITY_THRESHOLD})"
                else:
                    # 追踪稳定，使用卡尔曼预测点
                    pid_target_point = locked_target_info['aim_point_norm']
                    tracking_mode_text = f"状态: 已锁定 (预测)"
            else:
                # <<< 新增：目标丢失时重置状态 >>>
                locked_target_id = None
                locked_target_frame_count = 0

            # --- Actual PID Calculation and Gimbal Control ---
            if pid_target_point:
                err_x = pid_target_point[0] - 0.5
                err_y = 0.8 - pid_target_point[1]  # 目标y坐标在屏幕上方0.8处

                # Yaw (X-axis) PID
                accumulate_err_x_yaw += err_x * dt
                accumulate_err_x_yaw = np.clip(accumulate_err_x_yaw, -integral_limit_yaw, integral_limit_yaw)
                current_der_err_x = (err_x - prev_err_x_yaw) / dt
                filtered_der_err_x_yaw = alpha_d_filter_yaw * current_der_err_x + (
                            1 - alpha_d_filter_yaw) * filtered_der_err_x_yaw
                speed_x = (p_yaw * err_x) + (d_yaw * filtered_der_err_x_yaw) + (i_yaw * accumulate_err_x_yaw)

                # Pitch (Y-axis) PID
                accumulate_err_y_pitch += err_y * dt
                accumulate_err_y_pitch = np.clip(accumulate_err_y_pitch, -integral_limit_pitch, integral_limit_pitch)
                current_der_err_y = (err_y - prev_err_y_pitch) / dt
                filtered_der_err_y_pitch = alpha_d_filter_pitch * current_der_err_y + (
                            1 - alpha_d_filter_pitch) * filtered_der_err_y_pitch
                speed_y = (p_pitch * err_y) + (d_pitch * filtered_der_err_y_pitch) + (i_pitch * accumulate_err_y_pitch)

                max_gimbal_speed = 300
                speed_x = np.clip(speed_x, -max_gimbal_speed, max_gimbal_speed)
                speed_y = np.clip(speed_y, -max_gimbal_speed, max_gimbal_speed)

                ep_gimbal.drive_speed(pitch_speed=-speed_y, yaw_speed=speed_x)

                # 更新 prev_err 用于下一帧计算
                prev_err_x_yaw = err_x
                prev_err_y_pitch = err_y
            else:
                ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
                # 重置PID状态变量
                accumulate_err_x_yaw, accumulate_err_y_pitch = 0.0, 0.0
                prev_err_x_yaw, prev_err_y_pitch = 0.0, 0.0
                filtered_der_err_x_yaw, filtered_der_err_y_pitch = 0.0, 0.0

            prev_time = current_time

            # --- Display ---
            display_frame = processed_display_frame if processed_display_frame is not None else frame_exposed_current_loop
            if display_frame is not None:
                # <<< 新增：在屏幕上显示追踪状态 >>>
                cv2.putText(display_frame, tracking_mode_text, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 50), 2, cv2.LINE_AA)

                cv2.putText(display_frame, f"Exp: {exposure_factor:.1f}", (display_frame.shape[1] - 120, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
                if is_paused:
                    cv2.putText(display_frame, "PAUSED", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow(window_name, display_frame)

            # --- Key Handling ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                is_paused = not is_paused
            elif key == ord('+') or key == ord('='):
                exposure_factor = round(min(exposure_factor + 0.1, 3.0), 1)
            elif key == ord('-') or key == ord('_'):
                exposure_factor = round(max(exposure_factor - 0.1, 0.1), 1)
            if is_paused and (key == ord('+') or key == ord('=') or key == ord('-') or key == ord('_')):
                if paused_frame_raw is not None:
                    paused_frame_exposed = adjust_exposure_hsv(paused_frame_raw, exposure_factor)

    finally:
        print("释放资源...")
        if cap: cap.release()
        cv2.destroyAllWindows()
        if ep_robot:
            ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
            ep_robot.close()
            print('Robomaster连接已关闭。')
        print('程序结束.')


if __name__ == "__main__":
    main()