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
# 定义一个空的Trackbar回调函数，因为Trackbar变化时，我们只需要读取其值，不需要立即执行操作
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
    ep_robot = None  # 初始化为None，以便在finally中安全关闭
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
        time.sleep(1)  # 给设备一点时间稳定

    except Exception as e:
        print(f"初始化Robomaster时发生错误: {e}")
        # 如果Robomaster初始化失败，程序无法继续，直接退出
        if ep_robot: ep_robot.close()
        return

    # --- Load Model ---
    try:
        print(f"尝试加载FP16模型: {model_path}")
        model = InferSession(0, model_path)  # Assuming device_id 0
        print("InferSession 初始化完成。")
    except Exception as e:
        print(f"初始化 InferSession 时发生严重错误: {e}")
        if ep_robot: ep_robot.close()
        return  # Exit if model fails to load

    # --- Load Labels ---
    print("加载标签文件...")
    labels_dict = get_labels_from_txt(label_path)
    if labels_dict is None:
        print("加载标签文件失败，程序退出。")
        if ep_robot: ep_robot.close()
        return
    print(f"成功加载 {len(labels_dict)} 个标签: {labels_dict}")
    print(f"使用推理配置: {infer_config}")
    print(f"初始曝光调整因子: {exposure_factor:.2f}")

    # --- Initialize Camera ---
    print("初始化摄像头...")
    cap = cv2.VideoCapture(0)  # Use 0 for default camera
    if not cap.isOpened():
        print("错误: 无法打开摄像头。")
        if ep_robot: ep_robot.close()
        return

    # --- Setup Window and State Variables ---
    window_name = 'Object Tracking & Aiming (Kalman Filter)'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    # --- Create PID Control Window and Trackbars ---
    pid_window_name = 'PID Control'
    cv2.namedWindow(pid_window_name)
    # Trackbar只能处理整数，所以我们需要乘以一个因子来表示小数。
    # 例如，P值范围1-500，I值范围0-100 (代表0-1.0)，D值范围0-200 (代表0-200.0)
    # alpha_d_filter范围0-100 (代表0-1.0)
    p_max = 500
    i_max = 100  # Will be i / 100.0
    d_max = 200  # Will be d
    alpha_d_filter_max = 100  # Will be alpha_d_filter / 100.0
    integral_limit_max = 5000  # Will be integral_limit / 10.0

    # Initial PID values for trackbars (from config)
    initial_p_yaw = cfg.GIMBAL_YAW_KP_INIT
    initial_i_yaw = int(cfg.GIMBAL_YAW_KI_INIT * 100) # For trackbar: 0.1 -> 10
    initial_d_yaw = cfg.GIMBAL_YAW_KD_INIT
    initial_alpha_d_filter_yaw = int(cfg.GIMBAL_YAW_ALPHA_D_FILTER_INIT * 100) # For trackbar: 0.3 -> 30
    initial_integral_limit_yaw = int(cfg.GIMBAL_YAW_INTEGRAL_LIMIT_INIT * 10) # For trackbar: 10.0 -> 100

    initial_p_pitch = cfg.GIMBAL_PITCH_KP_INIT
    initial_i_pitch = int(cfg.GIMBAL_PITCH_KI_INIT * 100) # For trackbar: 0.1 -> 10
    initial_d_pitch = cfg.GIMBAL_PITCH_KD_INIT
    initial_alpha_d_filter_pitch = int(cfg.GIMBAL_PITCH_ALPHA_D_FILTER_INIT * 100) # For trackbar: 0.3 -> 30
    initial_integral_limit_pitch = int(cfg.GIMBAL_PITCH_INTEGRAL_LIMIT_INIT * 10) # For trackbar: 10.0 -> 100


    # Yaw PID Trackbars
    cv2.createTrackbar('P_Yaw', pid_window_name, initial_p_yaw, p_max, on_trackbar_change)
    cv2.createTrackbar('I_Yaw (x0.01)', pid_window_name, initial_i_yaw, i_max, on_trackbar_change)
    cv2.createTrackbar('D_Yaw', pid_window_name, initial_d_yaw, d_max, on_trackbar_change)
    cv2.createTrackbar('Alpha_D_Yaw (x0.01)', pid_window_name, initial_alpha_d_filter_yaw, alpha_d_filter_max, on_trackbar_change)
    cv2.createTrackbar('Integral_Limit_Yaw (x0.1)', pid_window_name, initial_integral_limit_yaw, integral_limit_max, on_trackbar_change)

    # Pitch PID Trackbars
    cv2.createTrackbar('P_Pitch', pid_window_name, initial_p_pitch, p_max, on_trackbar_change)
    cv2.createTrackbar('I_Pitch (x0.01)', pid_window_name, initial_i_pitch, i_max, on_trackbar_change)
    cv2.createTrackbar('D_Pitch', pid_window_name, initial_d_pitch, d_max, on_trackbar_change)
    cv2.createTrackbar('Alpha_D_Pitch (x0.01)', pid_window_name, initial_alpha_d_filter_pitch, alpha_d_filter_max, on_trackbar_change)
    cv2.createTrackbar('Integral_Limit_Pitch (x0.1)', pid_window_name, initial_integral_limit_pitch, integral_limit_max, on_trackbar_change)


    is_paused = False
    active_tracks_list = []  # Stores active Track objects

    # Buffers for paused state
    paused_frame_raw = None
    paused_frame_exposed = None
    processed_display_frame = None  # Stores the last processed frame for display when paused

    # PID state variables (Separate for Yaw and Pitch)
    prev_err_x_yaw = 0.0
    prev_err_y_pitch = 0.0
    accumulate_err_x_yaw = 0.0
    accumulate_err_y_pitch = 0.0

    filtered_der_err_x_yaw = 0.0
    filtered_der_err_y_pitch = 0.0

    prev_time = time.time()  # Initialize prev_time right before the loop

    try:
        while True:
            current_time = time.time()  # Start of the current loop iteration

            frame_raw_current_loop = None
            frame_exposed_current_loop = None
            locked_aim_point_normalized = None  # To store the returned normalized aim point

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


            if not is_paused:
                ret, frame_raw_cam = cap.read()
                if not ret:
                    print("错误: 无法从摄像头捕获帧。视频结束或摄像头断开?")
                    if paused_frame_raw is None:
                        break
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
                    frame_exposed_current_loop = None
                    paused_frame_exposed = None

                if frame_raw_current_loop is not None and frame_exposed_current_loop is not None:
                    processed_display_frame, active_tracks_list, locked_aim_point_normalized = process_frame_with_tracking(
                        model,
                        frame_raw_current_loop,
                        frame_exposed_current_loop,
                        active_tracks_list,
                        labels_dict,
                        infer_config
                    )
                else:
                    processed_display_frame = None
                    locked_aim_point_normalized = None

            else:  # When paused
                if paused_frame_raw is None:
                    print("Paused but no frame available. Resuming.")
                    is_paused = False
                    continue
                frame_raw_current_loop = paused_frame_raw
                frame_exposed_current_loop = paused_frame_exposed

                processed_display_frame, active_tracks_list, locked_aim_point_normalized = process_frame_with_tracking(
                    model,
                    frame_raw_current_loop,
                    frame_exposed_current_loop,
                    active_tracks_list,
                    labels_dict,
                    infer_config
                )

            # --- PID Control Logic ---
            dt = current_time - prev_time
            if dt < 1e-6:
                dt = 1e-6

            if locked_aim_point_normalized:
                err_x = locked_aim_point_normalized[0] - 0.5
                err_y = 0.8 - locked_aim_point_normalized[1]

                # --- Yaw (X-axis) PID Calculation ---
                accumulate_err_x_yaw += err_x * dt
                accumulate_err_x_yaw = np.clip(accumulate_err_x_yaw, -integral_limit_yaw, integral_limit_yaw)

                current_der_err_x = (err_x - prev_err_x_yaw) / dt
                filtered_der_err_x_yaw = alpha_d_filter_yaw * current_der_err_x + (1 - alpha_d_filter_yaw) * filtered_der_err_x_yaw

                speed_x = (p_yaw * err_x) + (d_yaw * filtered_der_err_x_yaw) + (i_yaw * accumulate_err_x_yaw)

                # --- Pitch (Y-axis) PID Calculation ---
                accumulate_err_y_pitch += err_y * dt
                accumulate_err_y_pitch = np.clip(accumulate_err_y_pitch, -integral_limit_pitch, integral_limit_pitch)

                current_der_err_y = (err_y - prev_err_y_pitch) / dt
                filtered_der_err_y_pitch = alpha_d_filter_pitch * current_der_err_y + (1 - alpha_d_filter_pitch) * filtered_der_err_y_pitch

                speed_y = (p_pitch * err_y) + (d_pitch * filtered_der_err_y_pitch) + (i_pitch * accumulate_err_y_pitch)

                max_gimbal_speed = 300
                speed_x = np.clip(speed_x, -max_gimbal_speed, max_gimbal_speed)
                speed_y = np.clip(speed_y, -max_gimbal_speed, max_gimbal_speed)

                ep_gimbal.drive_speed(pitch_speed=-speed_y, yaw_speed=speed_x)

                print(
                    f"锁定目标预瞄点 (Normalized): ({locked_aim_point_normalized[0]:.4f}, {locked_aim_point_normalized[1]:.4f}) | "
                    f"Err_x: {err_x:.4f}, Err_y: {err_y:.4f} | "
                    f"Speed_x: {speed_x:.1f}, Speed_y: {speed_y:.1f}")
            else:
                ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
                print("未锁定目标或预瞄点不可用。")
                # Reset all PID state variables for both axes
                accumulate_err_x_yaw = 0.0
                accumulate_err_y_pitch = 0.0
                prev_err_x_yaw = 0.0
                prev_err_y_pitch = 0.0
                filtered_der_err_x_yaw = 0.0
                filtered_der_err_y_pitch = 0.0

            # --- 更新PID状态变量以供下一帧使用 ---
            prev_time = current_time  # 当前循环开始的时间成为下一次的"之前时间"
            # 更新 prev_err_x 和 prev_err_y
            if locked_aim_point_normalized:
                prev_err_x_yaw = err_x
                prev_err_y_pitch = err_y
            else:
                # These are already reset to 0 above, but setting again for clarity
                prev_err_x_yaw = 0.0
                prev_err_y_pitch = 0.0


            # --- Display ---
            current_display = None
            if processed_display_frame is not None:
                current_display = processed_display_frame.copy()
                cv2.putText(current_display, f"Exp: {exposure_factor:.1f}",
                            (current_display.shape[1] - 120, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
                if is_paused:
                    cv2.putText(current_display, "PAUSED", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow(window_name, current_display)
            elif frame_exposed_current_loop is not None:
                cv2.imshow(window_name, frame_exposed_current_loop)
            elif frame_raw_current_loop is not None:
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

            elif key == ord('+') or key == ord('='):
                exposure_factor = round(min(exposure_factor + 0.1, 3.0), 1)
                print(f"曝光因子增加到: {exposure_factor:.1f}")
                if paused_frame_raw is not None:
                    paused_frame_exposed = adjust_exposure_hsv(paused_frame_raw, exposure_factor)

            elif key == ord('-') or key == ord('_'):
                exposure_factor = round(max(exposure_factor - 0.1, 0.1), 1)
                print(f"曝光因子减少到: {exposure_factor:.1f}")
                if paused_frame_raw is not None:
                    paused_frame_exposed = adjust_exposure_hsv(paused_frame_raw, exposure_factor)

    finally:
        print("释放资源...")
        if cap: cap.release()
        cv2.destroyAllWindows()  # 关闭所有OpenCV窗口，包括PID控制窗口
        if ep_robot:
            ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)  # 停止云台
            ep_robot.close()  # 关闭Robomaster连接
            print('Robomaster连接已关闭。')
        print('程序结束.')


if __name__ == "__main__":
    main()