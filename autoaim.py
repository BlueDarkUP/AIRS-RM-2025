# --- START OF FILE main.py ---

import cv2
import numpy as np
import time
import threading
import queue
from robomaster import robot, blaster, armor
# from ais_bench.infer.interface import InferSession # 在需要的地方局部导入

# 项目自定义模块导入
import config as cfg
from utils import get_labels_from_txt, adjust_exposure_hsv
from detector import process_frame_with_tracking

# --- 全局变量、线程通信与核心对象 ---
frame_queue = queue.Queue(maxsize=1)
result_queue = queue.Queue(maxsize=1)
stop_event = threading.Event()
ai_model_initialized = threading.Event()

# API可访问的核心对象
ep_robot = None
cap = None
ai_thread = None
pid_thread = None

# 控制初始巡逻移动的标志
_initial_patrol_approach_pending = False
_initial_patrol_approach_target_yaw = 0.0

# --- 机器人状态机与数据共享 ---
state_lock = threading.Lock()
robot_state = "IDLE"  # IDLE, SEARCHING, TRACKING, COASTING
hit_direction_info = {"target_angle": 0}

gimbal_angle_lock = threading.Lock()
current_gimbal_yaw = 0.0

# 全局可配置变量
g_patrol_yaw_min = -180
g_patrol_yaw_max = 180

# --- 全局常量定义 ---
HIT_ANGLE_MAP = {1: -210, 2: -30, 3: -120, 4: 30, 5: 0, 6: 0}
COASTING_TIMEOUT = 1.0
PATROL_YAW_SPEED = 40
GIMBAL_SEARCH_ROTATE_SPEED = 540
SEARCH_TIMEOUT = 5.0

# --- SDK回调函数 ---
def armor_hit_callback(hit_info):
    global robot_state, hit_direction_info
    armor_id, hit_type = hit_info
    with state_lock:
        if robot_state == "IDLE" or robot_state == "COASTING":
            target_angle = HIT_ANGLE_MAP.get(armor_id)
            if target_angle is not None:
                print(f"\n[{time.strftime('%H:%M:%S')}] 装甲 {armor_id} 被 '{hit_type}' 击中！-> [SEARCHING]")
                robot_state = "SEARCHING"
                hit_direction_info['target_angle'] = target_angle

def gimbal_angle_callback(angle_info):
    global current_gimbal_yaw
    if isinstance(angle_info, (list, tuple)) and len(angle_info) > 1:
        yaw_angle = angle_info[1]
    else:
        print(f"警告: gimbal_angle_callback 接收到非预期格式的 angle_info: {angle_info}")
        yaw_angle = 0.0
    with gimbal_angle_lock:
        current_gimbal_yaw = yaw_angle

# --- 工作线程定义 ---

def ai_processing_thread(model_path, label_path, infer_config):
    """
    线程一：负责AI识别。
    关键：在此线程内部导入并初始化模型，以保证硬件上下文与推理操作在同一线程。
    """
    from ais_bench.infer.interface import InferSession
    
    print("[AI Thread] 线程启动，正在初始化AI模型...")
    model = None
    labels_dict = None
    
    try:
        model = InferSession(0, model_path)
        labels_dict = get_labels_from_txt(label_path)
        if labels_dict is None:
            raise ValueError("无法加载标签文件")
        print("[AI Thread] AI模型初始化成功。")
        
        # --- 优化点: 增加模型预热 ---
        print("[AI Thread] 正在进行模型预热...")
        # 注意: 请根据你的模型实际输入调整这里的形状和类型
        dummy_input = np.zeros((1, 3, 640, 640), dtype=np.float32)
        if hasattr(model, 'infer'):
             # 假设模型的推理方法是 .infer()
            model.infer([dummy_input])
        elif hasattr(model, 'run'):
             # 或者 .run() 等
            model.run(None, {'images': dummy_input})
        print("[AI Thread] 模型预热完成。")

        ai_model_initialized.set()
    except Exception as e:
        print(f"[AI Thread] 严重错误: AI模型初始化失败: {e}")
        stop_event.set()
        return

    active_tracks_list = []
    while not stop_event.is_set():
        try:
            frame_raw, exposure_factor = frame_queue.get(timeout=1)
        except queue.Empty:
            continue

        frame_exposed = adjust_exposure_hsv(frame_raw, exposure_factor)
        _, active_tracks_list, locked_target_info = process_frame_with_tracking(
            model, frame_raw, frame_exposed, active_tracks_list, labels_dict, infer_config)

        if result_queue.full():
            try: result_queue.get_nowait()
            except queue.Empty: pass
        result_queue.put(locked_target_info)
        
    print("[AI Thread] 线程结束。")
    if model:
        try:
            if hasattr(model, 'release'): model.release()
            elif hasattr(model, 'close'): model.close()
            print("[AI Thread] AI模型资源已释放。")
        except Exception as e:
            print(f"[AI Thread] 释放AI模型时出错: {e}")


def pid_control_thread():
    """线程二：负责机器人状态控制与运动指令。"""
    global robot_state, hit_direction_info, current_gimbal_yaw, g_patrol_yaw_min, g_patrol_yaw_max
    global _initial_patrol_approach_pending, _initial_patrol_approach_target_yaw

    print("[PID Thread] 线程启动...")
    if ep_robot is None:
        print("[PID Thread] 错误: Robomaster未初始化。")
        stop_event.set()
        return

    ep_gimbal = ep_robot.gimbal
    ep_blaster = ep_robot.blaster

    p_yaw, i_yaw, d_yaw, f_yaw = cfg.GIMBAL_YAW_KP_INIT, cfg.GIMBAL_YAW_KI_INIT, cfg.GIMBAL_YAW_KD_INIT, cfg.GIMBAL_YAW_KF_INIT
    alpha_d_filter_yaw, integral_limit_yaw = cfg.GIMBAL_YAW_ALPHA_D_FILTER_INIT, cfg.GIMBAL_YAW_INTEGRAL_LIMIT_INIT
    p_pitch, i_pitch, d_pitch, f_pitch = cfg.GIMBAL_PITCH_KP_INIT, cfg.GIMBAL_PITCH_KI_INIT, cfg.GIMBAL_PITCH_KD_INIT, cfg.GIMBAL_PITCH_KF_INIT
    alpha_d_filter_pitch, integral_limit_pitch = cfg.GIMBAL_PITCH_ALPHA_D_FILTER_INIT, cfg.GIMBAL_PITCH_INTEGRAL_LIMIT_INIT

    prev_err_x_yaw, prev_err_y_pitch, accumulate_err_x_yaw, accumulate_err_y_pitch, filtered_der_err_x_yaw, filtered_der_err_y_pitch = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    locked_target_id, locked_target_frame_count = None, 0
    STABILITY_THRESHOLD, FIRE_COOLDOWN_SECONDS = cfg.STABILITY_THRESHOLD, cfg.FIRE_COOLDOWN_SECONDS
    last_fire_time = 0.0
    SCAN_YAW_SPEED = 45
    SCAN_DURATION_PER_SIDE = 1.5
    scan_start_time = 0
    scan_direction = 1
    last_target_info = None
    CONTROL_LOOP_INTERVAL = 1.0 / 50
    prev_time = time.time()
    coasting_start_time = 0
    last_known_velocity = (0.0, 0.0)
    patrol_direction = 1
    search_task_initiated = False
    search_operation_start_time = 0

    while not stop_event.is_set():
        loop_start_time = time.time()
        new_target_info = None
        try:
            new_target_info = result_queue.get_nowait()
        except queue.Empty:
            pass

        with state_lock:
            current_state = robot_state

        if new_target_info:
            last_target_info = new_target_info
            if current_state != "TRACKING":
                print(f"\n[{time.strftime('%H:%M:%S')}] 发现/重获目标！-> [TRACKING]")
                _initial_patrol_approach_pending = False
                with state_lock:
                    robot_state = "TRACKING"
        else:
            if current_state == "TRACKING":
                print(f"\n[{time.strftime('%H:%M:%S')}] 目标短暂丢失！-> [COASTING]")
                with state_lock:
                    robot_state = "COASTING"
                coasting_start_time = time.time()
                last_known_velocity = last_target_info.get('velocity_norm', (0.0, 0.0)) if last_target_info else (0.0, 0.0)

        with state_lock:
            current_state = robot_state
        with gimbal_angle_lock:
            yaw_now = current_gimbal_yaw

        if current_state == "IDLE":
            if _initial_patrol_approach_pending:
                print(f"\r[IDLE] 正在高速转向至巡逻起点 {_initial_patrol_approach_target_yaw:.1f}°，同时设置俯仰角为-18°...", end="")
                ep_gimbal.moveto(yaw=_initial_patrol_approach_target_yaw, pitch=-18, yaw_speed=GIMBAL_SEARCH_ROTATE_SPEED, pitch_speed=180).wait_for_completed()
                print(f"\n[{time.strftime('%H:%M:%S')}] 已到达巡逻起点和指定俯仰角。开始慢速巡逻。")
                _initial_patrol_approach_pending = False
                patrol_direction = 1
                ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=patrol_direction * PATROL_YAW_SPEED)
            else:
                print(f"\r[IDLE] 正在巡逻... 范围: [{g_patrol_yaw_min:.1f}, {g_patrol_yaw_max:.1f}], 方向: {'向右' if patrol_direction == 1 else '向左'}, 当前角度: {yaw_now:.1f}°", end="")
                if yaw_now >= g_patrol_yaw_max:
                    patrol_direction = -1
                elif yaw_now <= g_patrol_yaw_min:
                    patrol_direction = 1
                ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=patrol_direction * PATROL_YAW_SPEED)
            search_task_initiated = False

        elif current_state == "SEARCHING":
            if not search_task_initiated:
                search_task_initiated = True
                target_angle = hit_direction_info['target_angle']
                search_operation_start_time = time.time()
                print(f"[{time.strftime('%H:%M:%S')}] 定向任务：转向 {target_angle}°...")
                ep_gimbal.moveto(yaw=target_angle, yaw_speed=GIMBAL_SEARCH_ROTATE_SPEED).wait_for_completed()
                print(f"[{time.strftime('%H:%M:%S')}] 定向完成，开始扫描...")
                scan_start_time = time.time()
            if time.time() - search_operation_start_time > SEARCH_TIMEOUT:
                print(f"\n[{time.strftime('%H:%M:%S')}] 搜索超时 -> [IDLE]")
                with state_lock:
                    robot_state = "IDLE"
                _initial_patrol_approach_pending = True
                _initial_patrol_approach_target_yaw = g_patrol_yaw_min
                continue
            if time.time() - scan_start_time > SCAN_DURATION_PER_SIDE:
                scan_direction *= -1
                scan_start_time = time.time()
            ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=scan_direction * SCAN_YAW_SPEED)

        elif current_state == "COASTING":
            if time.time() - coasting_start_time > COASTING_TIMEOUT:
                print(f"\n[{time.strftime('%H:%M:%S')}] 惯性追踪超时 -> [IDLE]")
                with state_lock:
                    robot_state = "IDLE"
                _initial_patrol_approach_pending = True
                _initial_patrol_approach_target_yaw = g_patrol_yaw_min
                ep_gimbal.drive_speed(0, 0)
                continue
            else:
                speed_x = f_yaw * last_known_velocity[0]
                speed_y = f_pitch * last_known_velocity[1]
                ep_gimbal.drive_speed(speed_y, speed_x)
        
        elif current_state == "TRACKING":
            search_task_initiated = False
            if last_target_info:
                dt = max(time.time() - prev_time, 1e-6)
                current_target_id = last_target_info['id']
                if current_target_id == locked_target_id:
                    locked_target_frame_count += 1
                else:
                    locked_target_id, locked_target_frame_count = current_target_id, 1
                if locked_target_frame_count >= STABILITY_THRESHOLD:
                    pid_target_point = last_target_info['aim_point_norm']
                    if time.time() - last_fire_time >= FIRE_COOLDOWN_SECONDS:
                        ep_blaster.fire(times=1)
                        last_fire_time = time.time()
                else:
                    pid_target_point = last_target_info['center_point_norm']
                err_x, err_y = pid_target_point[0] - 0.5, 0.565 - pid_target_point[1]
                vx_norm, vy_norm = last_target_info.get('velocity_norm', (0.0, 0.0))
                accumulate_err_x_yaw = np.clip(accumulate_err_x_yaw + err_x * dt, -integral_limit_yaw, integral_limit_yaw)
                current_der_err_x = (err_x - prev_err_x_yaw) / dt
                filtered_der_err_x_yaw = alpha_d_filter_yaw * current_der_err_x + (1 - alpha_d_filter_yaw) * filtered_der_err_x_yaw
                speed_x = (p_yaw * err_x) + (i_yaw * accumulate_err_x_yaw) + (d_yaw * filtered_der_err_x_yaw) + (f_yaw * vx_norm)
                accumulate_err_y_pitch = np.clip(accumulate_err_y_pitch + err_y * dt, -integral_limit_pitch, integral_limit_pitch)
                current_der_err_y = (err_y - prev_err_y_pitch) / dt
                filtered_der_err_y_pitch = alpha_d_filter_pitch * current_der_err_y + (1 - alpha_d_filter_pitch) * filtered_der_err_y_pitch
                speed_y = (p_pitch * err_y) + (i_pitch * accumulate_err_y_pitch) + (d_pitch * filtered_der_err_y_pitch) + (f_pitch * vy_norm)
                max_gimbal_speed = 2000
                ep_gimbal.drive_speed(np.clip(speed_y, -max_gimbal_speed, max_gimbal_speed), np.clip(speed_x, -max_gimbal_speed, max_gimbal_speed))
                prev_err_x_yaw, prev_err_y_pitch = err_x, err_y
            else:
                with state_lock:
                    robot_state = "IDLE"
                _initial_patrol_approach_pending = True
                _initial_patrol_approach_target_yaw = g_patrol_yaw_min
        
        prev_time = time.time()
        sleep_time = CONTROL_LOOP_INTERVAL - (time.time() - loop_start_time)
        if sleep_time > 0:
            time.sleep(sleep_time)
    
    print()
    if ep_robot:
        ep_robot.gimbal.drive_speed(0, 0)
    print("[PID Thread] 线程结束。")

def _cleanup_resources():
    """内部函数：用于停止线程和释放所有占用的资源。"""
    global ep_robot, cap, ai_thread, pid_thread
    global _initial_patrol_approach_pending, _initial_patrol_approach_target_yaw
    
    print("\n正在停止所有线程并释放资源...")
    stop_event.set()
    
    if ep_robot:
        try:
            if ep_robot.armor:
                ep_robot.armor.unsub_hit_event()
                print("已取消打击订阅。")
            if ep_robot.gimbal:
                ep_robot.gimbal.unsub_angle()
                print("已取消角度订阅。")
        except Exception as e:
            print(f"取消订阅时出错: {e}")
            
    if ai_thread and ai_thread.is_alive():
        ai_thread.join(timeout=3)
    if pid_thread and pid_thread.is_alive():
        pid_thread.join(timeout=2)
        
    if cap and cap.isOpened():
        cap.release()
        cap = None
        print("摄像头已释放。")
        
    if ep_robot:
        try:
            ep_robot.close()
            ep_robot = None
            print('Robomaster连接已关闭。')
        except Exception as e:
            print(f"关闭Robomaster连接时出错: {e}")
            
    _initial_patrol_approach_pending = False
    _initial_patrol_approach_target_yaw = 0.0
    print('清理完成，程序结束。')

# --- API ---

def autoaim_init(patrol_yaw_min, patrol_yaw_max, target_id=None):
    """
    并行初始化自动瞄准系统。
    立即启动AI模型初始化线程，同时进行Robomaster和摄像头的初始化。
    
    :param patrol_yaw_min: 巡逻最小Yaw角度。
    :param patrol_yaw_max: 巡逻最大Yaw角度。
    :param target_id: 要攻击的目标ID (0 for Red, 2 for Blue)。如果为 None，则攻击所有目标。
    """
    global ep_robot, cap, g_patrol_yaw_min, g_patrol_yaw_max, robot_state
    global _initial_patrol_approach_pending, _initial_patrol_approach_target_yaw
    global ai_thread

    print("\n--- 自动瞄准系统并行初始化中 ---")
    
    # 根据传入的 target_id 更新配置中的白名单
    if target_id is None:
        cfg.TARGET_ID_WHITELIST = {0, 2}  # 攻击所有装甲板
        print("目标设置为：所有装甲板 (ID: 0 和 2)")
    elif isinstance(target_id, int):
        cfg.TARGET_ID_WHITELIST = {target_id}
        target_name = "红色装甲板" if target_id == 0 else "蓝色装甲板" if target_id == 2 else f"未知ID {target_id}"
        print(f"目标设置为：{target_name} (ID: {target_id})")
    else:
        print(f"警告：无效的 target_id 类型 ({type(target_id)})。将默认攻击所有目标。")
        cfg.TARGET_ID_WHITELIST = {0, 2}

    g_patrol_yaw_min, g_patrol_yaw_max = patrol_yaw_min, patrol_yaw_max
    print(f"自动巡逻范围设置为: [{g_patrol_yaw_min}, {g_patrol_yaw_max}]")
    
    stop_event.clear()
    ai_model_initialized.clear()
    robot_state = "IDLE"
    
    try:
        ai_thread = threading.Thread(target=ai_processing_thread, args=(cfg.DEFAULT_MODEL_PATH, cfg.DEFAULT_LABEL_PATH, cfg.DEFAULT_INFER_CONFIG), daemon=True)
        ai_thread.start()

        print("初始化Robomaster...")
        ep_robot = robot.Robot()
        ep_robot.initialize(conn_type="rndis")
        ep_robot.armor.sub_hit_event(callback=armor_hit_callback)
        print("已订阅装甲打击事件。")
        ep_robot.gimbal.sub_angle(freq=50, callback=gimbal_angle_callback)
        print("已订阅云台角度信息。")
        print("Robomaster初始化完成。")
        
        _initial_patrol_approach_pending = True
        _initial_patrol_approach_target_yaw = g_patrol_yaw_min
        
        print("初始化摄像头...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise ConnectionError("无法打开摄像头")
        print("摄像头初始化完成。")
        
        print("--- 自动瞄准系统核心组件初始化完成 ---")
        return True
    except Exception as e:
        print(f"自动瞄准系统初始化失败: {e}")
        _cleanup_resources()
        return False

def autoaim_start():
    """启动自动瞄准系统。"""
    global ep_robot, cap, ai_thread, pid_thread
    
    print("\n--- 自动瞄准系统启动中 ---")
    if ep_robot is None or cap is None or ai_thread is None:
        print("错误: 系统未初始化或初始化失败，请先成功调用 autoaim_init。")
        return
        
    try:
        print("等待AI模型加载完成...")
        ai_model_initialized.wait()
        
        if stop_event.is_set():
            print("AI模型初始化失败，无法启动主循环。")
            return
            
        print("AI模型已就绪。现在启动PID控制和主循环...")
        
        pid_thread = threading.Thread(target=pid_control_thread, daemon=True)
        pid_thread.start()
        print("PID控制线程启动成功。")
        
        print("开始捕获摄像头帧并发送进行处理...")
        while not stop_event.is_set():
            ret, frame_raw = cap.read()
            if not ret:
                print("错误: 无法捕获帧。")
                break
            if frame_queue.full():
                try: frame_queue.get_nowait()
                except queue.Empty: pass
            frame_queue.put((frame_raw, cfg.DEFAULT_EXPOSURE_FACTOR))
            time.sleep(0.001)
            
    except Exception as e:
        print(f"在autoaim_start主函数中发生严重错误: {e}")
    finally:
        _cleanup_resources()
        print("--- 自动瞄准系统已停止 ---")

def stop_autoaim():
    """停止自动瞄准系统。"""
    print("\n接收到停止信号...")
    stop_event.set()

# --- 主函数入口 ---
if __name__ == '__main__':
    print("这是 autoaim 模块的演示。")
    # 示例：初始化为只攻击蓝色装甲板(ID=2)，巡逻范围 0-90 度
    print("将以 0 到 90 度的巡逻范围启动，并只攻击蓝色装甲板(ID:2)。")
    print("在终端中按下 Ctrl+C 来停止程序。")
    
    if autoaim_init(0, 90, target_id=2):
        autoaim_start_thread = threading.Thread(target=autoaim_start)
        try:
            autoaim_start_thread.start()
            autoaim_start_thread.join()
        except KeyboardInterrupt:
            stop_autoaim()
            autoaim_start_thread.join()
    else:
        print("初始化失败，程序退出。")