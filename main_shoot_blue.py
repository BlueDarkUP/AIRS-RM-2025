import threading
import time
# 假设 autoaim_init, autoaim_start, stop_autoaim 已从您的模块中导入
from autoaim import autoaim_init, autoaim_start, stop_autoaim

def run_mission():
    print("--- 运行我的机器人任务 ---")

    # --- 示例1：只攻击蓝色装甲板 (ID: 2) ---
    print("\n任务阶段1：攻击蓝色装甲板")
    # 调用初始化函数，设置巡逻范围和目标ID
    init_successful = autoaim_init(patrol_yaw_min=45, patrol_yaw_max=170, target_id=2)

    if not init_successful:
        print("初始化失败，任务无法开始。")
        return

    # 如果初始化成功，则启动主任务线程
    mission_thread = threading.Thread(target=autoaim_start)
    mission_thread.start()

    try:
        # 让自动瞄准系统运行20秒
        print("自动瞄准系统将运行 20 秒...")
        time.sleep(20)
    except KeyboardInterrupt:
        print("任务被手动中断。")
    finally:
        # 无论任务是正常完成还是被中断，都发送停止信号
        stop_autoaim()
        # 等待 autoaim_start 线程完成所有清理工作并完全退出
        mission_thread.join()

    print("\n--- 任务阶段1结束 ---")

    # --- 示例2：切换到攻击红色装甲板 (ID: 0) ---
    print("\n任务阶段2：切换为攻击红色装甲板")
    # 再次初始化，这次设置目标为红色装甲板
    init_successful = autoaim_init(patrol_yaw_min=-90, patrol_yaw_max=90, target_id=0)

    if not init_successful:
        print("初始化失败，任务无法继续。")
        return

    mission_thread = threading.Thread(target=autoaim_start)
    mission_thread.start()

    try:
        print("自动瞄准系统将再次运行 20 秒...")
        time.sleep(20)
    except KeyboardInterrupt:
        print("任务被手动中断。")
    finally:
        stop_autoaim()
        mission_thread.join()
        
    print("--- 我的机器人任务结束 ---")


run_mission()