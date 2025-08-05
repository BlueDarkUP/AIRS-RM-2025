import serial
import time
import threading
import sys
import signal
import subprocess

# 全局退出标志
exit_flag = False

def serial_listener():
    """串口监听线程函数"""
    global exit_flag
    while not exit_flag:
        try:
            if ser.isOpen():
                line_raw = ser.readline()
                if line_raw:
                    line = line_raw.decode('utf-8', errors='ignore').strip()
                    if not line:
                        continue

                    print(f"收到串口指令: '{line}'")

                    # --- 当接收到指令 "1" 或 "2" ---
                    if line.startswith("1") or line.startswith("2"):
                        
                        # --- 根据指令选择要执行的脚本 ---
                        script_to_run = ""
                        if line.startswith("1"):
                            script_to_run = "main_shoot_blue.py"
                            print(f"执行[1]指令任务: 尝试运行 '{script_to_run}'...")
                        elif line.startswith("2"):
                            script_to_run = "main_shoot_red.py"
                            print(f"执行[2]指令任务: 尝试运行 '{script_to_run}'...")

                        # --- 定义通用路径和命令执行 ---
                        project_dir = "/home/HwHiAiUser/PyCharmMiscProject"
                        venv_python = f"{project_dir}/.venv1/bin/python"

                        try:
                            # 使用 subprocess.run 执行命令
                            result = subprocess.run(
                                [venv_python, script_to_run],
                                cwd=project_dir,
                                check=True,
                                capture_output=True,
                                text=True,
                                timeout=60 # 设置60秒超时
                            )
                            print(f"--> 脚本 '{script_to_run}' 执行成功。")
                            print(f"--> 脚本输出:\n---\n{result.stdout.strip()}\n---")

                        except FileNotFoundError:
                            print(f"--> 错误: 无法找到文件。请检查路径是否正确: \n    解释器: '{venv_python}'\n    工作目录: '{project_dir}'")
                        except subprocess.CalledProcessError as e:
                            print(f"--> 错误: 目标脚本 '{script_to_run}' 执行失败。返回码: {e.returncode}")
                            print(f"--> 错误输出:\n---\n{e.stderr.strip()}\n---")
                        except subprocess.TimeoutExpired:
                            print(f"--> 错误: 执行脚本 '{script_to_run}' 超时。")
                        except Exception as e:
                            print(f"--> 执行命令时发生未知错误: {e}")

                        # 任务执行完毕后，设置退出标志，终止监听
                        print("任务完成，准备退出程序。")
                        exit_flag = True
                        break # 退出 while 循环

                    elif line == "reset":
                        print("执行reset指令任务")
                    else:
                        print("无效指令")

        except serial.SerialException as e:
            print(f"串口异常: {e}")
            time.sleep(2) # 发生异常时等待一会
        except Exception as e:
            print(f"串口监听出现未知错误: {e}")
            break

    print("串口监听线程安全退出。")

def keyboard_exit(signum, frame):
    """安全退出函数"""
    global exit_flag
    if not exit_flag:
        print("\n收到终止信号 (Ctrl+C)，程序终止中...")
        exit_flag = True

if __name__ == '__main__':
    try:
        ser = serial.Serial(
            port='/dev/ttyAMA0',
            baudrate=115200,
            timeout=0.2
        )
    except serial.SerialException as e:
        print(f"致命错误: 无法打开串口 /dev/ttyAMA0: {e}")
        sys.exit(1)

    signal.signal(signal.SIGINT, keyboard_exit)
    signal.signal(signal.SIGTERM, keyboard_exit)

    print(f"串口已打开: {ser.isOpen()}")
    listener_thread = threading.Thread(target=serial_listener)
    listener_thread.start()
    print("串口监听已启动，等待指令 '1' 或 '2'，或按Ctrl+C退出...")

    listener_thread.join() # 主线程等待监听线程结束

    if ser.isOpen():
        ser.close()
        print("串口已关闭。")

    print("程序已安全退出。")