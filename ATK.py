import time
import math
import re  # For parsing the API output string
from _24point_api import find_simple_24_formula
# Placeholder for your actual API call
def get_24_solution_from_api(symbol_str: str, numbers: tuple) -> str or None:
    print(f"Actual API to be called with: RoboMaster symbol_str='{symbol_str}', RoboMaster numbers_tuple={numbers}")

    # 1. 将RoboMaster的符号字符串映射到API期望的单个字符
    symbol_mapping = {
        "add": "+",
        "sub": "-",
        "mul": "*",  # 假设你的API用 '*' 代表乘法
        "div": "/"   # 假设你的API用 '/' 代表除法
    }
    operation_symbol_for_api = symbol_mapping.get(symbol_str)

    if not operation_symbol_for_api:
        print(f"Error: Unknown symbol_str '{symbol_str}' cannot be mapped to API symbol.")
        return None
    input_digits_for_api = "".join(map(str, numbers))

    print(f"Calling find_simple_24_formula with: operation_symbol='{operation_symbol_for_api}', input_digits='{input_digits_for_api}'")

    try:
        # 调用你的API函数
        solution_formula = find_simple_24_formula(operation_symbol_for_api, input_digits_for_api)

        if solution_formula:
            print(f"API returned solution: '{solution_formula}'")
            return solution_formula
        else:
            print(f"API ({find_simple_24_formula.__name__}) did not find a solution.")
            return None
    except Exception as e:
        print(f"Error calling API ({find_simple_24_formula.__name__}): {e}")
        import traceback
        traceback.print_exc()
        return None


from robomaster import robot, gimbal, blaster, camera, vision, led
def toNumber(x):
    return (x / 180) * math.pi


def toDegree(x):
    return (x / math.pi) * 180


degreeX = {1: 96, 2: 58.09, 3: 39.94, 4: 31.03}
degreeY = {1: 54, 2: 28.59, 3: 19.28, 4: 14.52}


def zoom2Angles(vd, n):
    assert vd == 96 or vd == 54
    assert n in [1, 2, 3, 4]
    return degreeX.get(n) if vd == 96 else degreeY.get(n)


def numberCut(num: int):
    s_num = str(num)
    return [int(digit) for digit in s_num]
latest_marker_info = []


def on_detect_marker(marker_info):
    global latest_marker_info
    latest_marker_info = marker_info


class Shooter:
    NORMAL = 0
    CLOSE_PLACE = 1

    def __init__(self, ep_robot_instance, closediff: float = 0):
        self.ep_robot = ep_robot_instance
        self.ep_gimbal = ep_robot_instance.gimbal
        self.ep_blaster = ep_robot_instance.blaster
        self.ep_camera = ep_robot_instance.camera

        self.calc_zoom_level = 1
        # get_sight_bead_position returns (x, y) from 0.0 to 1.0
        sight_pos = self.ep_camera.get_sight_bead_position()
        self.px: float = sight_pos[0]
        self.py: float = sight_pos[1]
        print(f"Sight bead at: px={self.px}, py={self.py}")
        self.closeDiff = closediff
        self.baseYaw: float = 0.0
        self.basePitch: float = 0.0

    def init_start_point(self):
        # Get current gimbal attitude (absolute angles)
        current_yaw = self.ep_gimbal.get_attitude(gimbal.attitude_yaw)
        current_pitch = self.ep_gimbal.get_attitude(gimbal.attitude_pitch)
        self.baseYaw = current_yaw
        self.basePitch = current_pitch
        print(f"Base gimbal attitude: Yaw={self.baseYaw}, Pitch={self.basePitch}")

    def shoot_at_target(self, x_norm, y_norm, shoot_type=None, zoom_level=None):
        """
        x_norm, y_norm: Normalized screen coordinates of the target (0.0 to 1.0)
        """
        if shoot_type is None:
            shoot_type = self.NORMAL
        if zoom_level is None:
            zoom_level = self.calc_zoom_level

        # Calculate angular offset
        yaw_offset = zoom2Angles(96, zoom_level) * (x_norm - self.px)

        if shoot_type == self.NORMAL:
            pitch_offset = zoom2Angles(54, zoom_level) * (self.py - y_norm)  # Screen Y is often inverted
        elif shoot_type == self.CLOSE_PLACE:
            pitch_offset = zoom2Angles(54, zoom_level) * (self.py + self.closeDiff - y_norm)
        else:
            print("Error: Unknown shoot type")
            return

        target_yaw = self.baseYaw + yaw_offset
        target_pitch = self.basePitch + pitch_offset

        print(f"Shooting at norm: ({x_norm:.2f}, {y_norm:.2f})")
        print(f"Calculated offsets: Yaw={yaw_offset:.2f}, Pitch={pitch_offset:.2f}")
        print(f"Target gimbal attitude: Yaw={target_yaw:.2f}, Pitch={target_pitch:.2f}")

        self.ep_gimbal.moveto(yaw=target_yaw, pitch=target_pitch).wait_for_completed()
        time.sleep(0.2)  # Allow gimbal to settle
        self.ep_blaster.fire(times=1)
        print("Fired!")
        time.sleep(0.5)  # Cooldown/observation time

    def set_zoom(self, zoom_val: int):
        if self.ep_camera.set_zoom(zoom=zoom_val):
            self.calc_zoom_level = zoom_val
            print(f"Camera zoom set to: {zoom_val}")
        else:
            print(f"Failed to set camera zoom to: {zoom_val}")


class TargetProcessor:
    def __init__(self, raw_marker_data: list):
        self.code = 0
        self.tag_info = {}
        self.symbol_str = None
        self.numbers_tuple = tuple()

        if not raw_marker_data or len(raw_marker_data) == 0:
            self.code = 100
            return

        parsed_tags = []
        for marker_item in raw_marker_data:
            marker_id = marker_item[6]
            mx, my = marker_item[1], marker_item[2]
            parsed_tags.append({'id': marker_id, 'x': mx, 'y': my})

        if len(parsed_tags) != 5:  # Expect 1 symbol + 4 numbers
            print(f"Warning: Expected 5 markers, got {len(parsed_tags)}")
            if len(parsed_tags) < 2:
                self.code = 100
                return

        current_symbol_tag = None
        current_number_tags_details = []

        for tag_data in parsed_tags:
            tag_id = tag_data['id']
            tag_x, tag_y = tag_data['x'], tag_data['y']

            if tag_id == 50:
                self.symbol_str = "add"; current_symbol_tag = tag_data
            elif tag_id == 51:
                self.symbol_str = "sub"; current_symbol_tag = tag_data
            elif tag_id == 43:
                self.symbol_str = "mul"; current_symbol_tag = tag_data
            elif tag_id == 52:
                self.symbol_str = "div"; current_symbol_tag = tag_data
            elif 10 <= tag_id <= 19:  # It's a number
                digit_val = tag_id - 10
                current_number_tags_details.append({'val': digit_val, 'x': tag_x, 'y': tag_y})
            else:
                print(f"Warning: Unknown marker ID {tag_id} found.")

        if not self.symbol_str or not current_symbol_tag:
            self.code = 201
            print("Error: Symbol not found among detected markers.")
            return

        if len(current_number_tags_details) != 4:
            print(f"Warning: Expected 4 numbers, got {len(current_number_tags_details)}")
            self.code = 202
            return

        self.tag_info['symbol'] = {'name': self.symbol_str, 'x': current_symbol_tag['x'], 'y': current_symbol_tag['y']}
        self.tag_info['numbers_details'] = current_number_tags_details
        self.numbers_tuple = tuple(
            sorted([n['val'] for n in current_number_tags_details]))

        print(
            f"Processed Target: Symbol='{self.symbol_str}' at ({self.tag_info['symbol']['x']:.2f}, {self.tag_info['symbol']['y']:.2f})")
        print(f"Processed Numbers: {self.numbers_tuple}")
        for n_detail in self.tag_info['numbers_details']:
            print(f"  Digit {n_detail['val']} at ({n_detail['x']:.2f}, {n_detail['y']:.2f})")

    def get_coords_for_digit(self, digit_to_find: int):
        """Finds the coordinates of the first occurrence of a digit."""
        for num_detail in self.tag_info.get('numbers_details', []):
            if num_detail['val'] == digit_to_find:
                return num_detail['x'], num_detail['y']
        return None, None

    def get_coords_for_symbol(self):
        symbol_data = self.tag_info.get('symbol')
        if symbol_data:
            return symbol_data['x'], symbol_data['y']
        return None, None


if __name__ == '__main__':
    ep_robot = robot.Robot()
    try:
        ep_robot.initialize(conn_type="sta", sta_ip="192.168.1.100")
        print("Robot initialized.")

        ep_led = ep_robot.led

        shooter_controller = Shooter(ep_robot, closediff=0.17)

        shooter_controller.set_zoom(3)
        ep_robot.gimbal.set_rotate_speed(180)
        ep_robot.set_robot_mode(robot.FREE)

        ep_robot.vision.sub_detect_info(name="marker", callback=on_detect_marker)
        print("Subscribed to marker detection.")

        ep_robot.gimbal.recenter().wait_for_completed()
        time.sleep(0.5)
        ep_robot.gimbal.move(pitch=10, pitch_speed=60).wait_for_completed()
        time.sleep(1)

        shooter_controller.init_start_point()

        program_start_time = time.time()
        detection_attempt_start_time = time.time()
        try_count = 0
        target_object = None

        while True:
            current_markers = latest_marker_info
            if current_markers:
                print(f"Main loop: Saw {len(current_markers)} markers.")
                temp_target = TargetProcessor(current_markers)
                if temp_target.code == 0 and temp_target.symbol_str and len(temp_target.numbers_tuple) == 4:
                    target_object = temp_target
                    ep_led.play_sound(led.SOUND_RECOGNIZED)
                    print("Target acquired and processed successfully!")
                    break
                else:
                    print(f"TargetProcessor code: {temp_target.code}, not ready yet.")

            if time.time() - detection_attempt_start_time > 2.0:
                try_count += 1
                print(f"Timeout for detection attempt {try_count}. Retrying...")
                if try_count % 4 == 1:  # Toggle zoom
                    new_zoom = 1 if shooter_controller.calc_zoom_level == 3 else 3
                    shooter_controller.set_zoom(new_zoom)
                if try_count % 4 == 2:  # Nudge gimbal slightly
                    ep_robot.gimbal.move(yaw=5, yaw_speed=30).wait_for_completed()
                    shooter_controller.init_start_point()
                if try_count % 4 == 3:
                    ep_robot.gimbal.move(yaw=-10, yaw_speed=30).wait_for_completed()
                    shooter_controller.init_start_point()
                if try_count % 4 == 0:
                    ep_robot.gimbal.move(yaw=5, yaw_speed=30).wait_for_completed()  # back to center-ish
                    shooter_controller.init_start_point()

                if try_count > 12:  # Give up after ~24 seconds of trying
                    print("Max retries reached. Exiting.")
                    ep_robot.vision.unsub_detect_info(name="marker")
                    ep_robot.close()
                    exit()
                detection_attempt_start_time = time.time()  # Reset timer for next attempt

            time.sleep(0.2)  # Check periodically

        if not target_object or target_object.code != 0:
            print("Failed to acquire a valid target.")
        else:
            print("--- Starting 24-Point API Call and Shooting ---")
            print(f"Symbol: {target_object.symbol_str}, Numbers: {target_object.numbers_tuple}")

            # Call your API
            # The API needs the symbol as a string and the numbers.
            # The order of numbers might matter for your API.
            solution_str = get_24_solution_from_api(target_object.symbol_str, target_object.numbers_tuple)

            if solution_str:
                print(f"API Solution: {solution_str}")
                parts = re.findall(r'\d+|[+\-*/]', solution_str)
                print(f"Parsed solution parts: {parts}")
                available_number_details = list(target_object.tag_info['numbers_details'])

                for part in parts:
                    if part.isdigit():
                        num_val = int(part)
                        if len(str(num_val)) > 1:
                            digits_to_shoot = numberCut(num_val)
                            for digit in digits_to_shoot:
                                found_coords = False
                                for i, num_detail in enumerate(available_number_details):
                                    if num_detail['val'] == digit:
                                        print(
                                            f"Shooting digit: {digit} at ({num_detail['x']:.2f}, {num_detail['y']:.2f})")
                                        shooter_controller.shoot_at_target(num_detail['x'], num_detail['y'],
                                                                           shoot_type=Shooter.CLOSE_PLACE if shooter_controller.calc_zoom_level == 1 else Shooter.NORMAL)
                                        available_number_details.pop(i)
                                        found_coords = True
                                        break
                                if not found_coords:
                                    print(f"Error: Could not find coordinates for digit {digit} from available tags.")
                        else:
                            digit = num_val
                            found_coords = False
                            for i, num_detail in enumerate(available_number_details):
                                if num_detail['val'] == digit:
                                    print(f"Shooting digit: {digit} at ({num_detail['x']:.2f}, {num_detail['y']:.2f})")
                                    shooter_controller.shoot_at_target(num_detail['x'], num_detail['y'],
                                                                       shoot_type=Shooter.CLOSE_PLACE if shooter_controller.calc_zoom_level == 1 else Shooter.NORMAL)
                                    available_number_details.pop(i)
                                    found_coords = True
                                    break
                            if not found_coords:
                                print(f"Error: Could not find coordinates for digit {digit} from available tags.")

                    elif part in ["+", "-", "*", "/"]:
                        # Map parsed symbol to our internal symbol name if needed, or shoot based on API symbol
                        # Assuming API symbol matches our symbol name
                        print(f"Shooting symbol: {target_object.symbol_str}")
                        sx, sy = target_object.get_coords_for_symbol()
                        if sx is not None:
                            shooter_controller.shoot_at_target(sx, sy, shoot_type=Shooter.CLOSE_PLACE)
                        else:
                            print("Error: Could not find coordinates for symbol.")
                    else:
                        print(f"Warning: Unknown part in solution string '{part}'")
                print("Shooting sequence complete.")
            else:
                print("API did not return a solution.")

        print(f"Total program time: {time.time() - program_start_time:.2f} seconds")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback

        traceback.print_exc()
    finally:
        print("Cleaning up...")
        ep_robot.vision.unsub_detect_info(name="marker")
        ep_robot.gimbal.recenter().wait_for_completed()
        ep_robot.close()
        print("Robot connection closed.")