import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import config as cfg  # Use alias for brevity
import random  # 引入random用于生成随机颜色


class Track:
    def __init__(self, initial_bbox_xyxy, class_id, conf, track_id):
        self.track_id = track_id
        # Initialize Kalman Filter
        self.kf = self._create_kalman_filter(
            cfg.KF_DT,
            cfg.KF_PROCESS_NOISE_STD,
            cfg.KF_MEASUREMENT_NOISE_STD
        )
        # Initial state based on the first detection
        center_x = (initial_bbox_xyxy[0] + initial_bbox_xyxy[2]) / 2
        center_y = (initial_bbox_xyxy[1] + initial_bbox_xyxy[3]) / 2
        # State: [x, y, vx, vy]
        # x, y 是目标中心点坐标，vx, vy 是速度
        self.kf.x = np.array([center_x, center_y, 0., 0.])

        self.class_id = class_id  # Class ID after color check (can change)
        self.model_class_id = class_id  # Original class ID from model
        self.conf = conf
        self.last_bbox_xyxy = np.array(initial_bbox_xyxy)  # Store width/height reference

        self.hits = 1  # Number of times this track has been successfully updated
        self.age = 0  # Total age of the track in frames
        self.time_since_update = 0  # Frames since last successful update

        self.predicted_aim_point = None
        self.color_detection_suffix = ""  # e.g., "(R)", "(B)"
        self.display_label_name = ""  # Label name for display
        self.track_color = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))  # 使用random生成颜色

        self.rotation_angle = None
        self.estimated_distance = None

    def _create_kalman_filter(self, dt, process_noise_std, measurement_noise_std):
        """Initializes and returns a KalmanFilter object."""
        kf = KalmanFilter(dim_x=4, dim_z=2)  # State: [x, y, vx, vy], Measurement: [x, y]

        # State Transition Matrix (F): 定义状态如何随时间步长dt变化
        # x_k = x_{k-1} + vx_{k-1} * dt
        # y_k = y_{k-1} + vy_{k-1} * dt
        # vx_k = vx_{k-1}
        # vy_k = vy_{k-1}
        kf.F = np.array([[1, 0, dt, 0],
                         [0, 1, 0, dt],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])

        # Measurement Function (H): 将状态向量映射到测量向量
        # 测量只包含x, y坐标
        kf.H = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]])

        # Covariance Matrix (P) - 初始不确定性
        kf.P *= 500.  # 初始协方差矩阵，较大的值表示对初始状态不确定
        kf.P[2, 2] *= 1000  # 对速度分量赋予更高不确定性，以便快速收敛
        kf.P[3, 3] *= 1000

        # Measurement Noise Covariance Matrix (R) - 测量噪声
        kf.R = np.eye(2) * (measurement_noise_std ** 2)  # 根据测量标准差设置

        # Process Noise Covariance Matrix (Q) - 过程噪声
        # 描述模型自身的不确定性（如目标突然加速、减速等）
        q_var = process_noise_std ** 2
        # 经典的恒定速度模型过程噪声矩阵
        kf.Q = np.array([[(dt ** 3) / 3, 0, (dt ** 2) / 2, 0],
                         [0, (dt ** 3) / 3, 0, (dt ** 2) / 2],
                         [(dt ** 2) / 2, 0, dt, 0],
                         [0, (dt ** 2) / 2, 0, dt]]) * q_var
        # 另一种简单方式 (如果使用 block_size=2, order_by_dim=False, 且 dim=2 对应 vx,vy):
        # kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=process_noise_std**2, block_size=2, order_by_dim=False)

        return kf

    def predict_kf(self):
        """Predicts the next state of the track using the Kalman filter."""
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self.get_current_bbox_from_state()  # Return predicted bbox

    def update_kf(self, detection_bbox_xyxy, model_class_id, conf):
        """Updates the track state with a new detection."""
        # 从检测框计算中心点作为测量值
        center_x = (detection_bbox_xyxy[0] + detection_bbox_xyxy[2]) / 2
        center_y = (detection_bbox_xyxy[1] + detection_bbox_xyxy[3]) / 2
        measurement = np.array([center_x, center_y])

        self.kf.update(measurement)  # 更新卡尔曼滤波器状态

        self.last_bbox_xyxy = np.array(detection_bbox_xyxy)  # Update for size reference
        self.model_class_id = model_class_id  # Update original model class ID
        self.conf = conf
        self.hits += 1
        self.time_since_update = 0

    def get_current_bbox_from_state(self):
        """
        Estimates the current bounding box from the Kalman filter's state。
        假设宽度和高度从最近的检测结果中保持相对不变。
        """
        center_x, center_y = self.kf.x[0], self.kf.x[1]
        width = self.last_bbox_xyxy[2] - self.last_bbox_xyxy[0]
        height = self.last_bbox_xyxy[3] - self.last_bbox_xyxy[1]

        x1 = center_x - width / 2
        y1 = center_y - height / 2
        x2 = center_x + width / 2
        y2 = center_y + height / 2
        return np.array([x1, y1, x2, y2])

    def calculate_aim_point(self):
        """
        根据当前速度计算预测的瞄准点。
        """
        # 只有在目标类别是自动瞄准的类型时才计算预瞄点
        if self.class_id not in cfg.AUTO_AIM_CLASSES:
            self.predicted_aim_point = None
            return None

        curr_x, curr_y, curr_vx, curr_vy = self.kf.x  # 从卡尔曼滤波器获取当前状态：位置和速度

        # 如果速度非常低，直接瞄准当前中心点，避免预测抖动
        velocity_magnitude_sq = curr_vx ** 2 + curr_vy ** 2
        if velocity_magnitude_sq < cfg.MIN_VELOCITY_FOR_PREDICTION_SQ:  # 使用配置的阈值
            aim_x = curr_x
            aim_y = curr_y
        else:
            # 预测预瞄点：当前位置 + 速度 * 预测帧数
            # 这里的 cfg.AIM_PREDICTION_FRAMES 就是您需要调整的“B时间”参数
            aim_x = curr_x + curr_vx * cfg.AIM_PREDICTION_FRAMES
            aim_y = curr_y + curr_vy * cfg.AIM_PREDICTION_FRAMES

        self.predicted_aim_point = (int(aim_x), int(aim_y))
        return self.predicted_aim_point

    @property
    def is_tentative(self):
        """检查追踪是否仍处于临时状态（未达到激活所需的击中次数）。"""
        return self.hits < cfg.MIN_HITS_TO_ACTIVATE

    @property
    def is_lost(self):
        """检查追踪是否被认为是丢失状态（未更新的帧数过多）。"""
        return self.time_since_update > cfg.MAX_FRAMES_SINCE_UPDATE