# tracker.py
import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import config as cfg  # Use alias for brevity


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
        self.kf.x = np.array([center_x, center_y, 0., 0.])  # state: [x, y, vx, vy]

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
        self.track_color = (np.random.randint(50, 200), np.random.randint(50, 200), np.random.randint(50, 200))

        self.rotation_angle = None
        self.estimated_distance = None

    def _create_kalman_filter(self, dt, process_noise_std, measurement_noise_std):
        """Initializes and returns a KalmanFilter object."""
        kf = KalmanFilter(dim_x=4, dim_z=2)  # State: [x, y, vx, vy], Measurement: [x, y]
        # State Transition Matrix (F)
        kf.F = np.array([[1, 0, dt, 0],
                         [0, 1, 0, dt],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        # Measurement Function (H)
        kf.H = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]])
        # Covariance Matrix (P) - initial uncertainty
        kf.P *= 500.
        kf.P[2, 2] *= 1000  # Higher uncertainty for velocity
        kf.P[3, 3] *= 1000
        # Measurement Noise Covariance Matrix (R)
        kf.R = np.eye(2) * (measurement_noise_std ** 2)
        # Process Noise Covariance Matrix (Q)
        # Using filterpy.common.Q_discrete_white_noise for simplicity
        # Or define manually as in original code:
        q_var = process_noise_std ** 2
        kf.Q = np.array([[(dt ** 3) / 3, 0, (dt ** 2) / 2, 0],
                         [0, (dt ** 3) / 3, 0, (dt ** 2) / 2],
                         [(dt ** 2) / 2, 0, dt, 0],
                         [0, (dt ** 2) / 2, 0, dt]]) * q_var
        # kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=process_noise_std**2, block_size=2, order_by_dim=False) # Alternative
        return kf

    def predict_kf(self):
        """Predicts the next state of the track using the Kalman filter."""
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self.get_current_bbox_from_state()  # Return predicted bbox

    def update_kf(self, detection_bbox_xyxy, model_class_id, conf):
        """Updates the track state with a new detection."""
        center_x = (detection_bbox_xyxy[0] + detection_bbox_xyxy[2]) / 2
        center_y = (detection_bbox_xyxy[1] + detection_bbox_xyxy[3]) / 2
        measurement = np.array([center_x, center_y])

        self.kf.update(measurement)

        self.last_bbox_xyxy = np.array(detection_bbox_xyxy)  # Update for size reference
        self.model_class_id = model_class_id  # Update original model class ID
        self.conf = conf
        self.hits += 1
        self.time_since_update = 0

    def get_current_bbox_from_state(self):
        """
        Estimates the current bounding box from the Kalman filter's state.
        Assumes width and height remain relatively constant from the last detection.
        """
        center_x, center_y = self.kf.x[0], self.kf.x[1]
        width = self.last_bbox_xyxy[2] - self.last_bbox_xyxy[0]
        height = self.last_bbox_xyxy[3] - self.last_bbox_xyxy[1]

        x1 = center_x - width / 2
        y1 = center_y - height / 2
        x2 = center_x + width / 2
        y2 = center_y + height / 2
        return np.array([x1, y1, x2, y2])

    def calculate_aim_point(self, prediction_dt_aim=cfg.AIM_PREDICTION_FRAMES):
        """
        Calculates the predicted aiming point based on current velocity.
        """
        if self.class_id not in cfg.AUTO_AIM_CLASSES:
            self.predicted_aim_point = None
            return None

        curr_x, curr_y, curr_vx, curr_vy = self.kf.x

        # If velocity is very low, aim at current center
        velocity_magnitude_sq = curr_vx ** 2 + curr_vy ** 2
        if velocity_magnitude_sq < 0.5:  # Threshold to avoid jittery prediction for static/slow targets
            aim_x = curr_x
            aim_y = curr_y
        else:
            aim_x = curr_x + curr_vx * prediction_dt_aim
            aim_y = curr_y + curr_vy * prediction_dt_aim

        self.predicted_aim_point = (int(aim_x), int(aim_y))
        return self.predicted_aim_point

    @property
    def is_tentative(self):
        """Checks if the track is still tentative (not enough hits)."""
        return self.hits < cfg.MIN_HITS_TO_ACTIVATE

    @property
    def is_lost(self):
        """Checks if the track is considered lost (too many frames since update)."""
        return self.time_since_update > cfg.MAX_FRAMES_SINCE_UPDATE