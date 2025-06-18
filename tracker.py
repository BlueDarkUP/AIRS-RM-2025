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
        # Initial state based on the first detection's center
        center_x = (initial_bbox_xyxy[0] + initial_bbox_xyxy[2]) / 2
        center_y = (initial_bbox_xyxy[1] + initial_bbox_xyxy[3]) / 2
        # State: [x, y, vx, vy] (position and velocity)
        self.kf.x = np.array([center_x, center_y, 0., 0.])

        self.class_id = class_id  # Current class ID (can be refined by color detection)
        self.model_class_id = class_id  # Original class ID from model
        self.conf = conf  # Confidence of the last matched detection

        # Store the current bounding box based on the last detection or KF prediction.
        # This is used for drawing and deriving width/height/area.
        self.current_bbox_xyxy = np.array(initial_bbox_xyxy).astype(np.float32)

        self.hits = 1  # Number of times this track has been successfully updated by a detection
        self.age = 0  # Total age of the track in frames (incremented every frame)
        self.time_since_update = 0  # Frames since last successful update by a detection

        self.predicted_aim_point = None
        self.color_detection_suffix = ""  # e.g., "(R)", "(B)"
        self.display_label_name = ""  # Label name for display (e.g., "RedArmor")
        self.track_color = (random.randint(50, 200), random.randint(50, 200),
                            random.randint(50, 200))  # Use random for non-selected targets

        self.rotation_angle = None  # Estimated rotation angle
        self.estimated_distance = None  # Estimated distance in meters

        # Scoring related attributes (updated by detector.py)
        self.score = 0.0  # Combined score of the track for target selection
        self.last_dist_score = 0.0  # Individual component scores for debugging
        self.last_angle_score = 0.0
        self.last_size_score = 0.0

    def _create_kalman_filter(self, dt, process_noise_std, measurement_noise_std):
        """Initializes and returns a KalmanFilter object."""
        kf = KalmanFilter(dim_x=4, dim_z=2)  # State: [x, y, vx, vy], Measurement: [x, y]

        # State Transition Matrix (F): Defines how the state changes over time step dt
        kf.F = np.array([[1, 0, dt, 0],
                         [0, 1, 0, dt],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])

        # Measurement Function (H): Maps the state vector to the measurement vector
        kf.H = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]])

        # Covariance Matrix (P) - Initial uncertainty
        kf.P *= 500.  # Large initial uncertainty for all states
        kf.P[2, 2] *= 1000  # Higher uncertainty for velocity components to allow faster convergence
        kf.P[3, 3] *= 1000

        # Measurement Noise Covariance Matrix (R) - Noise in measurements
        kf.R = np.eye(2) * (measurement_noise_std ** 2)

        # Process Noise Covariance Matrix (Q) - Uncertainty in the model itself (e.g., sudden acceleration)
        q_var = process_noise_std ** 2
        # Standard constant velocity model process noise matrix
        kf.Q = np.array([[(dt ** 3) / 3, 0, (dt ** 2) / 2, 0],
                         [0, (dt ** 3) / 3, 0, (dt ** 2) / 2],
                         [(dt ** 2) / 2, 0, dt, 0],
                         [0, (dt ** 2) / 2, 0, dt]]) * q_var
        return kf

    def predict_kf(self):
        """Predicts the next state of the track using the Kalman filter."""
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        # Update current_bbox_xyxy based on predicted state's center, keeping last known dimensions
        self.current_bbox_xyxy = self._get_bbox_from_kf_state(self.kf.x[:2])
        return self.current_bbox_xyxy

    def update_kf(self, detection_bbox_xyxy, model_class_id, conf):
        """Updates the track state with a new detection."""
        # Calculate center point from detection bbox for measurement
        center_x = (detection_bbox_xyxy[0] + detection_bbox_xyxy[2]) / 2
        center_y = (detection_bbox_xyxy[1] + detection_bbox_xyxy[3]) / 2
        measurement = np.array([center_x, center_y])

        self.kf.update(measurement)  # Update Kalman filter state

        # Update current_bbox_xyxy directly from detection for most accurate size/position
        self.current_bbox_xyxy = np.array(detection_bbox_xyxy).astype(np.float32)
        self.model_class_id = model_class_id  # Update original model class ID
        self.conf = conf  # Update confidence
        self.hits += 1
        self.time_since_update = 0  # Reset frames since last update

    def _get_bbox_from_kf_state(self, center_coords):
        """
        Helper to construct a bounding box from Kalman state (center_x, center_y)
        and the last known width/height.
        """
        width = self.width  # Use the property which gets from current_bbox_xyxy
        height = self.height

        x1 = center_coords[0] - width / 2
        y1 = center_coords[1] - height / 2
        x2 = center_coords[0] + width / 2
        y2 = center_coords[1] + height / 2
        return np.array([x1, y1, x2, y2])

    def get_current_bbox_for_display(self):
        """Returns the current bounding box for display, reflecting the last updated or predicted state."""
        return self.current_bbox_xyxy

    @property
    def center_x(self):
        """Returns the current center X coordinate of the track's bounding box."""
        return (self.current_bbox_xyxy[0] + self.current_bbox_xyxy[2]) / 2

    @property
    def center_y(self):
        """Returns the current center Y coordinate of the track's bounding box."""
        return (self.current_bbox_xyxy[1] + self.current_bbox_xyxy[3]) / 2

    @property
    def width(self):
        """Returns the current width of the track's bounding box."""
        return self.current_bbox_xyxy[2] - self.current_bbox_xyxy[0]

    @property
    def height(self):
        """Returns the current height of the track's bounding box."""
        return self.current_bbox_xyxy[3] - self.current_bbox_xyxy[1]

    @property
    def area(self):
        """Returns the current area of the track's bounding box."""
        return self.width * self.height

    def calculate_aim_point(self):
        """
        Calculates the predicted aim point based on current velocity.
        """
        # Only calculate aim point if the track's class is designated for auto-aiming
        if self.class_id not in cfg.AUTO_AIM_CLASSES:
            self.predicted_aim_point = None
            return None

        curr_x, curr_y, curr_vx, curr_vy = self.kf.x  # Get current position and velocity from KF state

        # If velocity is very low, aim directly at the current center point to avoid prediction jitter
        velocity_magnitude_sq = curr_vx ** 2 + curr_vy ** 2
        if velocity_magnitude_sq < cfg.MIN_VELOCITY_FOR_PREDICTION_SQ:
            aim_x = curr_x
            aim_y = curr_y
        else:
            # Predict aim point: current position + velocity * prediction_frames
            aim_x = curr_x + curr_vx * cfg.AIM_PREDICTION_FRAMES
            aim_y = curr_y + curr_vy * cfg.AIM_PREDICTION_FRAMES

        self.predicted_aim_point = (int(aim_x), int(aim_y))
        return self.predicted_aim_point

    @property
    def is_tentative(self):
        """Checks if the track is still in a tentative state (hasn't met minimum hits for activation)."""
        return self.hits < cfg.MIN_HITS_TO_ACTIVATE

    @property
    def is_lost(self):
        """Checks if the track is considered lost (too many frames without an update)."""
        return self.time_since_update > cfg.MAX_FRAMES_SINCE_UPDATE