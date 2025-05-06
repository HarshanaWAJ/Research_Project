import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from collections import defaultdict
import time
import json
import os

class BowlerActionAnalyzer:
    """Analyzer for cricket bowling actions with focus on arm angle measurement using YOLOv11n-pose"""
    
    def __init__(self, model_path='yolo11n-pose.pt'):
        """
        Initialize the bowling action analyzer
        
        Args:
            model_path (str): Path to YOLOv11n pose model
        """
        self.model_path = model_path
        self.model = None
        
        # Constants for arm angle violation threshold (ICC rules)
        self.LEGAL_ANGLE_THRESHOLD = 15.0  # degrees
        
        # Colors for visualization
        self.LINE_COLOR = (0, 255, 0)  # Green for arm lines
        self.EXTENSION_COLOR = (255, 0, 0)  # Blue for extended reference line
        self.KEYPOINT_COLOR = (255, 0, 255)  # Magenta for keypoints
        self.WARNING_COLOR = (0, 0, 255)  # Red for warnings
        
        # Keypoint indices for YOLOv11n-pose (COCO format)
        self.NOSE = 0
        self.LEFT_EYE = 1
        self.RIGHT_EYE = 2
        self.LEFT_EAR = 3
        self.RIGHT_EAR = 4
        self.LEFT_SHOULDER = 5
        self.RIGHT_SHOULDER = 6
        self.LEFT_ELBOW = 7
        self.RIGHT_ELBOW = 8
        self.LEFT_WRIST = 9
        self.RIGHT_WRIST = 10
        self.LEFT_HIP = 11
        self.RIGHT_HIP = 12
        self.LEFT_KNEE = 13
        self.RIGHT_KNEE = 14
        self.LEFT_ANKLE = 15
        self.RIGHT_ANKLE = 16
    
    def load_model(self):
        """Load YOLOv11n pose estimation model"""
        if self.model is None:
            try:
                from ultralytics import YOLO
                self.model = YOLO(self.model_path)
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                raise
        return self.model

    def calculate_angle(self, point1, point2, point3):
    """
    Calculate angle between three points with improved error handling

    Args:
        point1, point2, point3: Coordinate points (point2 is the vertex)

    Returns:
        float: Angle in degrees, adjusted for bowling analysis
    """
    import numpy as np

    if None in (point1, point2, point3):
        return None

    a = np.array(point1)
    b = np.array(point2)
    c = np.array(point3)

    ba = a - b
    bc = c - b

    # Check for zero-length vectors to avoid division by zero
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba < 1e-10 or norm_bc < 1e-10:
        return None  # Return None for degenerate angles

    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    # Ensure cosine is within valid range [-1.0, 1.0]
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))

    # Adjust the angle for bowling analysis
    # If the angle is close to 180, we want to measure how much it deviates from straight
    if angle > 90:
        # Calculate how much the arm deviates from being completely straight
        straightening_angle = 180 - angle
        return straightening_angle

    return angle

    def extend_line(self, point1, point2, extension_length=100):
        """
        Extend a line beyond point2 in the same direction
        
        Args:
            point1, point2: Line points
            extension_length: Length to extend
            
        Returns:
            tuple: Coordinates of extended point
        """
        import numpy as np
        
        if None in (point1, point2):
            return None
        direction = np.array(point2) - np.array(point1)
        if np.all(direction == 0):
            return None
        direction = direction / np.linalg.norm(direction)
        extended_point = np.array(point2) + direction * extension_length
        return tuple(map(int, extended_point))

        def get_keypoint_coords(self, keypoints, idx, conf_threshold=0.5):
        """
        Extract coordinates from keypoints with confidence check
        
        Args:
            keypoints: Keypoints array [x, y, conf]
            idx: Index of keypoint to extract
            conf_threshold: Minimum confidence for valid keypoint
            
        Returns:
            tuple: (x, y) coordinates or None
        """
        try:
            if keypoints[idx][2] >= conf_threshold:
                return tuple(map(int, keypoints[idx][:2]))
        except (IndexError, TypeError) as e:
            print(f"Error getting keypoint coordinates: {str(e)}")
            return None
        return None

         def analyze_bowling_action(self, video_path, bowling_arm='right', output_path=None, progress_callback=None):
        """
        Analyze bowling action from video
        
        Args:
            video_path: Path to the video file
            bowling_arm: 'right' or 'left'
            output_path: Path to save analyzed video
            progress_callback: Function to report progress
            
        Returns:
            dict: Analysis results
        """
        import cv2
        import os
        import time
        import numpy as np

        # Set default output path if not provided
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = f"{base_name}_analyzed.mp4"

        # Load model
        try:
            model = self.load_model()
            print("Model loaded successfully:", self.model_path)
        except Exception as e:
            print(f"Failed to load model: {str(e)}")
            raise   

         # Initialize video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error: Could not open video file {video_path}")
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        # Define keypoint indices based on bowling arm
        if bowling_arm.lower() == 'right':
            shoulder_idx = self.RIGHT_SHOULDER
            elbow_idx = self.RIGHT_ELBOW
            wrist_idx = self.RIGHT_WRIST
        else:
            shoulder_idx = self.LEFT_SHOULDER
            elbow_idx = self.LEFT_ELBOW
            wrist_idx = self.LEFT_WRIST

        # Initialize tracking variables
        frame_count = 0
        detection_failures = 0
        arm_angles = []
        max_angle = 0
        critical_frames = []
        
        # Progress update interval
        update_interval = max(1, total_frames // 100)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            
        # Update progress
            if progress_callback and frame_count % update_interval == 0:
                progress = min(int(frame_count / total_frames * 100), 99)
                progress_callback(progress, f"Processing frame {frame_count}/{total_frames}")

            try:
                # Run YOLOv11n pose prediction
                results = model(frame, verbose=False)

                if len(results) > 0 and hasattr(results[0], 'keypoints') and len(results[0].keypoints.data) > 0:
                    # Find the bowler in the frame - usually the person most centered
                    # if multiple people are detected
                    best_person_idx = 0
                    if len(results[0].keypoints.data) > 1:
                        # If multiple people, find the person closest to center
                        center_x = frame_width / 2
                        center_y = frame_height / 2
                        min_distance = float('inf')
                        
                        for i, kpts in enumerate(results[0].keypoints.data):
                            # Use the nose keypoint as reference
                            if kpts[self.NOSE][2] > 0.5:  # Check confidence
                                nose_x, nose_y = kpts[self.NOSE][0], kpts[self.NOSE][1]
                                dist = ((nose_x - center_x) ** 2 + (nose_y - center_y) ** 2) ** 0.5
                                if dist < min_distance:
                                    min_distance = dist
                                    best_person_idx = i