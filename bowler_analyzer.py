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