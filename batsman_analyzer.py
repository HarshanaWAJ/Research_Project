import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from collections import defaultdict
import time
import json
import os

class CricketShotAnalyzer:
    def __init__(self, model_path, pitch_length_pixels=None, real_pitch_length=20.12):
        """
        Initialize the cricket shot analyzer
        
        Args:
            model_path (str): Path to the YOLOv8 custom cricket model
            pitch_length_pixels (float, optional): Length of cricket pitch in pixels for speed calibration
            real_pitch_length (float): Real-world length of cricket pitch in meters (default: 20.12m)
        """
        # Load the fine-tuned YOLOv8 model
        self.model = YOLO(model_path)
        
        # Class IDs from custom model
        self.BALL_CLASS = 0  # Cricket ball class ID
        self.BAT_CLASS = 1   # Cricket bat class ID
        
        # Colors for visualization
        self.BALL_COLOR = (0, 255, 0)  # Green
        self.BAT_COLOR = (0, 0, 255)   # Red
        self.IMPACT_COLOR = (255, 0, 255)  # Magenta for impact point
        
        # Speed calibration
        self.pitch_length_pixels = pitch_length_pixels
        self.real_pitch_length = real_pitch_length
        self.speed_conversion_factor = None
        if pitch_length_pixels:
            self.speed_conversion_factor = real_pitch_length / pitch_length_pixels
        
        # Initialize tracker
        self.tracker = sv.ByteTrack()
    
    def check_bbox_intersection(self, box1, box2):
        """Check if two bounding boxes intersect"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        return not (x1_max < x2_min or
                    x1_min > x2_max or
                    y1_max < y2_min or
                    y1_min > y2_max)
    
    def calculate_center(self, bbox):
        """Calculate center point of bounding box"""
        x_min, y_min, x_max, y_max = bbox
        return (int((x_min + x_max) / 2), int((y_min + y_max) / 2))
    
    def calculate_speed(self, points, fps, window_size=3):
        """
        Calculate speed from a series of points
        
        Args:
            points (list): List of (x, y) center points
            fps (float): Video frames per second
            window_size (int): Window size for smoothing
        
        Returns:
            speeds (list): List of speeds in pixels/second
            smoothed_speeds (list): Smoothed speeds
        """
        if len(points) < 2:
            return [], []
        
        # Calculate raw speeds
        speeds = []
        for i in range(1, len(points)):
            # Calculate distance between consecutive points
            dx = points[i][0] - points[i-1][0]
            dy = points[i][1] - points[i-1][1]
            distance = np.sqrt(dx**2 + dy**2)
            
            # Convert to speed (pixels per second)
            speed_pps = distance * fps
            speeds.append(speed_pps)
        
        # Convert to real-world speed if calibration is available
        if self.speed_conversion_factor:
            speeds = [s * self.speed_conversion_factor for s in speeds]
        
        # Apply smoothing using a moving average
        if len(speeds) >= window_size:
            smoothed_speeds = np.convolve(speeds, np.ones(window_size)/window_size, mode='valid').tolist()
        else:
            smoothed_speeds = speeds
        
        return speeds, smoothed_speeds
    
    def get_speed_color(self, speed, max_speed=40):
        """
        Get color based on speed (blue to red gradient)
        
        Args:
            speed (float): Current speed
            max_speed (float): Maximum expected speed for scaling
        
        Returns:
            tuple: BGR color tuple
        """
        # Normalize speed between 0 and 1
        normalized = min(speed / max_speed, 1.0)
        
        # Create color gradient: blue (slow) -> green -> yellow -> red (fast)
        if normalized < 0.25:
            # Blue to cyan
            b = 255
            g = int(255 * normalized * 4)
            r = 0
        elif normalized < 0.5:
            # Cyan to green
            b = int(255 * (0.5 - normalized) * 4)
            g = 255
            r = 0
        elif normalized < 0.75:
            # Green to yellow
            b = 0
            g = 255
            r = int(255 * (normalized - 0.5) * 4)
        else:
            # Yellow to red
            b = 0
            g = int(255 * (1.0 - normalized) * 4)
            r = 255
        
        return (b, g, r)
    
    def detect_significant_movement(self, points, threshold=10):
        """Detect if significant movement has occurred"""
        if len(points) < 3:
            return False
        
        # Get last points
        last_points = points[-3:]
        total_distance = 0
        
        for i in range(1, len(last_points)):
            dx = last_points[i][0] - last_points[i-1][0]
            dy = last_points[i][1] - last_points[i-1][1]
            total_distance += np.sqrt(dx**2 + dy**2)
        
        return total_distance > threshold
    
    def auto_calibrate_pitch(self, frame):
        """
        Automatically try to estimate pitch length in pixels
        This is a placeholder - would need to be implemented based on your specific setup
        """
        # This is just a placeholder example - you would need to implement a proper
        # method based on known markers or field layout
        height, width = frame.shape[:2]
        estimated_pitch_length = width * 0.7  # Just an example
        return estimated_pitch_length
    
    def process_video(self, video_path, output_path, progress_callback=None):
        """
        Process cricket video to track bat and ball speeds
        
        Args:
            video_path (str): Path to input video
            output_path (str): Path to save output video
            progress_callback (function): Callback function for progress updates
        
        Returns:
            dict: Analysis results
        """
        # Initialize video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize video writer
        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (frame_width, frame_height)
        )
        
        # Attempt auto-calibration if not provided
        if self.pitch_length_pixels is None:
            # Read first frame for calibration
            success, frame = cap.read()
            if success:
                self.pitch_length_pixels = self.auto_calibrate_pitch(frame)
                if self.pitch_length_pixels:
                    self.speed_conversion_factor = self.real_pitch_length / self.pitch_length_pixels
                # Reset video to start
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Initialize paths storage
        paths = defaultdict(list)
        object_types = {}  # Track object types (ball or bat)
        
        # Initialize analysis variables
        bat_movement_detected = False
        bat_movement_frame = None
        impact_detected = False
        impact_frame = None
        frame_count = 0
        
        # Store metrics
        ball_speeds = []
        bat_speeds = []
        max_ball_speed = 0
        max_bat_speed = 0
        impact_speed = None
        ball_speed_after_impact = None
        swing_to_impact_time = None
        
        # Process each frame
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            frame_count += 1
            
            # Update progress every 10 frames
            if progress_callback and frame_count % 10 == 0:
                progress = min(int((frame_count / total_frames) * 100), 99)
                progress_callback(progress, f"Processing frame {frame_count}/{total_frames}")
            
            # Create display frame for visualization
            display_frame = frame.copy()
            
            # Add frame counter
            cv2.putText(
                display_frame,
                f"Frame: {frame_count}",
                (10, frame_height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            # Draw historical paths
            for track_id, path_points in paths.items():
                if len(path_points) > 1:
                    # Get object type and base color
                    is_ball = object_types.get(track_id, False)
                    base_color = self.BALL_COLOR if is_ball else self.BAT_COLOR
                    
                    # Calculate speeds for this path
                    speeds, smoothed_speeds = self.calculate_speed(path_points, fps)
                    
                    # Draw path with speed-based colors
                    for i in range(1, len(path_points)):
                        # Get color based on speed (if available)
                        if i <= len(smoothed_speeds):
                            # Use last speed for the most recent points
                            speed_idx = min(i-1, len(smoothed_speeds)-1)
                            current_speed = smoothed_speeds[speed_idx]
                            color = self.get_speed_color(current_speed)
                        else:
                            color = base_color
                        
                        # Draw line segment
                        cv2.line(
                            display_frame,
                            path_points[i-1],
                            path_points[i],
                            color,
                            2
                        )
            
            # Run object detection
            results = self.model(frame, conf=0.35)
            
            # Convert to supervision Detections format
            # FIX: Updated to handle newer versions of supervision library and YOLO
            if len(results) > 0:
                boxes = results[0].boxes
                detections = sv.Detections(
                    xyxy=boxes.xyxy.cpu().numpy(),
                    confidence=boxes.conf.cpu().numpy(),
                    class_id=boxes.cls.cpu().numpy().astype(int)
                )
                
                current_bats = {}
                current_balls = {}
                
                if len(detections) > 0:
                    # Update tracks
                    tracked_detections = self.tracker.update_with_detections(detections)
                    
                    # Process each detection
                    for detection_idx in range(len(tracked_detections)):
                        bbox = tracked_detections.xyxy[detection_idx]
                        tracker_id = tracked_detections.tracker_id[detection_idx]
                        class_id = tracked_detections.class_id[detection_idx]
                        confidence = tracked_detections.confidence[detection_idx]
                        
                        # Determine if it's a ball or bat
                        is_ball = (class_id == self.BALL_CLASS)
                        
                        # Update object type
                        object_types[tracker_id] = is_ball
                        
                        # Store center point
                        center = self.calculate_center(bbox)
                        paths[tracker_id].append(center)
                        
                        # Store current detections for impact detection
                        if is_ball:
                            current_balls[tracker_id] = bbox
                        else:
                            current_bats[tracker_id] = bbox
                        
                        # Draw bounding box
                        color = self.BALL_COLOR if is_ball else self.BAT_COLOR
                        cv2.rectangle(
                            display_frame,
                            (int(bbox[0]), int(bbox[1])),
                            (int(bbox[2]), int(bbox[3])),
                            color,
                            2
                        )
                        
                        # Calculate and update speed
                        if len(paths[tracker_id]) >= 3:
                            _, smoothed_speeds = self.calculate_speed(paths[tracker_id], fps)
                            if smoothed_speeds:
                                current_speed = smoothed_speeds[-1]
                                
                                # Update max speeds
                                if is_ball:
                                    ball_speeds.append(current_speed)
                                    if current_speed > max_ball_speed:
                                        max_ball_speed = current_speed
                                else:
                                    bat_speeds.append(current_speed)
                                    if current_speed > max_bat_speed:
                                        max_bat_speed = current_speed
                                
                                # Add speed label
                                speed_unit = "km/h" if self.speed_conversion_factor else "px/s"
                                cv2.putText(
                                    display_frame,
                                    f"{current_speed:.1f} {speed_unit}",
                                    (int(bbox[0]), int(bbox[1] - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    color,
                                    2
                                )
                    
                    # Check for bat movement
                    for bat_id, bat_bbox in current_bats.items():
                        if not bat_movement_detected and bat_id in paths:
                            if self.detect_significant_movement(paths[bat_id]):
                                bat_movement_detected = True
                                bat_movement_frame = frame_count
                    
                    # Check for impact
                    if bat_movement_detected and not impact_detected:
                        for ball_id, ball_bbox in current_balls.items():
                            for bat_id, bat_bbox in current_bats.items():
                                if self.check_bbox_intersection(ball_bbox, bat_bbox):
                                    impact_detected = True
                                    impact_frame = frame_count
                                    
                                    # Calculate impact metrics
                                    if ball_id in paths and len(paths[ball_id]) >= 3:
                                        _, ball_smoothed_speeds = self.calculate_speed(paths[ball_id], fps)
                                        if ball_smoothed_speeds:
                                            impact_speed = ball_smoothed_speeds[-1]
                                    
                                    # Calculate time from swing to impact
                                    if bat_movement_frame:
                                        swing_to_impact_time = (impact_frame - bat_movement_frame) / fps
                                    
                                    # Draw impact marker
                                    impact_center = self.calculate_center(ball_bbox)
                                    cv2.circle(
                                        display_frame,
                                        impact_center,
                                        15,
                                        self.IMPACT_COLOR,
                                        -1
                                    )
                                    cv2.circle(
                                        display_frame,
                                        impact_center,
                                        15,
                                        (255, 255, 255),
                                        2
                                    )
                
                # Check for ball speed after impact
                if impact_detected and not ball_speed_after_impact:
                    # Look for ball speed in frames after impact
                    for ball_id, ball_bbox in current_balls.items():
                        if frame_count > impact_frame + 5 and ball_id in paths:
                            recent_path = paths[ball_id][-5:]
                            if len(recent_path) >= 3:
                                _, smoothed_speeds = self.calculate_speed(recent_path, fps)
                                if smoothed_speeds and len(smoothed_speeds) > 0:
                                    ball_speed_after_impact = smoothed_speeds[-1]
            
            # Add timing information
            if bat_movement_detected:
                if impact_detected:
                    status = "Impact Detected"
                    time_info = f"Swing to Impact: {swing_to_impact_time:.3f}s"
                    color = (0, 255, 0)  # Green
                else:
                    elapsed_time = (frame_count - bat_movement_frame) / fps
                    status = "Swing Detected"
                    time_info = f"Elapsed: {elapsed_time:.3f}s"
                    color = (0, 165, 255)  # Orange
                
                cv2.putText(
                    display_frame,
                    status,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )
                
                cv2.putText(
                    display_frame,
                    time_info,
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )
            
            # Add speed metrics
            metrics_y = 90
            if max_bat_speed > 0:
                cv2.putText(
                    display_frame,
                    f"Max Bat Speed: {max_bat_speed:.1f} {'km/h' if self.speed_conversion_factor else 'px/s'}",
                    (10, metrics_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    self.BAT_COLOR,
                    2
                )
                metrics_y += 30
            
            if max_ball_speed > 0:
                cv2.putText(
                    display_frame,
                    f"Max Ball Speed: {max_ball_speed:.1f} {'km/h' if self.speed_conversion_factor else 'px/s'}",
                    (10, metrics_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    self.BALL_COLOR,
                    2
                )
                metrics_y += 30
            
            if impact_speed:
                cv2.putText(
                    display_frame,
                    f"Impact Speed: {impact_speed:.1f} {'km/h' if self.speed_conversion_factor else 'px/s'}",
                    (10, metrics_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    self.IMPACT_COLOR,
                    2
                )
                metrics_y += 30
            
            if ball_speed_after_impact:
                cv2.putText(
                    display_frame,
                    f"After Impact: {ball_speed_after_impact:.1f} {'km/h' if self.speed_conversion_factor else 'px/s'}",
                    (10, metrics_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    self.BALL_COLOR,
                    2
                )
            
            # Write frame to output video
            out.write(display_frame)
        
        # Release resources
        cap.release()
        out.release()
        
        # Prepare results
        results = {
            "metrics": {
                "max_bat_speed": max_bat_speed,
                "max_ball_speed": max_ball_speed,
                "impact_speed": impact_speed if impact_speed else None,
                "ball_speed_after_impact": ball_speed_after_impact if ball_speed_after_impact else None,
                "swing_to_impact_time": swing_to_impact_time if swing_to_impact_time else None,
                "bat_movement_frame": bat_movement_frame,
                "impact_frame": impact_frame,
                "fps": fps,
                "total_frames": total_frames,
                "video_length_seconds": total_frames / fps,
                "speed_unit": "km/h" if self.speed_conversion_factor else "pixels/s"
            },
            "file_info": {
                "input_video": os.path.basename(video_path),
                "output_video": os.path.basename(output_path),
                "processed_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "calibration": {
                    "pitch_length_pixels": self.pitch_length_pixels,
                    "real_pitch_length": self.real_pitch_length,
                    "conversion_factor": self.speed_conversion_factor
                }
            }
        }
        
        # Notify completion
        if progress_callback:
            progress_callback(100, "Processing complete")
        
        return results
    
    def save_results(self, results, output_path):
        """Save analysis results to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)