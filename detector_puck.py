from collections import deque
import cv2
import numpy as np
import time
from typing import Optional

from vector_motion import VectorMotion


class DetectorPuck:
    MAX_LENGTH = 5

    def __init__(self):
        # Define yellow color range in HSV
        self.lower_yellow = np.array([20, 100, 100])
        self.upper_yellow = np.array([40, 255, 255])

        # Minimum area for check mark detection (adjust based on your needs)
        self.min_area = 10
        self.max_area = 200

        # Motion tracking
        self.previous_positions = deque(maxlen=self.MAX_LENGTH)
        self.previous_times = deque(maxlen=self.MAX_LENGTH)

    def detect_check_marks(self, image: np.ndarray) -> list:
        """Detect yellow check marks in the image"""
        # Convert BGR to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Create mask for yellow color
        mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)

        # Clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_marks = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area < area < self.max_area:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)

                # Check aspect ratio (approximately 2:1 for 10x5 check mark)
                aspect_ratio = w / h
                if 1.5 <= aspect_ratio <= 3.0:
                    center_x = x + w / 2
                    center_y = y + h / 2
                    detected_marks.append((center_x, center_y, w, h, area))

        return detected_marks

    def calculate_motion_vector(self) -> Optional[VectorMotion]:
        """Calculate motion vector based on previous positions"""
        if len(self.previous_positions) < 2:
            return None

        # Use the two most recent positions for more responsive tracking
        recent_positions = list(self.previous_positions)
        recent_times = list(self.previous_times)

        if len(recent_positions) < 2:
            return None

        # Calculate displacement and time difference
        prev_x, prev_y = recent_positions[0]
        curr_x, curr_y = recent_positions[-1]
        prev_time = recent_times[0]
        current_time = recent_times[-1]

        time_diff = current_time - prev_time
        if time_diff == 0:
            return None

        # Calculate velocity components (pixels per second)
        dx = (curr_x - prev_x) / time_diff
        dy = (curr_y - prev_y) / time_diff

        # Calculate speed
        speed = np.sqrt(dx**2 + dy**2)

        return VectorMotion(dx, dy, speed)

    def update_position(self, position: tuple[float, float]):
        """Update position history for motion tracking"""
        self.previous_positions.append(position)
        self.previous_times.append(time.time())
