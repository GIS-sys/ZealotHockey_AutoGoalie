import cv2
import numpy as np
import pyautogui
import time
from dataclasses import dataclass
from typing import Optional, Tuple
import threading
from collections import deque

@dataclass
class MotionVector:
    dx: float
    dy: float
    speed: float
    
    def predict_position(self, current_pos: Tuple[float, float], time_sec: float) -> Tuple[float, float]:
        """Predict future position based on current motion vector"""
        x, y = current_pos
        future_x = x + self.dx * time_sec
        future_y = y + self.dy * time_sec
        return (future_x, future_y)

class YellowCheckMarkDetector:
    MAX_LENGTH = 10

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
    
    def calculate_motion_vector(self) -> Optional[MotionVector]:
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
        
        return MotionVector(dx, dy, speed)
    
    def update_position(self, position: Tuple[float, float]):
        """Update position history for motion tracking"""
        self.previous_positions.append(position)
        self.previous_times.append(time.time())

class ScreenCapture:
    def __init__(self, target_fps: int = 30):
        self.target_fps = target_fps
        self.frame_delay = 1.0 / target_fps
        self.current_frame = None
        self.lock = threading.Lock()
        self.running = False
        
    def start_capture(self):
        """Start screen capture in a separate thread"""
        self.running = True
        capture_thread = threading.Thread(target=self._capture_loop)
        capture_thread.daemon = True
        capture_thread.start()
        
    def stop_capture(self):
        """Stop screen capture"""
        self.running = False
        
    def _capture_loop(self):
        """Main capture loop running in separate thread"""
        while self.running:
            start_time = time.time()
            
            # Capture screen
            screenshot = pyautogui.screenshot()
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Update current frame
            with self.lock:
                self.current_frame = frame
            
            # Maintain target FPS
            elapsed = time.time() - start_time
            sleep_time = max(0, self.frame_delay - elapsed)
            time.sleep(sleep_time)
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get the current frame"""
        with self.lock:
            return self.current_frame.copy() if self.current_frame is not None else None

class CheckMarkTracker:
    def __init__(self, target_fps: int = 30):
        self.screen_capture = ScreenCapture(target_fps)
        self.detector = YellowCheckMarkDetector()
        self.running = False
        self.target_fps = target_fps
        self.frame_delay = 1.0 / target_fps
        
        # Performance monitoring
        self.frame_count = 0
        self.start_time = time.time()
        
    def start_tracking(self):
        """Start the tracking system"""
        print("Starting yellow check mark tracker...")
        print("Press Ctrl+C to stop")
        
        self.screen_capture.start_capture()
        self.running = True
        self.start_time = time.time()
        self.frame_count = 0
        
        try:
            self._tracking_loop()
        except KeyboardInterrupt:
            self.stop_tracking()
    
    def stop_tracking(self):
        """Stop the tracking system"""
        self.running = False
        self.screen_capture.stop_capture()
        
        # Calculate and display performance statistics
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        print(f"\nTracking stopped. Average FPS: {fps:.2f}")
    
    def _tracking_loop(self):
        """Main tracking loop"""
        while self.running:
            loop_start = time.time()
            
            frame = self.screen_capture.get_frame()
            if frame is None:
                time.sleep(0.001)
                continue
            
            # Detect check marks
            check_marks = self.detector.detect_check_marks(frame)
            
            if check_marks:
                # Use the largest check mark (by area)
                largest_mark = max(check_marks, key=lambda x: x[4])
                center_x, center_y, width, height, area = largest_mark
                
                # Update position history
                self.detector.update_position((center_x, center_y))
                
                # Calculate motion vector
                motion_vector = self.detector.calculate_motion_vector()
                
                if motion_vector:
                    # Predict position in 0.5 seconds
                    predicted_x, predicted_y = motion_vector.predict_position(
                        (center_x, center_y), 0.1
                    )
                    
                    # Move mouse to predicted position
                    self._move_mouse_smooth(predicted_x, predicted_y)
                    
                    # Display info (optional - can be removed for maximum performance)
                    self._display_info(frame, center_x, center_y, width, height, 
                                     motion_vector, predicted_x, predicted_y)
                else:
                    # No motion vector yet, just move to current position
                    self._move_mouse_smooth(center_x, center_y)
                    self._display_info(frame, center_x, center_y, width, height)
            
            self.frame_count += 1
            
            # Maintain target FPS
            elapsed = time.time() - loop_start
            sleep_time = max(0, self.frame_delay - elapsed)
            time.sleep(sleep_time)
    
    def _move_mouse_smooth(self, x: float, y: float):
        """Move mouse to position with bounds checking"""
        # Get screen size
        screen_width, screen_height = pyautogui.size()
        
        # Ensure coordinates are within screen bounds
        x = max(0, min(x, screen_width - 1))
        y = max(0, min(y, screen_height - 1))
        
        # Move mouse
        pyautogui.moveTo(x, y, duration=0.1, _pause=False)
    
    def _display_info(self, frame: np.ndarray, x: float, y: float, width: float, 
                     height: float, motion_vector: Optional[MotionVector] = None,
                     pred_x: float = None, pred_y: float = None):
        """Display tracking information on frame (for debugging)"""
        # Draw bounding box around detected check mark
        cv2.rectangle(frame, 
                     (int(x - width/2), int(y - height/2)),
                     (int(x + width/2), int(y + height/2)),
                     (0, 255, 0), 2)
        
        # Draw center point
        cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)
        
        if motion_vector and pred_x is not None and pred_y is not None:
            # Draw motion vector
            end_x = int(x + motion_vector.dx * 0.1)  # Scale for visibility
            end_y = int(y + motion_vector.dy * 0.1)
            cv2.arrowedLine(frame, (int(x), int(y)), (end_x, end_y), 
                          (255, 0, 0), 2)
            
            # Draw predicted position
            cv2.circle(frame, (int(pred_x), int(pred_y)), 5, (255, 255, 0), 2)
            
            # Display information
            info_text = [
                f"Speed: {motion_vector.speed:.1f} px/s",
                f"Position: ({int(x)}, {int(y)})",
                f"Predicted: ({int(pred_x)}, {int(pred_y)})"
            ]
            
            for i, text in enumerate(info_text):
                cv2.putText(frame, text, (10, 30 + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display FPS
        fps = self.frame_count / (time.time() - self.start_time)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show frame (optional - comment out for maximum performance)
        cv2.imshow('Yellow Check Mark Tracker', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.stop_tracking()

def main():
    # Create and start tracker with target FPS
    tracker = CheckMarkTracker(target_fps=30)  # Adjust FPS as needed
    tracker.start_tracking()

if __name__ == "__main__":
    main()