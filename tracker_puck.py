import cv2
import numpy as np
import pyautogui
import time
from typing import Optional

from vector_motion import VectorMotion
from detector_puck import DetectorPuck
from screen_capture import ScreenCapture


class TrackerPuck:
    def __init__(self, target_fps: int = 30):
        self.screen_capture = ScreenCapture(target_fps)
        self.detector = DetectorPuck()
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
                     height: float, motion_vector: Optional[VectorMotion] = None,
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
