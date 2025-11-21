import cv2
import numpy as np
import pyautogui
import threading
import time
from typing import Optional


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
