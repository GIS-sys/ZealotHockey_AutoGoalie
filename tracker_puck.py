import cv2
import numpy as np
import pyautogui
import time
from typing import Optional

from vector_motion import VectorMotion
from detector_puck import DetectorPuck
from screen_capture import ScreenCapture

import tkinter as tk
from PIL import ImageTk, Image


DEBUG_OVERLAY = True
DEBUG_MOUSE = True


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
                        (center_x, center_y), 0.3
                    )
                    next_x, next_y = motion_vector.predict_position(
                        (center_x, center_y), 0.6
                    )

                    # Move mouse to predicted position
                    self._move_mouse_smooth(predicted_x, predicted_y)

                    # Display info (optional - can be removed for maximum performance)
                    self._display_info(predicted_x, predicted_y, width, height,
                                       motion_vector, next_x, next_y)
                else:
                    # No motion vector yet, just move to current position
                    self._move_mouse_smooth(center_x, center_y)
                    self._display_info(center_x, center_y, width, height)

            self.frame_count += 1

            # Maintain target FPS
            elapsed = time.time() - loop_start
            sleep_time = max(0, self.frame_delay - elapsed)
            time.sleep(sleep_time)

    def _move_mouse_smooth(self, x: float, y: float):
        """Move mouse to position with bounds checking"""
        if not DEBUG_MOUSE:
            return

        # Get screen size
        screen_width, screen_height = pyautogui.size()

        # Ensure coordinates are within screen bounds
        x = max(1, min(x, screen_width - 2))
        y = max(1, min(y, screen_height - 2))

        # Move mouse
        try:
            pyautogui.moveTo(x, y, duration=0.1, _pause=False)
        except pyautogui.FailSafeException:
            print(f"pyautogui.FailSafeException: {x} {y}")

    def _display_info(self, x: float, y: float, width: float, height: float,
                    motion_vector: Optional[VectorMotion] = None,
                    pred_x: float = None, pred_y: float = None):
        """Display tracking information as an overlay on the screen"""
        if not DEBUG_OVERLAY:
            return

        try:
            # Get screen size
            screen_width, screen_height = pyautogui.size()

            # Create a transparent overlay image
            overlay = np.zeros((screen_height, screen_width, 4), dtype=np.uint8)

            # Draw bounding box around detected check mark (green)
            # cv2.rectangle(overlay,
            #             (int(x - width/2), int(y - height/2)),
            #             (int(x + width/2), int(y + height/2)),
            #             (0, 255, 0, 255), 2)  # Green with full opacity

            # Draw center point (blue)
            cv2.circle(overlay, (int(x), int(y)), 50, (0, 0, 255, 255), 3)

            if motion_vector and pred_x is not None and pred_y is not None:
                # Draw motion vector (red arrow)
                end_x = int(x + motion_vector.dx * 0.2)  # Scale for visibility
                end_y = int(y + motion_vector.dy * 0.2)
                # cv2.arrowedLine(overlay, (int(x), int(y)), (end_x, end_y),
                #             (255, 0, 0, 255), 3)  # red with full opacity

                # Draw predicted position (cyan)
                cv2.circle(overlay, (int(pred_x), int(pred_y)), 8, (0, 255, 255, 255), 3)

                # Draw prediction line (yellow)
                # cv2.line(overlay, (int(x), int(y)), (int(pred_x), int(pred_y)),
                #         (255, 255, 0, 255), 2)

            # Create information panel in top-left corner
            panel_height = 120
            panel_width = 300
            cv2.rectangle(overlay, (10, 10), (panel_width, panel_height),
                        (0, 0, 0, 200), -1)  # Semi-transparent black background
            cv2.rectangle(overlay, (10, 10), (panel_width, panel_height),
                        (255, 255, 255, 255), 1)  # White border

            # Calculate FPS
            current_time = time.time()
            fps = self.frame_count / (current_time - self.start_time) if current_time > self.start_time else 0

            # Prepare information text
            info_lines = []
            info_lines.append(f"FPS: {fps:.1f}")
            info_lines.append(f"Position: ({int(x)}, {int(y)})")

            if motion_vector:
                info_lines.append(f"Speed: {motion_vector.speed:.1f} px/s")
                info_lines.append(f"Direction: ({motion_vector.dx:.1f}, {motion_vector.dy:.1f})")
                if pred_x is not None and pred_y is not None:
                    info_lines.append(f"Predicted: ({int(pred_x)}, {int(pred_y)})")

            # Draw text on overlay
            for i, line in enumerate(info_lines):
                y_position = 35 + i * 20
                cv2.putText(overlay, line, (20, y_position),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255, 255), 1)

            # Draw legend in bottom-left corner
            legend_y = screen_height - 100
            cv2.rectangle(overlay, (10, legend_y), (200, legend_y + 80),
                        (0, 0, 0, 200), -1)

            legend_items = [
                # ((0, 255, 0), "Green Box", "Detected Mark"),
                ((0, 0, 255), "Blue Circle", "Current Puck (+0.3s)"),
                # ((0, 0, 255), "Red Arrow", "Motion Vector"),
                ((0, 255, 255), "Cyan Circle", "Predicted (+0.6s)"),
                # ((255, 255, 0), "Yellow Line", "Prediction Path"),
            ]

            for i, (color, color_name, description) in enumerate(legend_items):
                y_pos = legend_y + 20 + i * 15
                # Draw color indicator
                cv2.circle(overlay, (20, y_pos - 5), 4, color + (255,), -1)
                # Draw text
                cv2.putText(overlay, f"{color_name}: {description}", (30, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255, 255), 1)

            # Convert to PIL Image and display as overlay
            pil_image = Image.fromarray(overlay, 'RGBA')

            # Create a simple overlay window using Tkinter
            if not hasattr(self, 'overlay_window'):
                self._create_overlay_window()
            self.overlay_window.lift()

            # Update the overlay window
            if hasattr(self, 'overlay_window') and self.overlay_window:
                try:
                    self.overlay_photo = ImageTk.PhotoImage(pil_image)
                    self.overlay_canvas.create_image(0, 0, image=self.overlay_photo, anchor='nw')
                    self.overlay_window.update()
                except Exception as e:
                    # Recreate window if it was closed
                    self._create_overlay_window()

        except Exception as e:
            print(f"Overlay error: {e}")

    def _create_overlay_window(self):
        """Create a transparent overlay window"""
        try:
            self.overlay_window = tk.Tk()
            self.overlay_window.attributes('-fullscreen', True)
            self.overlay_window.attributes('-topmost', True)
            self.overlay_window.attributes('-transparentcolor', 'black')
            self.overlay_window.configure(bg='black')
            self.overlay_window.overrideredirect(True)

            # Create canvas
            self.overlay_canvas = tk.Canvas(self.overlay_window,
                                        bg='black',
                                        highlightthickness=0)
            self.overlay_canvas.pack(fill='both', expand=True)

            # Add close button (invisible but clickable in top-right)
            close_btn = tk.Button(self.overlay_window, text="X",
                                command=self.stop_tracking,
                                bg='red', fg='white', font=('Arial', 12))
            self.overlay_canvas.create_window(50, 50, window=close_btn)

            print("Overlay window created. Press the red 'X' button to stop tracking.")

        except Exception as e:
            print(f"Failed to create overlay window: {e}")
            self.overlay_window = None
