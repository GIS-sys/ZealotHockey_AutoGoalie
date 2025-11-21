import pyautogui

from detector_puck import DetectorPuck


DEBUG_MOUSE = True


class Controller:
    def __init__(self, detector: DetectorPuck):
        self.detector = detector

    def do(self, x: float, y: float):
        motion_vector = self.detector.calculate_motion_vector()
        x, y = motion_vector.predict_position((x, y), 0.1)
        self._move_mouse_smooth(x, y)

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