import keyboard
import math
import pyautogui
import time

from detector_puck import DetectorPuck


DEBUG_MOUSE = True
CREASE_CORNER_TOP = (200, 450)
CREASE_CORNER_BOT = (200, 700)


class Controller:
    class KEYBINDS:
        ENABLE_CONTROL = "f1"
        SET_CORNER_TOP = "f2"
        SET_CORNER_BOT = "f3"

    def __init__(self, detector: DetectorPuck):
        print("Keybinds:")
        print(f"Press {Controller.KEYBINDS.ENABLE_CONTROL} to enable auto control")
        print(f"Press {Controller.KEYBINDS.SET_CORNER_TOP} to set top gate corner")
        print(f"Press {Controller.KEYBINDS.SET_CORNER_BOT} to set bot gate corner")
        self.detector = detector
        self.pressed_keys = set()
        def update_keys(e):
            if e.event_type == keyboard.KEY_DOWN:
                self.pressed_keys.add(e.name)
            elif e.event_type == keyboard.KEY_UP:
                self.pressed_keys.discard(e.name)
        keyboard.hook(update_keys)

    def tick(self):
        global CREASE_CORNER_TOP, CREASE_CORNER_BOT
        if Controller.KEYBINDS.SET_CORNER_TOP in self.pressed_keys:
            CREASE_CORNER_TOP = (pyautogui.position()[0], pyautogui.position()[1])
        if Controller.KEYBINDS.SET_CORNER_BOT in self.pressed_keys:
            CREASE_CORNER_BOT = (pyautogui.position()[0], pyautogui.position()[1])

    def do(self, pos: tuple[float, float] = None):
        if Controller.KEYBINDS.ENABLE_CONTROL not in self.pressed_keys:
            return

        motion_vector = self.detector.calculate_motion_vector()
        if pos is None:
            self._move_mouse_smooth((CREASE_CORNER_TOP[0] + CREASE_CORNER_BOT[0]) // 2, (CREASE_CORNER_TOP[1] + CREASE_CORNER_BOT[1]) // 2)
            return



        ix, iy, distance, on_segment = self.find_intersection((CREASE_CORNER_TOP, CREASE_CORNER_BOT), pos, (motion_vector.dx, motion_vector.dy))
        self._move_mouse_smooth(ix, iy)
        if on_segment and distance < 1:
            pyautogui.rightClick()

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

    @staticmethod
    def find_intersection(line: tuple[tuple[float, float], tuple[float, float]], point: tuple[float, float], speed: tuple[float, float]) -> tuple[float, float, float, bool]:
        (x1, y1), (x2, y2) = line
        x, y = point
        vx, vy = speed

        dx = x2 - x1
        dy = y2 - y1

        det = -dx * vy + dy * vx
        if abs(det) < 1e-10:
            return 0.0, 0.0, 0.0, False

        t = (dx * (y - y1) - dy * (x - x1)) / det
        u = (-vy * (x - x1) + vx * (y - y1)) / det

        ix = x + t * vx
        iy = y + t * vy

        speed_mag = math.sqrt(vx*vx + vy*vy)
        distance = t if t >= 0 else 0.0

        on_segment = (0 <= u <= 1) and (t >= 0)

        return ix, iy, distance, on_segment