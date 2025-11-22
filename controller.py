import keyboard
import math
import pyautogui
import pydirectinput
import time

from detector_puck import DetectorPuck


CREASE_CORNER_TOP = (200, 450)
CREASE_CORNER_BOT = (200, 700)


class Controller:
    class KEYBINDS:
        ENABLE_CONTROL = "f1"
        SET_CORNER_AUTO = "f2"
        SET_CORNER_TOP = "f3"
        SET_CORNER_BOT = "f4"

    def __init__(self, detector: DetectorPuck):
        print("Keybinds:")
        print(f"Press {Controller.KEYBINDS.ENABLE_CONTROL} to enable auto control")
        print(f"Press {Controller.KEYBINDS.SET_CORNER_TOP} to set top gate corner")
        print(f"Press {Controller.KEYBINDS.SET_CORNER_BOT} to set bot gate corner")
        self.detector_puck = detector
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
        if Controller.KEYBINDS.SET_CORNER_AUTO in self.pressed_keys:
            pass # TODO auto determine corners

    def do(self, pos: tuple[float, float] = None):
        if Controller.KEYBINDS.ENABLE_CONTROL not in self.pressed_keys:
            return

        motion_vector = self.detector_puck.calculate_motion_vector()
        if pos is None:
            a = time.time() % 2
            a = a if a < 1 else 2 - a
            inter_x = (CREASE_CORNER_TOP[0] * a + CREASE_CORNER_BOT[0] * (1 - a))
            inter_y = (CREASE_CORNER_TOP[1] * a + CREASE_CORNER_BOT[1] * (1 - a))
            self._move_mouse(int(inter_x), int(inter_y))
            return

        ix, iy, distance, on_segment = self.find_intersection((CREASE_CORNER_TOP, CREASE_CORNER_BOT), pos, (motion_vector.dx, motion_vector.dy))
        if on_segment:
            if distance < 2:
                self._move_mouse(ix, iy)
                #pydirectinput.move(int(ix), int(iy))
                pydirectinput.mouseDown(button=pyautogui.RIGHT)
                #time.sleep(0.01)
                pydirectinput.mouseUp(button=pyautogui.RIGHT)
                pydirectinput.keyDown("q")
                #time.sleep(0.01)
                pydirectinput.keyUp("q")
                pydirectinput.keyDown("s")
                #time.sleep(0.01)
                pydirectinput.keyUp("s")

    def _move_mouse(self, x: float, y: float):
        """Move mouse to position with bounds checking"""
        screen_width, screen_height = pyautogui.size()
        x = max(50, min(x, screen_width - 50))
        y = max(50, min(y, screen_height - 50))
        pydirectinput.moveTo(int(x), int(y))

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
