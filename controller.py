import keyboard
import pyautogui
import time

from detector_puck import DetectorPuck


DEBUG_MOUSE = True
CREASE_CORNER_TOP = (200, 450)
CREASE_CORNER_BOTTOM = (200, 700)


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
        global CREASE_CORNER_TOP, CREASE_CORNER_BOTTOM
        if Controller.KEYBINDS.SET_CORNER_TOP in self.pressed_keys:
            CREASE_CORNER_TOP = (pyautogui.position()[0], pyautogui.position()[1])
        if Controller.KEYBINDS.SET_CORNER_BOT in self.pressed_keys:
            CREASE_CORNER_BOTTOM = (pyautogui.position()[0], pyautogui.position()[1])

    def do(self, pos: tuple[float, float] = None):
        if Controller.KEYBINDS.ENABLE_CONTROL not in self.pressed_keys:
            return

        motion_vector = self.detector.calculate_motion_vector()
        if pos is None:
            self._move_mouse_smooth((CREASE_CORNER_TOP[0] + CREASE_CORNER_BOTTOM[0]) // 2, (CREASE_CORNER_TOP[1] + CREASE_CORNER_BOTTOM[1]) // 2)
            return

        new_x, new_y = motion_vector.predict_position(pos, 0.1)
        self._move_mouse_smooth(new_x, new_y)
        #pyautogui.rightClick()

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