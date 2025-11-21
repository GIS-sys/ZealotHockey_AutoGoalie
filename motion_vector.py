from dataclasses import dataclass


@dataclass
class MotionVector:
    dx: float
    dy: float
    speed: float
    
    def predict_position(self, current_pos: tuple[float, float], time_sec: float) -> tuple[float, float]:
        """Predict future position based on current motion vector"""
        x, y = current_pos
        future_x = x + self.dx * time_sec
        future_y = y + self.dy * time_sec
        return (future_x, future_y)
