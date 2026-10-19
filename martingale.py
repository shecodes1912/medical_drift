import numpy as np
from typing import List, Dict, Union

class ConformalMartingaleMonitor:
    def __init__(self, delta: float = 0.05, epsilon: float = 0.5):
        self.delta = delta
        self.epsilon = epsilon
        self.threshold = 1.0 / delta
        self.S_cal: np.ndarray = np.array([])
        self.M_t: float = 1.0
        self.history: List[Dict[str, float]] = []

    def fit_calibration(self, calibration_scores: Union[List[float], np.ndarray]):
        """Phase 1: Fit and store calibration reference S_cal once (offline)."""
        self.S_cal = np.sort(np.array(calibration_scores))
        self.reset()

    def reset(self):
        """Reset online martingale state."""
        self.M_t = 1.0
        self.history = []

    def update(self, test_score: float) -> Dict[str, Union[float, bool]]:
        """Phase 2: Perform scalar martingale update per test image (online)."""
        if len(self.S_cal) == 0:
            raise RuntimeError("Monitor must be calibrated with fit_calibration() first.")

        n = len(self.S_cal)
        # Compute empirical conformal p-value (higher score = higher nonconformity)
        p_t = (1.0 + np.sum(self.S_cal >= test_score)) / (n + 1.0)
        
        # Power martingale betting factor update
        betting_factor = self.epsilon * (p_t ** (self.epsilon - 1.0))
        self.M_t *= betting_factor

        alarm_triggered = bool(self.M_t >= self.threshold)
        
        step_result = {
            "p_value": float(p_t),
            "test_score": float(test_score),
            "martingale_value": float(self.M_t),
            "alarm": alarm_triggered
        }
        self.history.append(step_result)
        return step_result