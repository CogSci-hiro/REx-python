import numpy as np

from ..utils import sigmoid


class Modulator:
    def __init__(self, amplitude_threshold: float = 0.5, amplitude_sharpness: float = 40.0,
                 direction_threshold: float = 0.5, direction_sharpness: float = 20.0):

        self.amplitude_threshold = amplitude_threshold
        self.amplitude_sharpness = amplitude_sharpness
        self.direction_threshold = direction_threshold
        self.direction_sharpness = direction_sharpness


    def modulate(self, current_scores: np.array, average_scores: np.array) -> float:

        amplitude = sigmoid((np.max(average_scores) - self.amplitude_threshold) * self.amplitude_sharpness)

        # Multiply by 2 (1 - (-1)) and move down by 1 to get range (-1, 1)
        direction = -(sigmoid((current_scores.max() - self.direction_threshold) * self.direction_sharpness) * 2 - 1)

        return amplitude * direction
