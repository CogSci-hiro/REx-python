from math import pi


class Oscillator:
    def __init__(self, frequency: float = 11.0):

        self.frequency = frequency                      # in Hz
        self.phase_delta = 2 * pi * frequency / 1000    # in ms

        self.phase = 0.0

    def update(self, modulation: float) -> bool:

        self.phase += self.phase_delta * (1 + modulation)

        if self.phase >= 2 * pi:
            self._reset()
            return True

        return False

    def _reset(self):
        self.phase = 0.0
