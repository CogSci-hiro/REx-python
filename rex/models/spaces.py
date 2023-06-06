import numpy as np
from scipy.special import softmax
from sklearn.metrics.pairwise import cosine_similarity

from ..models.oscillators import Oscillator
from ..models.modulators import Modulator


class ExemplarSpace:
    def __init__(self, n_exemplars: int, n_dimensions: int, n_steps: int, temperature: float = 0.1):

        self.n_exemplars = n_exemplars
        self.n_dimensions = n_dimensions
        self.n_steps = n_steps
        self.temperature = temperature

        self.keys = np.zeros((n_exemplars, n_dimensions, n_steps))

        self.scores = np.zeros((n_exemplars,))

    def retrieve(self, query: np.array) -> np.array:

        scores = np.zeros((self.n_exemplars, self.n_steps))
        for i in range(self.n_exemplars):
            scores[i] = cosine_similarity(self.keys[i].T, query.reshape(1, -1)).squeeze()

        scores = np.max(scores, axis=1)  # (n_exemplars,)
        if self.temperature > 0.0:
            self.scores = softmax(scores / self.temperature)

        return scores


class ExemplarSpace2d:
    def __init__(self, n_axis_0: int, n_axis_1: int, n_dimensions: int, temperature: float = 0.1):

        self.n_axis_0 = n_axis_0
        self.n_axis_1 = n_axis_1
        self.n_dimensions = n_dimensions

        self.temperature = temperature

        self.keys = np.zeros((n_axis_0, n_axis_1, n_dimensions))

        self.scores = None

    def retrieve(self, query: np.array, axis: int, mask=None) -> np.array:

        if axis == 0:
            n_exemplars = self.n_axis_0
            n_axis = self.n_axis_1
        else:
            n_exemplars = self.n_axis_1
            n_axis = self.n_axis_0

        scores = np.zeros((n_exemplars, n_axis))
        for i in range(n_exemplars):

            if axis == 0:
                keys = self.keys[i, :, :]
            else:
                keys = self.keys[:, i, :]

            if mask is not None:
                keys *= mask

            scores[i] = cosine_similarity(keys, query.reshape(1, -1)).squeeze()

        scores = np.max(scores, axis=1)  # (n_exemplars,)
        if self.temperature > 0.0:
            self.scores = softmax(scores / self.temperature)

        return self.scores


########################################################################################################################


class FastSpace:
    def __init__(self, n_exemplars: int, n_dimensions: int, n_steps: int,
                 frequency: float = 12.0, lr: float = 0.01,
                 amplitude_threshold: float = 0.5, amplitude_sharpness: float = 40.0,
                 direction_threshold: float = 0.5, direction_sharpness: float = 20.0,
                 memory_capacity: int = 100, temperature: float = 0.1):

        self.n_exemplars = n_exemplars
        self.n_dimensions = n_dimensions
        self.n_steps = n_steps
        self.memory_capacity = memory_capacity
        self.lr = lr
        self.temperature = temperature

        self.exemplar_space = ExemplarSpace(n_exemplars, n_dimensions, n_steps, temperature)
        self.oscillator = Oscillator(frequency)
        self.modulator = Modulator(amplitude_threshold, amplitude_sharpness, direction_threshold, direction_sharpness)

        self.working_memory = []
        self.short_term_memory = []
        self.current_scores = np.zeros((n_exemplars,))
        self.average_scores = np.zeros((n_exemplars,))

    def update(self, query: np.array) -> bool:

        self.working_memory.append(query)

        # Calculate scores
        self.current_scores = self.exemplar_space.retrieve(query)

        # Update phase
        modulation = self.modulator.modulate(self.current_scores, self.average_scores)
        reset = self.oscillator.update(modulation)

        # Calculate running average
        n = len(self.working_memory)
        self.average_scores = (self.current_scores + self.average_scores * n) / (1 + n)

        return reset



########################################################################################################################


class SlowSpace:
    def __init__(self, n_exemplars: int, n_dimensions: int, n_steps: int, temperature: float = 0.1):

        self.exemplar_space = ExemplarSpace(n_exemplars=n_exemplars, n_dimensions=n_dimensions, n_steps=n_steps,
                                            temperature=temperature)
        pass
