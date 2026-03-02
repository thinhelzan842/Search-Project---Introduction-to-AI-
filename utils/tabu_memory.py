from collections import deque
import numpy as np

class TabuList:
    def __init__(self, max_size: int, tolerance: int = 1):
        self.memory = deque(maxlen=max_size)
        self.tolerance = tolerance 

    def _hash_state(self, state: np.ndarray) -> tuple:
        return tuple(np.round(state, self.tolerance))

    def add(self, state: np.ndarray):
        self.memory.append(self._hash_state(state))

    def is_tabu(self, state: np.ndarray) -> bool:
        return self._hash_state(state) in self.memory