import numpy as np
import random
from typing import List, Tuple, Any

from core import *

class Rastrigin(ContinuousProblemBase):
    def __init__(self, dim, bound):
        self.dim = dim
        self.l_bound = -bound
        self.r_bound = bound

    def random_solution_generate(self):
        return [random.uniform(self.l_bound, self.r_bound) for _ in range(self.dim)]

    def name(self) -> str:
        return "Rastrigin"
    
    def info(self) -> Any:
        return {"dim"           :self.dim,
                "left_bound"    :self.l_bound,
                "right_bound"   :self.r_bound}

    def evaluate(self, solution) -> float: #list of [x, y, z,...] ~ dimensions
        solution = np.asarray(solution)
        return 10 * self.dim + np.sum(solution**2 - 10 * np.cos(2 * np.pi * solution))

    def is_discrete(self) -> bool:
        return False

    def is_min_optimization(self) -> bool:
        return True

    def is_goal(self, solution) -> bool:
        return self.evaluate(solution) <= 1e-6

    def get_neighbors(self) -> Any:
        raise NotImplementedError("Continuous problems don't support returning neighbors")

    def get_bounds(self) -> List[Tuple[float, float]]:
        return [(self.l_bound, self.r_bound) for _ in range(self.dim)]