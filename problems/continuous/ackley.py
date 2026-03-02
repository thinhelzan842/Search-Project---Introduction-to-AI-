import math
import random
from typing import List, Tuple, Any

from core import *

class Ackley(ContinuousProblemBase):
    def __init__(self, dim, bound, a=20, b=0.2, c=2*math.pi):
        self.dim = dim
        self.l_bound = -bound
        self.r_bound = bound
        self.a = a
        self.b = b
        self.c = c

    def random_solution_generate(self):
        return [random.uniform(self.l_bound, self.r_bound) for _ in range(self.dim)]

    def name(self) -> str:
        return "Ackley"
    
    def info(self) -> Any:
        return {"dim"           :self.dim,
                "left_bound"    :self.l_bound,
                "right_bound"   :self.r_bound,
                "a"             :self.a,
                "b"             :self.b,
                "c"             :self.c}

    def evaluate(self, solution) -> float: #list of [x, y, z,...] ~ dimensions
        return -self.a*math.exp(-self.b*math.sqrt((1.0/self.dim)*sum(x*x for x in solution))) - math.exp((1.0/self.dim)*sum(math.cos(self.c*x) for x in solution)) + self.a + math.exp(1)

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