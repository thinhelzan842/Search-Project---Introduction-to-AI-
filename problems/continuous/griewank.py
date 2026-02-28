import math

from core import *

class Griewank(ProblemBase):
    def __init__(self, dim, bound):
        self.dim = dim
        self.l_bound = -bound
        self.r_bound = bound

    def random_solution_generate(self):
        return [random.uniform(self.l_bound, self.r_bound) for _ in range(self.dim)]

    def name(self) -> str:
        return "Griewank"

    def evaluate(self, solution) -> float: #list of [x, y, z,...] ~ dimensions
        return sum((x*x/4000.0) for x in solution) - math.prod((math.cos(x/math.sqrt(i)) + 1) for i,x in solution)

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