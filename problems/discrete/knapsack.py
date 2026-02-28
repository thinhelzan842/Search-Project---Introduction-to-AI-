import random

from core import *

class Knapsack(ProblemBase):
    def __init__(self, size, limit):
        self.size = size
        self.limit = limit
        self.profits = [random.randint(1, size) for _ in range(self.size)]
        self.weights = [random.randint(1, size) for _ in range(self.size)]

    def random_solution_generate(self):
        return [random.randint(0,1) for _ in self.size]

    def name(self) -> str:
        return "Knapsack"

    def evaluate(self, solution) -> float: #solution is a 0/1 list
        res = 0
        weight = 0
        for i,v in enumerate(solution):
            res += self.profits[i] * v
            weight += self.weights[i] * v
        if weight > self.limit:
            return -1
        else:
            return res

    def is_discrete(self) -> bool:
        return True

    def is_min_optimization(self) -> bool:
        return False

    def is_goal(self, solution) -> bool:
        pass

    def get_neighbors(self) -> Any:
        pass

    def get_bounds(self) -> List[Tuple[float, float]]:
        return [(0, 1) for _ in range(self.size)]