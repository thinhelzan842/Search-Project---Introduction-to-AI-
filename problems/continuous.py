import math
import random
import numpy as np
from typing import List, Tuple, Any
from core import *

class Ackley(ContinuousProblemBase):
    def __init__(self, dim, bound, a=20, b=0.2, c=2 * np.pi):
        self.dim, self.l_bound, self.r_bound = dim, -bound, bound
        self.a, self.b, self.c = a, b, c

    def random_solution_generate(self):
        return [random.uniform(self.l_bound, self.r_bound) for _ in range(self.dim)]

    def name(self) -> str: return "Ackley"

    def info(self) -> Any:
        return {"dim": self.dim, "left_bound": self.l_bound, "right_bound": self.r_bound, "a": self.a, "b": self.b, "c": self.c}

    def evaluate(self, solution) -> float:
        x = np.asarray(solution)
        sum_sq = np.sum(x**2)
        sum_cos = np.sum(np.cos(self.c * x))
        return -self.a * np.exp(-self.b * np.sqrt(sum_sq / self.dim)) - np.exp(sum_cos / self.dim) + self.a + np.e

    def is_discrete(self) -> bool: return False
    def is_min_optimization(self) -> bool: return True
    def is_goal(self, solution) -> bool: return self.evaluate(solution) <= 1e-6
    def get_bounds(self) -> List[Tuple[float, float]]: return [(self.l_bound, self.r_bound)] * self.dim


class Griewank(ContinuousProblemBase):
    def __init__(self, dim, bound):
        self.dim, self.l_bound, self.r_bound = dim, -bound, bound

    def random_solution_generate(self):
        return [random.uniform(self.l_bound, self.r_bound) for _ in range(self.dim)]

    def name(self) -> str: return "Griewank"
    def info(self) -> Any: return {"dim": self.dim, "left_bound": self.l_bound, "right_bound": self.r_bound}

    def evaluate(self, solution) -> float:
        x = np.asarray(solution)
        sum_term = np.sum(x**2) / 4000.0
        i = np.arange(1, self.dim + 1)
        prod_term = np.prod(np.cos(x / np.sqrt(i)))
        return sum_term - prod_term + 1.0

    def is_discrete(self) -> bool: return False
    def is_min_optimization(self) -> bool: return True
    def is_goal(self, solution) -> bool: return self.evaluate(solution) <= 1e-6
    def get_bounds(self) -> List[Tuple[float, float]]: return [(self.l_bound, self.r_bound)] * self.dim


class Rastrigin(ContinuousProblemBase):
    def __init__(self, dim, bound):
        self.dim, self.l_bound, self.r_bound = dim, -bound, bound

    def random_solution_generate(self):
        return [random.uniform(self.l_bound, self.r_bound) for _ in range(self.dim)]

    def name(self) -> str: return "Rastrigin"
    def info(self) -> Any: return {"dim": self.dim, "left_bound": self.l_bound, "right_bound": self.r_bound}

    def evaluate(self, solution) -> float:
        x = np.asarray(solution)
        return 10 * self.dim + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

    def is_discrete(self) -> bool: return False
    def is_min_optimization(self) -> bool: return True
    def is_goal(self, solution) -> bool: return self.evaluate(solution) <= 1e-6
    def get_bounds(self) -> List[Tuple[float, float]]: return [(self.l_bound, self.r_bound)] * self.dim


class Rosenbrock(ContinuousProblemBase):
    def __init__(self, dim, bound):
        self.dim, self.l_bound, self.r_bound = dim, -bound, bound

    def random_solution_generate(self):
        return [random.uniform(self.l_bound, self.r_bound) for _ in range(self.dim)]

    def name(self) -> str: return "Rosenbrock"
    def info(self) -> Any: return {"dim": self.dim, "left_bound": self.l_bound, "right_bound": self.r_bound}

    def evaluate(self, solution) -> float:
        x = np.asarray(solution)
        return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

    def is_discrete(self) -> bool: return False
    def is_min_optimization(self) -> bool: return True
    def is_goal(self, solution) -> bool: return self.evaluate(solution) <= 1e-6
    def get_bounds(self) -> List[Tuple[float, float]]: return [(self.l_bound, self.r_bound)] * self.dim


class Sphere(ContinuousProblemBase):
    def __init__(self, dim, bound):
        self.dim, self.l_bound, self.r_bound = dim, -bound, bound

    def random_solution_generate(self):
        return [random.uniform(self.l_bound, self.r_bound) for _ in range(self.dim)]

    def name(self) -> str: return "Sphere"
    def info(self) -> Any: return {"dim": self.dim, "left_bound": self.l_bound, "right_bound": self.r_bound}

    def evaluate(self, solution) -> float:
        return float(np.sum(np.square(np.asarray(solution))))

    def is_discrete(self) -> bool: return False
    def is_min_optimization(self) -> bool: return True
    def is_goal(self, solution) -> bool: return self.evaluate(solution) <= 1e-6
    def get_bounds(self) -> List[Tuple[float, float]]: return [(self.l_bound, self.r_bound)] * self.dim