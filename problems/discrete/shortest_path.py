import random
import math
from typing import List, Tuple, Any

from core import *


class ShortestPath(DiscreteProblemBase):
    """
    Weighted undirected graph — find the minimum-cost path from node 0 to
    node (size - 1).

    Supports two categories of algorithm:
    - Graph-search (BFS, DFS, UCS, Greedy BFS, A*) via
      .start / .goal / .adj / .weights / .heuristic(node)
    - Metaheuristic (Hill Climbing, SA, Tabu) via
      .random_solution_generate() / .evaluate()
    """

    def __init__(self, size: int):
        self.size  = size
        self.start = 0
        self.goal  = size - 1

        # 2-D positions for an admissible Euclidean heuristic
        self.positions = [(random.uniform(0, 100), random.uniform(0, 100))
                         for _ in range(size)]

        # Weight matrix — float('inf') means no direct edge
        self.weights = [[float('inf')] * size for _ in range(size)]
        for i in range(size):
            self.weights[i][i] = 0

        # Guarantee connectivity via a backbone chain 0 - 1 - … - (size-1)
        for i in range(size - 1):
            w = random.randint(1, 20)
            self.weights[i][i + 1] = w
            self.weights[i + 1][i] = w

        # Add extra random edges for variety
        extra   = random.randint(size, size * 2)
        added   = 0
        attempts = 0
        while added < extra and attempts < size * size:
            i = random.randint(0, size - 1)
            j = random.randint(0, size - 1)
            if i != j and self.weights[i][j] == float('inf'):
                w = random.randint(1, 20)
                self.weights[i][j] = w
                self.weights[j][i] = w
                added += 1
            attempts += 1

        # Adjacency list derived from weight matrix
        self.adj = [[] for _ in range(size)]
        for i in range(size):
            for j in range(size):
                if i != j and self.weights[i][j] != float('inf'):
                    self.adj[i].append(j)

    # ------------------------------------------------------------------
    # Heuristic for informed search (admissible Euclidean distance / 10)
    # ------------------------------------------------------------------
    def heuristic(self, node: int) -> float:
        x1, y1 = self.positions[node]
        x2, y2 = self.positions[self.goal]
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) / 10.0

    # ------------------------------------------------------------------
    # ProblemBase interface
    # ------------------------------------------------------------------
    def name(self) -> str:
        return "Shortest Path"

    def info(self) -> Any:
        return {"size": self.size, "start": self.start, "goal": self.goal}

    def evaluate(self, solution) -> float:
        """Sum of edge weights along solution path. Returns inf for invalid paths."""
        if not solution or solution[0] != self.start or solution[-1] != self.goal:
            return float('inf')
        total = 0.0
        for i in range(len(solution) - 1):
            w = self.weights[solution[i]][solution[i + 1]]
            if w == float('inf'):
                return float('inf')
            total += w
        return total

    def random_solution_generate(self) -> Any:
        """Random walk from start to goal; falls back to chain path."""
        for _ in range(15):
            path    = [self.start]
            visited = {self.start}
            node    = self.start
            for _ in range(self.size * 5):
                if node == self.goal:
                    return path
                choices = [v for v in self.adj[node] if v not in visited]
                if not choices:
                    break
                node = random.choice(choices)
                path.append(node)
                visited.add(node)
        return list(range(self.size))  # guaranteed valid fallback (chain)

    def get_initial_state(self) -> Any:
        return [self.start]

    def is_discrete(self) -> bool:
        return True

    def is_min_optimization(self) -> bool:
        return True

    def is_goal(self, solution) -> bool:
        return (len(solution) > 0
                and solution[-1] == self.goal
                and self.evaluate(solution) < float('inf'))

    def get_neighbors(self, path) -> Any:
        """Extend path by one unvisited neighbour (used by graph-search / tabu)."""
        node    = path[-1]
        visited = set(path)
        return [list(path) + [v] for v in self.adj[node] if v not in visited]

    def get_bounds(self) -> List[Tuple[float, float]]:
        return [(0, self.size - 1) for _ in range(self.size)]

