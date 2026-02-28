import random

from core import *

class ShortestPath(ProblemBase):
    def __init__(self, size):
        self.size = size
        self.adj = [[] for _ in range(size)]

        for i, v in enumerate(self.adj):
            cnt = len(v)
            add = random.randint(0 if cnt > 0 else 1, self.size - 1 - cnt)
            if add + cnt == size:
                for i_, _ in enumerate(self.adj):
                    v.append(i_)
                    self.adj[i_].append(i)  # undirected
            else:
                while add > 0:
                    new = random.randint(0, self.size - 1)
                    if new not in v:
                        v.append(new)
                        self.adj[new].append(i)  # undirected
                        add -= 1

    def name(self) -> str:
        return "Shortest Path"

    def random_solution_generate(self) -> Any:
        pass

    def evaluate(self, solution) -> float:
        return len(solution)

    def is_discrete(self) -> bool:
        return True

    def is_min_optimization(self) -> bool:
        return True

    def is_goal(self, solution) -> bool:
        pass

    def get_neighbors(self, solution) -> Any:
        pass

    def get_bounds(self) -> List[Tuple[float, float]]:
        return [(0, self.size - 1) for _ in self.size]
