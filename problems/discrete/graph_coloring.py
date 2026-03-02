import random
from typing import List, Tuple, Any

from core import *

class GraphColoring(DiscreteProblemBase):
    def __init__(self, size):
        self.size = size
        self.adj = [[] for _ in range(self.size)]

        for i, v in enumerate(self.adj):
            cnt = len(v)
            add = random.randint(0 if cnt > 0 else 1, self.size - 1 - cnt)
            if add + cnt == self.size:
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

    def random_solution_generate(self): #assist genetic algorithm
        sol = [random.randint(0, self.size-1) for _ in range(self.size)]
        """
        while self.evaluate(sol) == -1:
            for v in sol:
                clr = random.randint(0, self.size-1)
                v = clr
        """
        return sol

    def name(self) -> str:
        return "Graph Coloring"
    
    def info(self) -> Any:
        return {"size"  :self.size,
                "graph" :self.adj}

    def evaluate(self, solution) -> float: #solution is color list
        unique_clrs = []
        for i,v in enumerate(self.adj):
            for v_ in v:
                if solution[i] == solution[v_]:
                    return float('inf')
                else:
                    if solution[i] not in unique_clrs:
                        unique_clrs.append(solution[i])
        return len(unique_clrs)

    def is_discrete(self) -> bool:
        return True

    def is_min_optimization(self) -> bool:
        return True

    def is_goal(self, solution) -> bool:
        pass

    def get_neighbors(self) -> Any:
        raise NotImplementedError("Graph coloring doesn't support returning neighbors-cannot use BFS, DFS, ...")

    def get_bounds(self) -> List[Tuple[float, float]]:
        return [(0, self.size - 1) for _ in self.size]