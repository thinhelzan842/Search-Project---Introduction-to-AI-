import random
from typing import List, Tuple, Any

from core import *

class TravelingSalesman(DiscreteProblemBase):
    def __init__(self, size, time_limit, cost_limit):
        self.size = size
        self.time_limit = time_limit
        self.cost_limit = cost_limit

        self.costs = [[float('inf') for _ in range(size)] for _ in range (size)]
        for i in range(size-1):
            for j in range(i+1, size):
                add = random.randint(0, 1)
                val = random.randint(1, size)
                if add == 1:
                    self.costs[i][j] = val
                    self.costs[j][i] = val #undirected

        self.times = [(0,0) for _ in range(size)] #time interval
        for i in range(size):
            start = random.randint(0, time_limit-1)
            end = random.randint(start+1, time_limit)
            self.times[i] = (start, end)

    def random_solution_generate(self):
        sol = list(range(self.size))
        random.shuffle(sol)
        return sol

    def name(self) -> str:
        return "Traveling Salesman"
    
    def info(self) -> Any:
        return {"size"          :self.size,
                "time_limit"    :self.time_limit,
                "cost_limit"    :self.cost_limit,
                "costs"         :self.costs,
                "times"         :self.times}

    def evaluate(self, solution) -> float: #city list filled
        sum_cost = 0
        cur_time = 0
        for i in range(solution):
            u = solution[i]
            v = solution[i+1]

            cost = self.costs[u][v]
            sum_cost += cost
            cur_time += cost #assume proportional to cost

            start, end = self.times[v]
            if cur_time > end:
                return float('inf')
            cur_time = max(start, cur_time)

        if sum_cost > self.cost_limit or cur_time > self.time_limit:
            return float('inf')

        return sum_cost


    def is_discrete(self) -> bool:
        return True

    def is_min_optimization(self) -> bool:
        return True

    def is_goal(self, solution) -> bool:
        pass

    def get_neighbors(self, progress) -> Any:
        node = progress[-1]
        res = []
        for i in range(self.size):
            if self.costs[node][i] != float('inf'):
                temp = progress
                temp.append(i)
                res.append(temp)
        return res

    def get_bounds(self) -> List[Tuple[float, float]]:
        return [(0, self.size - 1) for _ in range(self.size)]