import random
from typing import List, Tuple, Any
from core import *


class GraphColoring(DiscreteProblemBase):
    def __init__(self, size, edge_probability=0.3):
        self.size = size
        self.adj = [[] for _ in range(self.size)]
        # Simplified Erdős-Rényi graph generation
        for i in range(self.size):
            for j in range(i + 1, self.size):
                if random.random() < edge_probability:
                    self.adj[i].append(j)
                    self.adj[j].append(i)

    def random_solution_generate(self):
        # Constructive random generation to avoid infinite loops
        res = []
        for i in range(self.size):
            # Try to pick a color that doesn't conflict with neighbors
            neighbor_colors = {res[n] for n in self.adj[i] if n < len(res)}
            available_colors = [c for c in range(self.size) if c not in neighbor_colors]
            if available_colors:
                res.append(random.choice(available_colors))
            else:
                res.append(random.randint(0, self.size - 1))  # Fallback
        return res

    def get_initial_state(self):
        return []

    def name(self) -> str:
        return "Graph Coloring"

    def info(self) -> Any:
        return {"size": self.size, "graph": self.adj}

    def evaluate(self, solution) -> float:
        # Evaluate partial paths without returning float('inf') just for being incomplete
        unique_clrs = set()
        for i, color in enumerate(solution):
            for neighbor in self.adj[i]:
                if neighbor < len(solution) and solution[i] == solution[neighbor]:
                    return float('inf')  # Conflict found
            unique_clrs.add(color)
        return len(unique_clrs)
    
    def get_heuristic(self, solution): pass

    def is_discrete(self) -> bool:
        return True

    def is_min_optimization(self) -> bool:
        return True

    def is_goal(self, solution) -> bool:
        return len(solution) == self.size and self.evaluate(solution) != float('inf')

    def get_neighbors(self, progress) -> Any:
        if len(progress) >= self.size: return []
        res = []
        for color in range(self.size):
            temp = progress.copy()
            temp.append(color)
            res.append(temp)
        return res


class Knapsack(DiscreteProblemBase):
    def __init__(self, size, limit):
        self.size = size
        self.limit = limit
        self.profits = [random.randint(1, size) for _ in range(self.size)]
        self.weights = [random.randint(1, size) for _ in range(self.size)]

    def random_solution_generate(self):
        # Constructive logic: add items randomly until weight is hit
        res = [0] * self.size
        indices = list(range(self.size))
        random.shuffle(indices)
        current_weight = 0
        for i in indices:
            if current_weight + self.weights[i] <= self.limit:
                res[i] = 1
                current_weight += self.weights[i]
        return res

    def get_initial_state(self):
        return []

    def name(self) -> str:
        return "Knapsack"

    def info(self) -> Any:
        return {"size": self.size, "limit": self.limit, "profits": self.profits, "weights": self.weights}

    def evaluate(self, solution) -> float:
        # Safely evaluates partial solutions to guide search algorithms
        res, weight = 0, 0
        for i, v in enumerate(solution):
            res += self.profits[i] * v
            weight += self.weights[i] * v
        return -1 if weight > self.limit else res

    def is_discrete(self) -> bool:
        return True

    def is_min_optimization(self) -> bool:
        return False

    def is_goal(self, solution) -> bool:
        return len(solution) == self.size

    def get_neighbors(self, progress) -> Any:
        if len(progress) >= self.size: return []
        take = progress.copy()
        not_take = progress.copy()
        take.append(1)
        not_take.append(0)
        return [take, not_take]


class ShortestPath(DiscreteProblemBase):
    def __init__(self, size, edge_probability=0.4):
        self.size = size
        self.adj = [[] for _ in range(size)]
        # Simplified Erdős-Rényi graph generation
        for i in range(self.size):
            for j in range(i + 1, self.size):
                if random.random() < edge_probability:
                    self.adj[i].append(j)
                    self.adj[j].append(i)
        # Guarantee at least one valid path from start to end
        for i in range(self.size - 1):
            if i + 1 not in self.adj[i]:
                self.adj[i].append(i + 1)
                self.adj[i + 1].append(i)

    def random_solution_generate(self):
        path = [0]
        max_steps = self.size
        for _ in range(max_steps):
            current = path[-1]
            if current == self.size - 1: break
            neighbors = self.adj[current]
            possible = [n for n in neighbors if n not in path]
            if not possible: break
            path.append(random.choice(possible))
        return path

    def get_initial_state(self):
        return [0]

    def name(self) -> str:
        return "Shortest Path"

    def info(self) -> Any:
        return {"size": self.size, "graph": self.adj}

    def evaluate(self, solution) -> float:
        return len(solution)

    def is_discrete(self) -> bool:
        return True

    def is_min_optimization(self) -> bool:
        return True

    def is_goal(self, solution) -> bool:
        return solution[-1] == self.size - 1 if solution else False

    def get_neighbors(self, progress) -> Any:
        if not progress: return []
        node = progress[-1]
        res = []
        for v in self.adj[node]:
            if v not in progress:
                temp = progress.copy()
                temp.append(v)
                res.append(temp)
        return res


class TravelingSalesman(DiscreteProblemBase):
    def __init__(self, size, time_limit, cost_limit):
        self.size, self.time_limit, self.cost_limit = size, time_limit, cost_limit
        self.costs = [[float('inf') for _ in range(size)] for _ in range(size)]

        for i in range(size):
            j = (i + 1) % size
            val = random.randint(1, size // 2)
            self.costs[i][j] = self.costs[j][i] = val

        for i in range(size):
            for j in range(i + 1, size):
                if self.costs[i][j] == float('inf') and random.random() < 0.3:
                    val = random.randint(1, size)
                    self.costs[i][j] = self.costs[j][i] = val

        self.times = [(0, time_limit) for _ in range(size)]
        for i in range(1, size):
            earliest_possible = i * (size // 4)
            start = random.randint(0, min(earliest_possible, time_limit - 10))
            end = random.randint(start + 20, time_limit)
            self.times[i] = (start, end)

    def random_solution_generate(self):
        # Constructive heuristic generation instead of blind shuffling
        res = [0]  # Always start at 0
        cur_time = 0
        unvisited = set(range(1, self.size))

        while unvisited:
            current = res[-1]
            valid_next = []

            # Check for valid next steps based on cost/time constraints
            for nxt in unvisited:
                cost = self.costs[current][nxt]
                if cost != float('inf'):
                    arrival_time = cur_time + cost
                    start, end = self.times[nxt]
                    if arrival_time <= end:
                        valid_next.append((nxt, cost, max(arrival_time, start)))

            if not valid_next:
                break  # Dead end, return the partial path

            # Pick a random valid next city
            nxt, cost, new_time = random.choice(valid_next)
            res.append(nxt)
            cur_time = new_time
            unvisited.remove(nxt)

        return res

    def get_initial_state(self):
        return [0]

    def name(self) -> str:
        return "Traveling Salesman"

    def info(self) -> Any:
        return {"size": self.size, "time_limit": self.time_limit, "cost_limit": self.cost_limit, "costs": self.costs,
                "times": self.times}

    def evaluate(self, solution) -> float:
        if not solution: return 0.0
        sum_cost, cur_time = 0, 0
        for i in range(len(solution) - 1):
            u, v = solution[i], solution[i + 1]
            cost = self.costs[u][v]
            if cost == float('inf'): return float('inf')

            sum_cost += cost
            cur_time += cost
            start, end = self.times[v]
            if cur_time > end: return float('inf')
            cur_time = max(start, cur_time)

        if sum_cost > self.cost_limit or cur_time > self.time_limit:
            return float('inf')
        return sum_cost

    def is_discrete(self) -> bool:
        return True

    def is_min_optimization(self) -> bool:
        return True

    def is_goal(self, solution) -> bool:
        return len(solution) == self.size

    def get_neighbors(self, progress) -> Any:
        if not progress: return []
        node = progress[-1]
        res = []
        for i in range(self.size):
            if i not in progress and self.costs[node][i] != float('inf'):
                temp = progress.copy()
                temp.append(i)
                res.append(temp)
        return res