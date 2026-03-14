import math
import random
from typing import List, Any
from core import DiscreteSearchProblem

class GraphColoring(DiscreteSearchProblem[List[int]]):
    def __init__(self, size: int, edge_probability: float=0.3):
        self.size = size
        self.adj = [[] for _ in range(self.size)]
        for i in range(self.size):
            for j in range(i + 1, self.size):
                if random.random() < edge_probability:
                    self.adj[i].append(j)
                    self.adj[j].append(i)

    def random_solution_generate(self) -> List[int]:
        res = []
        for i in range(self.size):
            neighbor_colors = {res[n] for n in self.adj[i] if n < len(res)}
            available_colors = [c for c in range(self.size) if c not in neighbor_colors]
            if available_colors:
                res.append(random.choice(available_colors))
            else:
                res.append(random.randint(0, self.size - 1))
        return res

    def get_initial_state(self) -> List[int]: return []
    def name(self) -> str: return "Graph Coloring"
    def info(self) -> Any: return {"size": self.size, "graph": self.adj}

    def evaluate(self, solution: List[int]) -> float:
        unique_clrs = set()
        for i, color in enumerate(solution):
            for neighbor in self.adj[i]:
                if neighbor < len(solution) and solution[i] == solution[neighbor]:
                    return float('inf')  # Conflict found
            unique_clrs.add(color)
        return float(len(unique_clrs))

    def is_goal(self, solution: List[int]) -> bool:
        return len(solution) == self.size and self.evaluate(solution) != float('inf')

    def get_neighbors(self, current_state: List[int]) -> List[List[int]]:
        if len(current_state) >= self.size: return []
        return [current_state + [color] for color in range(self.size)]

    def get_heuristic(self, state: List[int]) -> float:
        # NP-Hard to bound tightly. Safest admissible heuristic is 0.
        return 0.0


class Knapsack(DiscreteSearchProblem[List[int]]):
    def __init__(self, size: int, limit: int):
        self.size = size
        self.limit = limit
        self.profits = [random.randint(1, size) for _ in range(self.size)]
        self.weights = [random.randint(1, size) for _ in range(self.size)]

    def random_solution_generate(self) -> List[int]:
        res = [0] * self.size
        indices = list(range(self.size))
        random.shuffle(indices)
        current_weight = 0
        for i in indices:
            if current_weight + self.weights[i] <= self.limit:
                res[i] = 1
                current_weight += self.weights[i]
        return res

    def get_initial_state(self) -> List[int]: return []
    def name(self) -> str: return "Knapsack"
    def info(self) -> Any: return {"size": self.size, "limit": self.limit, "profits": self.profits, "weights": self.weights}

    def evaluate(self, solution: List[int]) -> float:
        res, weight = 0, 0
        for i, v in enumerate(solution):
            res += self.profits[i] * v
            weight += self.weights[i] * v
        # Return negative profit to enforce unified min-optimization
        return float('inf') if weight > self.limit else float(-res)

    def is_goal(self, solution: List[int]) -> bool:
        return len(solution) == self.size

    def get_neighbors(self, current_state: List[int]) -> List[List[int]]:
        if len(current_state) >= self.size: return []
        return [current_state + [1], current_state + [0]]

    def get_heuristic(self, state: List[int]) -> float:
        # Fractional Knapsack Relaxation
        current_weight = sum(self.weights[i] * state[i] for i in range(len(state)))
        if current_weight > self.limit: return float('inf')
        
        remaining_capacity = self.limit - current_weight
        unprocessed_idx = range(len(state), self.size)
        
        remaining_items = sorted(unprocessed_idx, key=lambda i: self.profits[i] / self.weights[i], reverse=True)
        
        bound_profit = 0.0
        for i in remaining_items:
            if self.weights[i] <= remaining_capacity:
                bound_profit += self.profits[i]
                remaining_capacity -= self.weights[i]
            else:
                bound_profit += self.profits[i] * (remaining_capacity / self.weights[i])
                break 
                
        return float(-bound_profit)


class ShortestPath(DiscreteSearchProblem[List[int]]):
    def __init__(self, size: int, edge_probability: float=0.4):
        self.size = size
        self.adj = [[] for _ in range(size)]
        
        # Spatial coordinates needed for Euclidean heuristic
        self.coords = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(size)]
        
        for i in range(self.size):
            for j in range(i + 1, self.size):
                if random.random() < edge_probability:
                    self.adj[i].append(j)
                    self.adj[j].append(i)
        for i in range(self.size - 1):
            if i + 1 not in self.adj[i]:
                self.adj[i].append(i + 1)
                self.adj[i + 1].append(i)

    def random_solution_generate(self) -> List[int]:
        path = [0]
        for _ in range(self.size):
            current = path[-1]
            if current == self.size - 1: break
            possible = [n for n in self.adj[current] if n not in path]
            if not possible: break
            path.append(random.choice(possible))
        return path

    def get_initial_state(self) -> List[int]: return [0]
    def name(self) -> str: return "Shortest Path"
    def info(self) -> Any: return {"size": self.size, "graph": self.adj}

    def evaluate(self, solution: List[int]) -> float:
        if not solution: return 0.0
        # Đảm bảo các node liền kề thực sự có cạnh nối với nhau
        for i in range(len(solution) - 1):
            if solution[i + 1] not in self.adj[solution[i]]:
                return float('inf') # Phạt nặng nếu là đường ảo
        return float(len(solution) - 1)

    def is_goal(self, solution: List[int]) -> bool:
        return solution[-1] == self.size - 1 if solution else False

    def get_neighbors(self, current_state: List[int]) -> List[List[int]]:
        if not current_state: return []
        node = current_state[-1]
        return [current_state + [v] for v in self.adj[node] if v not in current_state]

    def get_heuristic(self, state: List[int]) -> float:
        if not state: return 0.0
        current = state[-1]
        goal = self.size - 1
        if current == goal: return 0.0
        
        x1, y1 = self.coords[current]
        x2, y2 = self.coords[goal]
        # In an unweighted graph, distance is a proxy. 
        # Scaled down to remain admissible if physical distance > edge hops
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2) / 100.0


class TravelingSalesman(DiscreteSearchProblem[List[int]]):
    def __init__(self, size: int, time_limit: int, cost_limit: int):
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

    def random_solution_generate(self) -> List[int]:
        res = [0]
        cur_time = 0
        unvisited = set(range(1, self.size))

        while unvisited:
            current = res[-1]
            valid_next = []
            for nxt in unvisited:
                cost = self.costs[current][nxt]
                if cost != float('inf'):
                    arrival_time = cur_time + cost
                    start, end = self.times[nxt]
                    if arrival_time <= end:
                        valid_next.append((nxt, cost, max(arrival_time, start)))

            if not valid_next: break
            nxt, cost, new_time = random.choice(valid_next)
            res.append(nxt)
            cur_time = new_time
            unvisited.remove(nxt)

        return res

    def get_initial_state(self) -> List[int]: return [0]
    def name(self) -> str: return "Traveling Salesman"
    
    def info(self) -> Any:
        return {"size": self.size, "time_limit": self.time_limit, "cost_limit": self.cost_limit}

    def evaluate(self, solution: List[int]) -> float:
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
        return float(sum_cost)

    def is_goal(self, solution: List[int]) -> bool:
        return len(solution) == self.size

    def get_neighbors(self, current_state: List[int]) -> List[List[int]]:
        if not current_state: return []
        node = current_state[-1]
        res = []
        for i in range(self.size):
            if i not in current_state and self.costs[node][i] != float('inf'):
                res.append(current_state + [i])
        return res

    def get_heuristic(self, state: List[int]) -> float:
        if not state or len(state) == self.size: return 0.0
            
        unvisited = set(range(self.size)) - set(state)
        current_node = state[-1]
        h_cost = 0.0
        
        min_leave_current = min((self.costs[current_node][v] for v in unvisited if self.costs[current_node][v] != float('inf')), default=0)
        h_cost += min_leave_current
        
        for u in unvisited:
            possible_destinations = unvisited - {u}
            possible_destinations.add(state[0]) 
            min_edge = min((self.costs[u][v] for v in possible_destinations if self.costs[u][v] != float('inf')), default=0)
            h_cost += min_edge
            
        return float(h_cost)