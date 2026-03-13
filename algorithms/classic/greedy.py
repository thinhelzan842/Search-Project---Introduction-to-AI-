import heapq
from core import AlgorithmBase, DiscreteSearchProblem

class GreedyBFS(AlgorithmBase):
    def name(self) -> str: return "Greedy BFS"

    def run(self, problem):
        if not isinstance(problem, DiscreteSearchProblem):
            raise TypeError("Greedy BFS requires a DiscreteSearchProblem.")
            
        start = problem.get_initial_state()
        best_sol, best_score = start, float('inf')
        iteration, counter = 0, 0

        yield {
            'iteration': 0, 'current_solution': best_sol, 'current_score': best_score,
            'best_solution': best_sol, 'best_score': best_score,
        }

        # heap entry: (h_cost, tiebreaker, state)
        frontier = [(problem.get_heuristic(start), counter, start)]
        def to_hash(state): return tuple(state) if isinstance(state, list) else state
        visited = set()

        while frontier:
            _, _, state = heapq.heappop(frontier)
            h_state = to_hash(state)
            
            if h_state in visited: continue
            visited.add(h_state)
            iteration += 1

            if problem.is_goal(state):
                cost = problem.evaluate(state)
                yield {
                    'iteration': iteration, 'current_solution': state, 'current_score': cost,
                    'best_solution': state, 'best_score': cost,
                }
                return

            for nb in problem.get_neighbors(state):
                h_nb = to_hash(nb)
                if h_nb not in visited:
                    counter += 1
                    heapq.heappush(frontier, (problem.get_heuristic(nb), counter, nb))

            yield {
                'iteration': iteration, 'current_solution': state, 'current_score': best_score,
                'best_solution': best_sol, 'best_score': best_score,
            }

"""import heapq
from core.base import AlgorithmBase
from algorithms.classic._graph_utils import require_graph, require_heuristic


class GreedyBFS(AlgorithmBase):

    def name(self) -> str:
        return "Greedy BFS"

    def run(self, problem):
        require_graph(problem)
        require_heuristic(problem, self.name())
        start, goal = problem.start, problem.goal
        h          = problem.heuristic
        best_sol   = [start]
        best_score = float('inf')
        iteration  = 0
        counter    = 0

        yield {
            'iteration':        0,
            'current_solution': best_sol,
            'current_score':    best_score,
            'best_solution':    best_sol,
            'best_score':       best_score,
        }

        # heap entry: (h(n), counter, node, path)
        frontier = [(h(start), counter, start, [start])]
        visited  = set()

        while frontier:
            _, _, node, path = heapq.heappop(frontier)
            if node in visited:
                continue
            visited.add(node)
            iteration += 1

            if node == goal:
                cost = problem.evaluate(path)
                best_sol, best_score = path, cost
                yield {
                    'iteration':        iteration,
                    'current_solution': path,
                    'current_score':    cost,
                    'best_solution':    best_sol,
                    'best_score':       best_score,
                }
                return

            for nb in problem.adj[node]:
                if nb not in visited:
                    counter += 1
                    heapq.heappush(frontier, (h(nb), counter, nb, path + [nb]))

            yield {
                'iteration':        iteration,
                'current_solution': path,
                'current_score':    best_score,
                'best_solution':    best_sol,
                'best_score':       best_score,
            }
"""