from collections import deque
from core import AlgorithmBase, DiscreteSearchProblem

class BFS(AlgorithmBase):
    def name(self) -> str: return "BFS"

    def run(self, problem):
        if not isinstance(problem, DiscreteSearchProblem):
            raise TypeError("BFS requires a DiscreteSearchProblem.")
            
        start = problem.get_initial_state()
        best_sol, best_score = start, float('inf')
        iteration = 0

        yield {
            'iteration': 0, 'current_solution': best_sol, 'current_score': best_score,
            'best_solution': best_sol, 'best_score': best_score,
        }

        frontier = deque([start])
        def to_hash(state): return tuple(state) if isinstance(state, list) else state
        visited = {to_hash(start)}

        while frontier:
            state = frontier.popleft()
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
                    visited.add(h_nb)
                    frontier.append(nb)

            yield {
                'iteration': iteration, 'current_solution': state, 'current_score': best_score,
                'best_solution': best_sol, 'best_score': best_score,
            }

"""from collections import deque
from core.base import AlgorithmBase
from algorithms.classic._graph_utils import require_graph


class BFS(AlgorithmBase):
    def name(self) -> str:
        return "BFS"

    def run(self, problem):
        require_graph(problem)
        start, goal = problem.start, problem.goal
        best_sol   = [start]
        best_score = float('inf')
        iteration  = 0

        yield {
            'iteration':        0,
            'current_solution': best_sol,
            'current_score':    best_score,
            'best_solution':    best_sol,
            'best_score':       best_score,
        }

        # Mark nodes visited on *enqueue* to prevent duplicate queue entries
        frontier = deque([(start, [start])])
        visited  = {start}

        while frontier:
            node, path = frontier.popleft()
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
                return  # BFS first reach = optimal hop-count path

            for nb in problem.adj[node]:
                if nb not in visited:
                    visited.add(nb)
                    frontier.append((nb, path + [nb]))

            yield {
                'iteration':        iteration,
                'current_solution': path,
                'current_score':    best_score,
                'best_solution':    best_sol,
                'best_score':       best_score,
            }
"""