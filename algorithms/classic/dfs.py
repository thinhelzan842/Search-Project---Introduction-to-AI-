from core import AlgorithmBase, DiscreteSearchProblem

class DFS(AlgorithmBase):
    def name(self) -> str: return "DFS"

    def run(self, problem):
        if not isinstance(problem, DiscreteSearchProblem):
            raise TypeError("DFS requires a DiscreteSearchProblem.")
            
        start = problem.get_initial_state()
        best_sol, best_score = start, float('inf')
        iteration = 0

        yield {
            'iteration': 0, 'current_solution': best_sol, 'current_score': best_score,
            'best_solution': best_sol, 'best_score': best_score,
        }

        stack = [start]
        def to_hash(state): return tuple(state) if isinstance(state, list) else state
        visited = set()

        while stack:
            state = stack.pop()
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

            for nb in reversed(problem.get_neighbors(state)):
                if to_hash(nb) not in visited:
                    stack.append(nb)

            yield {
                'iteration': iteration, 'current_solution': state, 'current_score': best_score,
                'best_solution': best_sol, 'best_score': best_score,
            }

"""from core.base import AlgorithmBase
from algorithms.classic._graph_utils import require_graph


class DFS(AlgorithmBase):

    def name(self) -> str:
        return "DFS"

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

        stack   = [(start, [start])]
        visited = set()

        while stack:
            node, path = stack.pop()
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
                return  # Returns the first complete path found

            # Reverse so the leftmost neighbour is explored first
            for nb in reversed(problem.adj[node]):
                if nb not in visited:
                    stack.append((nb, path + [nb]))

            yield {
                'iteration':        iteration,
                'current_solution': path,
                'current_score':    best_score,
                'best_solution':    best_sol,
                'best_score':       best_score,
            }
"""