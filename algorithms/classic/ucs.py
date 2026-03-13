import heapq
from core import AlgorithmBase, DiscreteSearchProblem

class UCS(AlgorithmBase):
    def name(self) -> str: return "UCS"

    def run(self, problem):
        if not isinstance(problem, DiscreteSearchProblem):
            raise TypeError("UCS requires a DiscreteSearchProblem.")
            
        start = problem.get_initial_state()
        best_sol, best_score = start, float('inf')
        iteration, counter = 0, 0 

        yield {
            'iteration': 0, 'current_solution': best_sol, 'current_score': best_score,
            'best_solution': best_sol, 'best_score': best_score,
        }

        # heap entry: (g_cost, tiebreaker, state)
        frontier = [(problem.evaluate(start), counter, start)]
        def to_hash(state): return tuple(state) if isinstance(state, list) else state
        visited = set()

        while frontier:
            cost, _, state = heapq.heappop(frontier)
            h_state = to_hash(state)
            
            if h_state in visited: continue
            visited.add(h_state)
            iteration += 1

            if problem.is_goal(state):
                yield {
                    'iteration': iteration, 'current_solution': state, 'current_score': cost,
                    'best_solution': state, 'best_score': cost,
                }
                return

            for nb in problem.get_neighbors(state):
                h_nb = to_hash(nb)
                if h_nb not in visited:
                    counter += 1
                    heapq.heappush(frontier, (problem.evaluate(nb), counter, nb))

            yield {
                'iteration': iteration, 'current_solution': state, 'current_score': cost,
                'best_solution': best_sol, 'best_score': best_score,
            }

"""import heapq
from core.base import AlgorithmBase
from algorithms.classic._graph_utils import require_graph


class UCS(AlgorithmBase):

    def name(self) -> str:
        return "UCS"

    def run(self, problem):
        require_graph(problem)
        start, goal = problem.start, problem.goal
        best_sol   = [start]
        best_score = float('inf')
        iteration  = 0
        counter    = 0  # tiebreaker for heap entries with equal cost

        yield {
            'iteration':        0,
            'current_solution': best_sol,
            'current_score':    best_score,
            'best_solution':    best_sol,
            'best_score':       best_score,
        }

        # heap entry: (g, counter, node, path)
        frontier = [(0, counter, start, [start])]
        visited  = set()

        while frontier:
            g, _, node, path = heapq.heappop(frontier)
            if node in visited:
                continue
            visited.add(node)
            iteration += 1

            if node == goal:
                best_sol, best_score = path, g
                yield {
                    'iteration':        iteration,
                    'current_solution': path,
                    'current_score':    g,
                    'best_solution':    best_sol,
                    'best_score':       best_score,
                }
                return  # First pop of goal = globally optimal cost

            for nb in problem.adj[node]:
                if nb not in visited:
                    new_g = g + problem.weights[node][nb]
                    counter += 1
                    heapq.heappush(frontier, (new_g, counter, nb, path + [nb]))

            yield {
                'iteration':        iteration,
                'current_solution': path,
                'current_score':    g,
                'best_solution':    best_sol,
                'best_score':       best_score,
            }
"""