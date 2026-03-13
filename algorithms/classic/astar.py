import heapq
from core import AlgorithmBase, DiscreteSearchProblem

class AStar(AlgorithmBase):
    """
    A* Search.

    Combines the exact path cost so far g(n) with an admissible 
    heuristic estimate to the goal h(n) via f(n) = g(n) + h(n).
    Guarantees optimality when h is admissible (never overestimates).
    """

    def name(self) -> str: 
        return "A*"

    def run(self, problem):
        if not isinstance(problem, DiscreteSearchProblem):
            raise TypeError("A* requires a DiscreteSearchProblem with get_heuristic() implemented.")
            
        start = problem.get_initial_state()
        best_sol, best_score = start, float('inf')
        iteration, counter = 0, 0

        yield {
            'iteration': 0, 'current_solution': best_sol, 'current_score': best_score,
            'best_solution': best_sol, 'best_score': best_score,
        }

        # Calculate initial costs
        g_start = problem.evaluate(start)
        h_start = problem.get_heuristic(start)
        
        # heap entry: (f_cost, tiebreaker, g_cost, state)
        frontier = [(g_start + h_start, counter, g_start, start)]
        
        # Helper to hash list-based paths for the visited set
        def to_hash(state): 
            return tuple(state) if isinstance(state, list) else state
            
        visited = set()

        while frontier:
            f, _, g, state = heapq.heappop(frontier)
            h_state = to_hash(state)
            
            if h_state in visited: 
                continue
            visited.add(h_state)
            iteration += 1

            if problem.is_goal(state):
                yield {
                    'iteration': iteration, 'current_solution': state, 'current_score': g,
                    'best_solution': state, 'best_score': g,
                }
                return  # First pop of goal in A* is guaranteed optimal

            for nb in problem.get_neighbors(state):
                h_nb = to_hash(nb)
                if h_nb not in visited:
                    new_g = problem.evaluate(nb)
                    new_f = new_g + problem.get_heuristic(nb)
                    counter += 1
                    heapq.heappush(frontier, (new_f, counter, new_g, nb))

            yield {
                'iteration': iteration, 'current_solution': state, 'current_score': g,
                'best_solution': best_sol, 'best_score': best_score,
            }

"""import heapq
from core.base import AlgorithmBase
from algorithms.classic._graph_utils import require_graph, require_heuristic


class AStar(AlgorithmBase):

    def name(self) -> str:
        return "A*"

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

        # heap entry: (f, g, counter, node, path)
        frontier = [(h(start), 0, counter, start, [start])]
        visited  = set()

        while frontier:
            f, g, _, node, path = heapq.heappop(frontier)
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
                return  # Optimal when h is admissible

            for nb in problem.adj[node]:
                if nb not in visited:
                    new_g = g + problem.weights[node][nb]
                    new_f = new_g + h(nb)
                    counter += 1
                    heapq.heappush(frontier, (new_f, new_g, counter, nb, path + [nb]))

            yield {
                'iteration':        iteration,
                'current_solution': path,
                'current_score':    g,
                'best_solution':    best_sol,
                'best_score':       best_score,
            }
"""