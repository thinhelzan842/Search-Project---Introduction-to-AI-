import heapq
from core.base import AlgorithmBase
from algorithms.classic._graph_utils import require_graph, require_heuristic


class AStar(AlgorithmBase):
    """
    A* Search.

    Combines UCS (g = actual cost so far) with Greedy BFS (h = admissible
    heuristic estimate to goal) via f(n) = g(n) + h(n).

    Guarantees optimality when h is admissible (never overestimates true cost).
    Expands far fewer nodes than plain UCS on most practical graphs.

    Complexity: Time/Space O((V + E) log V), better in practice with a good h.

    Compatible with any problem that exposes:
        problem.start, problem.goal, problem.adj, problem.weights,
        problem.evaluate(), problem.heuristic(node) -> float
    """

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
