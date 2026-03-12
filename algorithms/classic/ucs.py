import heapq
from core.base import AlgorithmBase
from algorithms.classic._graph_utils import require_graph


class UCS(AlgorithmBase):
    """
    Uniform Cost Search (UCS) — equivalent to Dijkstra's algorithm.

    Expands the node with the lowest accumulated path cost g(n).
    Guaranteed optimal for non-negative edge weights.

    Complexity: Time/Space O((V + E) log V).

    Compatible with any problem that exposes:
        problem.start, problem.goal, problem.adj, problem.weights, problem.evaluate()
    """

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
