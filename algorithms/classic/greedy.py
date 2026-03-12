import heapq
from core.base import AlgorithmBase
from algorithms.classic._graph_utils import require_graph, require_heuristic


class GreedyBFS(AlgorithmBase):
    """
    Greedy Best-First Search.

    Expands the node that looks *closest to the goal* according to heuristic h(n).
    Fast in practice, but NOT guaranteed optimal — it may follow a misleading
    heuristic path and miss cheaper routes.

    Complexity: Time/Space O((V + E) log V) in the worst case.

    Compatible with any problem that exposes:
        problem.start, problem.goal, problem.adj, problem.weights,
        problem.evaluate(), problem.heuristic(node) -> float
    """

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
