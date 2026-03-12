from core.base import AlgorithmBase
from algorithms.classic._graph_utils import require_graph


class DFS(AlgorithmBase):
    """
    Depth-First Search (DFS).

    Explores as deep as possible before backtracking via a LIFO stack.
    NOT guaranteed to find the shortest or cheapest path.

    Complexity: Time O(V + E), Space O(V).

    Compatible with any problem that exposes:
        problem.start, problem.goal, problem.adj, problem.weights, problem.evaluate()
    """

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
