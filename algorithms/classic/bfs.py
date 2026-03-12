from collections import deque
from core.base import AlgorithmBase
from algorithms.classic._graph_utils import require_graph


class BFS(AlgorithmBase):
    """
    Breadth-First Search (BFS).

    Explores nodes level by level using a FIFO queue.
    Guarantees the path with the fewest hops (edges), NOT minimum cost.

    Complexity: Time O(V + E), Space O(V).

    Compatible with any problem that exposes:
        problem.start, problem.goal, problem.adj, problem.weights, problem.evaluate()
    """

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
