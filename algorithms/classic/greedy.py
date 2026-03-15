import heapq
from core import AlgorithmBase, DiscreteSearchProblem

class GreedyBFS(AlgorithmBase):
    def name(self) -> str: return "Greedy BFS"

    def _get_state_key(self, state, problem):
        if isinstance(state, list) and hasattr(problem, 'name'):
            p_name = problem.name().lower()
            if "shortest path" in p_name:
                return state[-1] if state else ()
            if "traveling salesman" in p_name:
                return (state[-1], frozenset(state)) if state else ()
        return tuple(state) if isinstance(state, list) else state

    def run(self, problem):
        if not isinstance(problem, DiscreteSearchProblem):
            raise TypeError("Greedy BFS requires a DiscreteSearchProblem.")
            
        start = problem.get_initial_state()
        best_sol, best_score = start, float('inf')
        iteration, counter = 0, 0

        frontier = [(problem.get_heuristic(start), counter, start)]
        visited = set()

        yield {
            'iteration': 0, 'current_solution': best_sol, 'current_score': best_score,
            'best_solution': best_sol, 'best_score': best_score,
        }

        while frontier:
            _, _, state = heapq.heappop(frontier)
            state_key = self._get_state_key(state, problem)
            
            if state_key in visited: continue
            visited.add(state_key)
            iteration += 1

            if problem.is_goal(state):
                cost = problem.evaluate(state)
                yield {
                    'iteration': iteration, 'current_solution': state, 'current_score': cost,
                    'best_solution': state, 'best_score': cost,
                }
                return

            for nb in problem.get_neighbors(state):
                if problem.evaluate(nb) == float('inf'):
                    continue
                    
                nb_key = self._get_state_key(nb, problem)
                if nb_key not in visited:
                    counter += 1
                    heapq.heappush(frontier, (problem.get_heuristic(nb), counter, nb))

            yield {
                'iteration': iteration, 'current_solution': state, 'current_score': best_score,
                'best_solution': best_sol, 'best_score': best_score,
            }