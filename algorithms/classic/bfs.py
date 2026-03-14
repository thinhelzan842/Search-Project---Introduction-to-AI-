from collections import deque
from core import AlgorithmBase, DiscreteSearchProblem

class BFS(AlgorithmBase):
    def name(self) -> str: return "BFS"

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
            raise TypeError("BFS requires a DiscreteSearchProblem.")
            
        start = problem.get_initial_state()
        best_sol, best_score = start, float('inf')
        iteration = 0

        frontier = deque([start])
        visited = {self._get_state_key(start, problem)}

        yield {
            'iteration': 0, 'current_solution': best_sol, 'current_score': best_score,
            'best_solution': best_sol, 'best_score': best_score,
        }

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
                if problem.evaluate(nb) == float('inf'):
                    continue
                    
                nb_key = self._get_state_key(nb, problem)
                if nb_key not in visited:
                    visited.add(nb_key)
                    frontier.append(nb)

            yield {
                'iteration': iteration, 'current_solution': state, 'current_score': best_score,
                'best_solution': best_sol, 'best_score': best_score,
            }