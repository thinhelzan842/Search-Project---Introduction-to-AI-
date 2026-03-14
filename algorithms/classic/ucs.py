import heapq
from core import AlgorithmBase, DiscreteSearchProblem

class UCS(AlgorithmBase):
    def name(self) -> str: return "UCS"

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
            raise TypeError("UCS requires a DiscreteSearchProblem.")
            
        start = problem.get_initial_state()
        best_sol, best_score = start, float('inf')
        iteration, counter = 0, 0 

        frontier = [(problem.evaluate(start), counter, start)]
        visited_costs = {}

        yield {
            'iteration': 0, 'current_solution': best_sol, 'current_score': best_score,
            'best_solution': best_sol, 'best_score': best_score,
        }

        while frontier:
            cost, _, state = heapq.heappop(frontier)
            state_key = self._get_state_key(state, problem)
            
            if state_key in visited_costs and visited_costs[state_key] < cost: 
                continue
            visited_costs[state_key] = cost
            iteration += 1

            if problem.is_goal(state):
                yield {
                    'iteration': iteration, 'current_solution': state, 'current_score': cost,
                    'best_solution': state, 'best_score': cost,
                }
                return

            for nb in problem.get_neighbors(state):
                new_cost = problem.evaluate(nb)
                
                if new_cost == float('inf'):
                    continue
                    
                nb_key = self._get_state_key(nb, problem)
                if nb_key not in visited_costs or new_cost < visited_costs[nb_key]:
                    counter += 1
                    heapq.heappush(frontier, (new_cost, counter, nb))

            yield {
                'iteration': iteration, 'current_solution': state, 'current_score': cost,
                'best_solution': best_sol, 'best_score': best_score,
            }