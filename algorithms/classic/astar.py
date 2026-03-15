import heapq
from core import AlgorithmBase, DiscreteSearchProblem

class AStar(AlgorithmBase):
    def name(self) -> str: return "A*"

    def _get_state_key(self, state, problem):
        """Helper hỗ trợ Pruning thông minh cho Graph Search"""
        if isinstance(state, list) and hasattr(problem, 'name'):
            p_name = problem.name().lower()
            if "shortest path" in p_name:
                return state[-1] if state else ()
            if "traveling salesman" in p_name:
                return (state[-1], frozenset(state)) if state else ()
        return tuple(state) if isinstance(state, list) else state

    def run(self, problem):
        if not isinstance(problem, DiscreteSearchProblem):
            raise TypeError("A* requires a DiscreteSearchProblem with get_heuristic() implemented.")
            
        start = problem.get_initial_state()
        best_sol, best_score = start, float('inf')
        iteration, counter = 0, 0

        g_start = problem.evaluate(start)
        h_start = problem.get_heuristic(start)
        
        # heap entry: (f_cost, tiebreaker, g_cost, state)
        frontier = [(g_start + h_start, counter, g_start, start)]
        visited_costs = {}

        yield {
            'iteration': 0, 'current_solution': best_sol, 'current_score': best_score,
            'best_solution': best_sol, 'best_score': best_score,
        }

        while frontier:
            f, _, g, state = heapq.heappop(frontier)
            state_key = self._get_state_key(state, problem)
            
            # Pruning: Bỏ qua nếu đã tìm thấy đường rẻ hơn đến state này
            if state_key in visited_costs and visited_costs[state_key] < g: 
                continue
            visited_costs[state_key] = g
            iteration += 1

            if problem.is_goal(state):
                yield {
                    'iteration': iteration, 'current_solution': state, 'current_score': g,
                    'best_solution': state, 'best_score': g,
                }
                return 

            for nb in problem.get_neighbors(state):
                new_g = problem.evaluate(nb)
                
                # Bỏ qua các nhánh không hợp lệ
                if new_g == float('inf'): 
                    continue
                    
                nb_key = self._get_state_key(nb, problem)
                if nb_key not in visited_costs or new_g < visited_costs[nb_key]:
                    new_f = new_g + problem.get_heuristic(nb)
                    counter += 1
                    heapq.heappush(frontier, (new_f, counter, new_g, nb))

            yield {
                'iteration': iteration, 'current_solution': state, 'current_score': g,
                'best_solution': best_sol, 'best_score': best_score,
            }