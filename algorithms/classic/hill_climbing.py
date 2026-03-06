import random
import numpy as np
from core.base import AlgorithmBase

class HillClimbing(AlgorithmBase):
    def __init__(self, max_iters=1000, step_size=0.1):
        self.max_iters = max_iters
        self.step_size = step_size

    def name(self) -> str:
        return "Hill Climbing"

    def _get_neighbor(self, current, problem):
        neighbor = list(current)
        if not problem.is_discrete():
            bounds = problem.get_bounds()
            for i in range(len(neighbor)):
                val = neighbor[i] + random.gauss(0, self.step_size)
                neighbor[i] = max(bounds[i][0], min(bounds[i][1], val))
        else:
            if len(neighbor) > 1:
                idx1, idx2 = random.sample(range(len(neighbor)), 2)
                neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
        return neighbor

    def run(self, problem):
        current_sol = problem.random_solution_generate()
        current_score = problem.evaluate(current_sol)
        is_min = problem.is_min_optimization()
        
        best_sol, best_score = list(current_sol), current_score

        # Yield first judge  
        yield {
            'iteration': 0,
            'current_solution': list(current_sol),
            'current_score': current_score,
            'best_solution': list(best_sol),
            'best_score': best_score
        }

        for i in range(self.max_iters):
            neighbor_sol = self._get_neighbor(current_sol, problem)
            neighbor_score = problem.evaluate(neighbor_sol)
            
            # Compare and Update
            if (is_min and neighbor_score < current_score) or (not is_min and neighbor_score > current_score):
                current_sol, current_score = neighbor_sol, neighbor_score
                
                # Update Global Best
                if (is_min and current_score < best_score) or (not is_min and current_score > best_score):
                    best_sol, best_score = list(current_sol), current_score

            yield {
                'iteration': i + 1,
                'current_solution': list(current_sol),
                'current_score': current_score,
                'best_solution': list(best_sol),
                'best_score': best_score
            }

            if not problem.is_discrete() and problem.is_goal(best_sol): 
                break

        return best_sol, best_score