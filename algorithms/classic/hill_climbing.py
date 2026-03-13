import random
import numpy as np
from core.base import AlgorithmBase

class HillClimbing(AlgorithmBase):
    def __init__(self, max_iters=1300, step_size=0.5, num_neighbors=15, step_decay=0.99):
        self.max_iters = max_iters
        self.step_size = step_size
        self.num_neighbors = num_neighbors
        self.step_decay = step_decay

    def name(self) -> str:
        return "Hill Climbing"

    def _get_neighbor(self, current, problem, current_step_size):
        neighbor = list(current)
        if not problem.is_discrete():
            bounds = problem.get_bounds()
            for i in range(len(neighbor)):
                val = neighbor[i] + random.gauss(0, current_step_size)
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
        
        current_step_size = self.step_size 

        yield {
            'iteration': 0,
            'current_solution': list(current_sol),
            'current_score': current_score,
            'best_solution': list(best_sol),
            'best_score': best_score
        }

        for i in range(self.max_iters):
            best_candidate_sol = None
            best_candidate_score = float('inf') if is_min else float('-inf')

            # Steepest Ascent
            for _ in range(self.num_neighbors):
                n_sol = self._get_neighbor(current_sol, problem, current_step_size)
                n_score = problem.evaluate(n_sol)
                if (is_min and n_score < best_candidate_score) or (not is_min and n_score > best_candidate_score):
                    best_candidate_score = n_score
                    best_candidate_sol = n_sol
            
            # Compare and Update 
            if best_candidate_sol is not None:
                if (is_min and best_candidate_score < current_score) or (not is_min and best_candidate_score > current_score):
                    current_sol, current_score = best_candidate_sol, best_candidate_score
                    
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

            if not problem.is_discrete():
                current_step_size *= self.step_decay
                if problem.is_goal(best_sol): 
                    break

        return best_sol, best_score