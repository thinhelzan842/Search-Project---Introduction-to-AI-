import math
import random
import numpy as np
from core.base import AlgorithmBase

class SimulatedAnnealing(AlgorithmBase):
    def __init__(self, max_epochs=1000, initial_temp=100.0, cooling_rate=0.99, step_size=0.5, markov_chain_length=20, step_decay=0.99):
        self.max_epochs = max_epochs
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.step_size = step_size
        self.markov_chain_length = markov_chain_length
        self.step_decay = step_decay

    def name(self) -> str:
        return "Simulated Annealing"

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
        best_sol, best_score = list(current_sol), current_score
        is_min = problem.is_min_optimization()
        temp = self.initial_temp

        yield {
            'iteration': 0,
            'current_solution': list(current_sol),
            'current_score': current_score,
            'best_solution': list(best_sol),
            'best_score': best_score,
            'temperature': temp
        }

        for epoch in range(self.max_epochs):
            # Markov Chain 
            for _ in range(self.markov_chain_length):
                neighbor_sol = self._get_neighbor(current_sol, problem)
                neighbor_score = problem.evaluate(neighbor_sol)
                
                delta_e = neighbor_score - current_score if is_min else current_score - neighbor_score
                
                if delta_e < 0:
                    current_sol, current_score = neighbor_sol, neighbor_score
                    if (is_min and current_score < best_score) or (not is_min and current_score > best_score):
                        best_sol, best_score = list(current_sol), current_score
                else: 
                    if temp > 1e-8:
                        try:
                            if random.random() < math.exp(-delta_e / temp):
                                current_sol, current_score = neighbor_sol, neighbor_score
                        except OverflowError:
                            pass
            
            temp *= self.cooling_rate
            
            yield {
                'iteration': epoch + 1,
                'current_solution': list(current_sol),
                'current_score': current_score,
                'best_solution': list(best_sol),
                'best_score': best_score,
                'temperature': temp
            }

            # Adaptive Step Size for Cons 
            if not problem.is_discrete():
                self.step_size *= self.step_decay
                if problem.is_goal(best_sol): 
                    break

        return best_sol, best_score