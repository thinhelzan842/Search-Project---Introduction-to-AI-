import random
import numpy as np
from core import AlgorithmBase, ContinuousProblem

class GravitationalSearchAlgorithm(AlgorithmBase):
    def __init__(self, pop_size=30, max_iters=500, G0=100.0, alpha=20.0):
        self.pop_size, self.max_iters = pop_size, max_iters
        self.G0, self.alpha = G0, alpha

    def name(self) -> str: return "Gravitational Search Algorithm"

    def run(self, problem):
        if not isinstance(problem, ContinuousProblem):
            raise TypeError("GSA requires a ContinuousProblem.")
            
        bounds = problem.get_bounds()
        dim = len(bounds)
        l_bound, r_bound = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])

        X = np.random.uniform(l_bound, r_bound, (self.pop_size, dim))
        V = np.zeros((self.pop_size, dim))
        best_sol, best_score = None, float('inf')

        for iteration in range(self.max_iters):
            scores = np.array([problem.evaluate(ind) for ind in X])
            
            curr_best_idx, curr_worst_idx = np.argmin(scores), np.argmax(scores)
            if scores[curr_best_idx] < best_score:
                best_score, best_sol = scores[curr_best_idx], X[curr_best_idx].copy()

            best_val, worst_val = scores[curr_best_idx], scores[curr_worst_idx]
            
            if best_val == worst_val:
                M = np.ones(self.pop_size)
            else:
                m = (scores - worst_val) / (best_val - worst_val)
                M = m / np.sum(m) 

            G = self.G0 * np.exp(-self.alpha * (iteration / self.max_iters))
            A = np.zeros((self.pop_size, dim))
            
            for i in range(self.pop_size):
                force = np.zeros(dim)
                for j in range(self.pop_size):
                    if i != j:
                        R = np.linalg.norm(X[i] - X[j]) + 1e-8 
                        force += random.random() * M[j] * ((X[j] - X[i]) / R)
                A[i] = G * force 

            V = random.random() * V + A
            X = np.clip(X + V, l_bound, r_bound)

            yield {
                'iteration': iteration + 1, 'current_solution': list(best_sol), 'current_score': best_score,
                'best_solution': list(best_sol), 'best_score': best_score, 'population_scores': list(scores),
                'population': X.copy()
            }