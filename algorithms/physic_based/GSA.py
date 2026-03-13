import random
import numpy as np
from core.base import AlgorithmBase

class GravitationalSearchAlgorithm(AlgorithmBase):
    def __init__(self, pop_size=30, max_iters=500, G0=100.0, alpha=20.0):
        self.pop_size = pop_size
        self.max_iters = max_iters
        self.G0 = G0      
        self.alpha = alpha

    def name(self) -> str:
        return "Gravitational Search Algorithm"

    def run(self, problem):
        if problem.is_discrete():
            raise NotImplementedError("X")
            
        is_min = problem.is_min_optimization()
        bounds = problem.get_bounds()
        dim = len(bounds)
        l_bound = np.array([b[0] for b in bounds])
        r_bound = np.array([b[1] for b in bounds])

        # Intialize Population (Positions) and Velocities
        X = np.random.uniform(l_bound, r_bound, (self.pop_size, dim))
        V = np.zeros((self.pop_size, dim))
        
        best_sol = None
        best_score = float('inf') if is_min else float('-inf')

        for iteration in range(self.max_iters):
            # Fitness Evaluation
            scores = np.array([problem.evaluate(ind) for ind in X])
            
            # Update current best/worst scores  
            curr_best_idx = np.argmin(scores) if is_min else np.argmax(scores)
            curr_worst_idx = np.argmax(scores) if is_min else np.argmin(scores)
            
            if (is_min and scores[curr_best_idx] < best_score) or (not is_min and scores[curr_best_idx] > best_score):
                best_score = scores[curr_best_idx]
                best_sol = X[curr_best_idx].copy()

            # Calc Mass (Scale) for each particle
            best_val, worst_val = scores[curr_best_idx], scores[curr_worst_idx]
            
            if best_val == worst_val:
                M = np.ones(self.pop_size)
            else:
                m = (scores - worst_val) / (best_val - worst_val)
                M = m / np.sum(m) 

            # Calc Gravitational Constant G
            G = self.G0 * np.exp(-self.alpha * (iteration / self.max_iters))

            # Calc Acceleration for each particle
            A = np.zeros((self.pop_size, dim))
            for i in range(self.pop_size):
                force = np.zeros(dim)
                for j in range(self.pop_size):
                    if i != j:
                        R = np.linalg.norm(X[i] - X[j]) + 1e-8 
                        force += random.random() * M[j] * ((X[j] - X[i]) / R)
                A[i] = G * force 

            # Update Velocity and Position
            V = random.random() * V + A
            X = X + V
            
            # Boundary constraints
            X = np.clip(X, l_bound, r_bound)

            yield {
                'iteration': iteration + 1,
                'current_solution': list(best_sol),
                'current_score': best_score,
                'best_solution': list(best_sol),
                'best_score': best_score
            }

            if problem.is_goal(best_sol):
                break

        return best_sol, best_score