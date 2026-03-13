import numpy as np
from core import AlgorithmBase, ContinuousProblem

class ACO(AlgorithmBase):
    def __init__(self, num_ants=30, max_iters=300, alpha=1.0, beta=2.0, evaporation=0.1):
        self.num_ants, self.max_iters = num_ants, max_iters
        self.alpha, self.beta, self.evaporation = alpha, beta, evaporation

    def name(self) -> str: return "ACO"

    def run(self, problem):
        if not isinstance(problem, ContinuousProblem):
            raise TypeError("This ACO implementation requires a ContinuousProblem.")
            
        bounds = problem.get_bounds()
        dim = len(bounds)
        pheromone = np.ones((dim, 2)) * 0.1
        
        best_solution = problem.random_solution_generate()
        best_score = float('inf')
        
        yield {
            'iteration': 0, 'current_solution': best_solution, 'current_score': best_score,
            'best_solution': best_solution, 'best_score': best_score
        }
        
        for iteration in range(self.max_iters):
            solutions, scores = [], []
            
            for _ in range(self.num_ants):
                solution = []
                for d in range(dim):
                    prob = pheromone[d] / pheromone[d].sum()
                    choice = np.random.choice([0, 1], p=prob)
                    low, high = bounds[d]
                    value = low + choice * (high - low) / 2 + np.random.uniform(-0.1, 0.1) * (high - low)
                    solution.append(np.clip(value, low, high))
                
                solutions.append(solution)
                scores.append(problem.evaluate(solution))
            
            scores = np.array(scores)
            iter_best_idx = np.argmin(scores)
            iter_best_solution, iter_best_score = solutions[iter_best_idx], scores[iter_best_idx]
            
            if iter_best_score < best_score:
                best_score, best_solution = iter_best_score, iter_best_solution.copy()
            
            pheromone *= (1 - self.evaporation)
            
            for d in range(dim):
                low, high = bounds[d]
                choice = 0 if best_solution[d] < (low + high) / 2 else 1
                # Add pheromone inversely proportional to the strict minimization score
                pheromone[d][choice] += 1.0 / (abs(best_score) + 1e-8) 
            
            yield {
                'iteration': iteration + 1, 'current_solution': best_solution, 'current_score': best_score,
                'best_solution': best_solution, 'best_score': best_score
            }