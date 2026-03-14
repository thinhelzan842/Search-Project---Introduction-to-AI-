import numpy as np
import math
from core import AlgorithmBase, ContinuousProblem

class CuckooOptimization(AlgorithmBase):
    def __init__(self, num_nests=30, max_iters=300, pa=0.25, step_size=0.5):
        self.num_nests, self.max_iters = num_nests, max_iters
        self.pa, self.step_size = pa, step_size

    def name(self) -> str: return "Cuckoo Optimization"

    def run(self, problem):
        if not isinstance(problem, ContinuousProblem):
            raise TypeError("Cuckoo Optimization requires a ContinuousProblem.")
            
        bounds = problem.get_bounds()
        dim = len(bounds)
        
        nests = np.array([problem.random_solution_generate() for _ in range(self.num_nests)])
        fitness = np.array([problem.evaluate(nest) for nest in nests])
        
        best_idx = np.argmin(fitness)
        best_nest, best_fitness = nests[best_idx].copy(), fitness[best_idx]
        
        yield {
            'iteration': 0, 'current_solution': best_nest.tolist(), 'current_score': best_fitness,
            'best_solution': best_nest.tolist(), 'best_score': best_fitness, 'population_scores': list(fitness)
        }
        
        for iteration in range(self.max_iters):
            for i in range(self.num_nests):
                new_nest = nests[i].copy()
                # Thuật toán Mantegna tạo Levy Flight
                beta = 1.5
                sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / 
                        (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
                u = np.random.normal(0, sigma, dim)
                v = np.random.normal(0, 1, dim)
                step = u / (np.abs(v) ** (1 / beta))
                
                for d in range(dim):
                    # Sử dụng step (Levy) thay vì Gaussian
                    new_nest[d] = np.clip(nests[i][d] + self.step_size * step[d], bounds[d][0], bounds[d][1])
                
                new_fitness = problem.evaluate(new_nest)
                if new_fitness < fitness[i]:
                    nests[i], fitness[i] = new_nest, new_fitness
            
            for i in range(self.num_nests):
                if np.random.random() < self.pa:
                    nests[i] = problem.random_solution_generate()
                    fitness[i] = problem.evaluate(nests[i])
            
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_fitness:
                best_fitness, best_nest = fitness[best_idx], nests[best_idx].copy()
            
            yield {
                'iteration': iteration + 1, 'current_solution': best_nest.tolist(), 'current_score': best_fitness,
                'best_solution': best_nest.tolist(), 'best_score': best_fitness, 'population_scores': list(fitness)
            }