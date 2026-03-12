import numpy as np
from core.base import AlgorithmBase

class CuckooOptimization(AlgorithmBase):
    def __init__(self, num_nests=30, max_iters=300, pa=0.25, step_size=0.5):
        self.num_nests = num_nests
        self.max_iters = max_iters
        self.pa = pa  # probability of discovery
        self.step_size = step_size

    def name(self) -> str:
        return "Cuckoo Optimization"

    def run(self, problem):
        is_min = problem.is_min_optimization()
        bounds = problem.get_bounds()
        dim = len(bounds)
        
        # Initialize nests
        nests = np.array([problem.random_solution_generate() for _ in range(self.num_nests)])
        fitness = np.array([problem.evaluate(nest) for nest in nests])
        
        best_idx = np.argmin(fitness) if is_min else np.argmax(fitness)
        best_nest = nests[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        yield {
            'iteration': 0,
            'current_solution': best_nest.tolist(),
            'current_score': best_fitness,
            'best_solution': best_nest.tolist(),
            'best_score': best_fitness
        }
        
        for iteration in range(self.max_iters):
            # Generate new cuckoo solution
            for i in range(self.num_nests):
                # Levy flight
                new_nest = nests[i].copy()
                levy = np.random.normal(0, 1, dim)
                
                for d in range(dim):
                    new_nest[d] = nests[i][d] + self.step_size * levy[d]
                    low, high = bounds[d]
                    new_nest[d] = np.clip(new_nest[d], low, high)
                
                new_fitness = problem.evaluate(new_nest)
                
                # Replace if better
                if (is_min and new_fitness < fitness[i]) or (not is_min and new_fitness > fitness[i]):
                    nests[i] = new_nest
                    fitness[i] = new_fitness
            
            # Abandon worst nests with probability pa
            for i in range(self.num_nests):
                if np.random.random() < self.pa:
                    nests[i] = problem.random_solution_generate()
                    fitness[i] = problem.evaluate(nests[i])
            
            # Update best
            best_idx = np.argmin(fitness) if is_min else np.argmax(fitness)
            if (is_min and fitness[best_idx] < best_fitness) or (not is_min and fitness[best_idx] > best_fitness):
                best_fitness = fitness[best_idx]
                best_nest = nests[best_idx].copy()
            
            yield {
                'iteration': iteration + 1,
                'current_solution': best_nest.tolist(),
                'current_score': best_fitness,
                'best_solution': best_nest.tolist(),
                'best_score': best_fitness
            }