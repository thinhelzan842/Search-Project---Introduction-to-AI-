import numpy as np
from core import *

class FireflyAlgorithm(AlgorithmBase):
    def __init__(self, alpha=0.5, beta0=1.0, gamma=1.0, popsize=20, gen=1000):
        self.alpha = alpha   # Randomness
        self.beta0 = beta0   # Base attraction
        self.gamma = gamma   # Light absorption
        self.size = popsize
        self.gen = gen

    def name(self) -> str:
        return "Firefly Algorithm"
        
    def is_compatible(self, problem) -> bool:
        return hasattr(problem, 'get_bounds') and hasattr(problem, 'evaluate')

    def run(self, problem):
        # Initialize population
        bounds = problem.get_bounds()
        l_bound, r_bound = np.asarray(bounds).T
        dim = len(bounds)
        pop = np.random.rand(self.size, dim)
        for i in range(self.size):
            pop[i] = l_bound + pop[i] * (r_bound - l_bound)

        # First evaluation
        fitness = np.asarray([problem.evaluate(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        best_sol = pop[best_idx].copy()
        best_scr = fitness[best_idx]

        yield {
            'generation': 0,
            'population': pop.copy(),
            'fitness': fitness.copy(),
            'best_solution': best_sol.copy(),
            'best_score': best_scr
        }

        # Scale factor for alpha (random step) relative to bounds
        scale = np.abs(r_bound - l_bound)

        for gen in range(self.gen):
            for i in range(self.size):
                for j in range(self.size):
                    # In minimization, smaller fitness is "brighter"
                    if fitness[j] < fitness[i]:
                        # Euclidean distance
                        r = np.linalg.norm(pop[i] - pop[j])
                        
                        # Calculate attraction
                        beta = self.beta0 * np.exp(-self.gamma * (r ** 2))
                        
                        # Move firefly i towards j with random noise
                        rand_step = self.alpha * (np.random.rand(dim) - 0.5) * scale
                        pop[i] = pop[i] + beta * (pop[j] - pop[i]) + rand_step
                        
                        # Enforce bounds
                        pop[i] = np.clip(pop[i], l_bound, r_bound)
                        
                        # Re-evaluate moved firefly
                        fitness[i] = problem.evaluate(pop[i])
                        
            # Update best overall solution
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_scr:
                best_scr = fitness[current_best_idx]
                best_sol = pop[current_best_idx].copy()
                
            yield {
                'generation': gen + 1,
                'population': pop.copy(),
                'fitness': fitness.copy(),
                'best_solution': best_sol.copy(),
                'best_score': best_scr
            }

        return best_sol, best_scr