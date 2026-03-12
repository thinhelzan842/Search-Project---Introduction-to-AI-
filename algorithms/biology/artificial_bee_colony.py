import numpy as np
from core import *

class ArtificialBeeColony(AlgorithmBase):
    def __init__(self, limit=50, popsize=20, gen=1000):
        self.size = popsize  # Number of food sources
        self.limit = limit   # Trials before a source is abandoned
        self.gen = gen

    def name(self) -> str:
        return "Artificial Bee Colony"
    
    def is_compatible(self, problem) -> bool:
        return hasattr(problem, 'get_bounds') and hasattr(problem, 'evaluate')

    def run(self, problem):
        # Initialize population (food sources)
        bounds = problem.get_bounds()
        l_bound, r_bound = np.asarray(bounds).T
        dim = len(bounds)
        pop = np.random.rand(self.size, dim)
        for i in range(self.size):
            pop[i] = l_bound + pop[i] * (r_bound - l_bound)

        # First evaluation
        fitness = np.asarray([problem.evaluate(ind) for ind in pop])
        trials = np.zeros(self.size)  # Track abandonment
        
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

        for gen in range(self.gen):
            # 1. Employed Bee Phase
            for i in range(self.size):
                # Pick a random partner j != i
                idxs = [idx for idx in range(self.size) if idx != i]
                j = np.random.choice(idxs)
                
                # Pick a random dimension to mutate
                d = np.random.randint(0, dim)
                phi = np.random.uniform(-1, 1)
                
                # Create trial solution
                trial = pop[i].copy()
                trial[d] = trial[d] + phi * (trial[d] - pop[j][d])
                trial = np.clip(trial, l_bound, r_bound)
                
                f = problem.evaluate(trial)
                if f < fitness[i]:
                    pop[i] = trial
                    fitness[i] = f
                    trials[i] = 0
                else:
                    trials[i] += 1

            # 2. Onlooker Bee Phase
            # Calculate selection probabilities based on fitness
            # (Standard ABC mapping to handle negative objective values)
            fit_vals = np.where(fitness >= 0, 1.0 / (1.0 + fitness), 1.0 + np.abs(fitness))
            probs = fit_vals / np.sum(fit_vals)
            
            m = 0
            i = 0
            while m < self.size:
                if np.random.rand() < probs[i]:
                    m += 1
                    idxs = [idx for idx in range(self.size) if idx != i]
                    j = np.random.choice(idxs)
                    
                    d = np.random.randint(0, dim)
                    phi = np.random.uniform(-1, 1)
                    
                    trial = pop[i].copy()
                    trial[d] = trial[d] + phi * (trial[d] - pop[j][d])
                    trial = np.clip(trial, l_bound, r_bound)
                    
                    f = problem.evaluate(trial)
                    if f < fitness[i]:
                        pop[i] = trial
                        fitness[i] = f
                        trials[i] = 0
                    else:
                        trials[i] += 1
                i = (i + 1) % self.size

            # Update best solution found so far
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_scr:
                best_scr = fitness[current_best_idx]
                best_sol = pop[current_best_idx].copy()

            # 3. Scout Bee Phase
            for i in range(self.size):
                if trials[i] > self.limit:
                    pop[i] = l_bound + np.random.rand(dim) * (r_bound - l_bound)
                    fitness[i] = problem.evaluate(pop[i])
                    trials[i] = 0

            yield {
                'generation': gen + 1,
                'population': pop.copy(),
                'fitness': fitness.copy(),
                'best_solution': best_sol.copy(),
                'best_score': best_scr
            }

        return best_sol, best_scr