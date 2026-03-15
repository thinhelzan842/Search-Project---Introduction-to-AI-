import numpy as np
from core import AlgorithmBase, ContinuousProblem

class ArtificialBeeColony(AlgorithmBase):
    def __init__(self, limit=50, popsize=20, gen=1000):
        self.size, self.limit, self.gen = popsize, limit, gen

    def name(self) -> str: return "Artificial Bee Colony"

    def run(self, problem):
        if not isinstance(problem, ContinuousProblem):
            raise TypeError("ABC requires a ContinuousProblem.")
            
        bounds = problem.get_bounds()
        l_bound, r_bound = np.asarray(bounds).T
        dim = len(bounds)
        pop = np.random.rand(self.size, dim) * (r_bound - l_bound) + l_bound

        fitness = np.asarray([problem.evaluate(ind) for ind in pop])
        trials = np.zeros(self.size)
        
        best_idx = np.argmin(fitness)
        best_sol, best_scr = pop[best_idx].copy(), fitness[best_idx]
        
        yield {
            'generation': 0, 'population': pop.copy(), 'fitness': fitness.copy(),
            'best_solution': best_sol.copy(), 'best_score': best_scr, 'population_scores': list(fitness)
        }

        for gen in range(self.gen):
            # Employed Bees
            for i in range(self.size):
                j = np.random.choice([idx for idx in range(self.size) if idx != i])
                d = np.random.randint(0, dim)
                phi = np.random.uniform(-1, 1)
                
                trial = pop[i].copy()
                trial[d] = np.clip(trial[d] + phi * (trial[d] - pop[j][d]), l_bound[d], r_bound[d])
                
                f = problem.evaluate(trial)
                if f < fitness[i]:
                    pop[i], fitness[i], trials[i] = trial, f, 0
                else:
                    trials[i] += 1

            # Onlooker Bees
            fit_vals = np.where(fitness >= 0, 1.0 / (1.0 + fitness), 1.0 + np.abs(fitness))
            probs = fit_vals / np.sum(fit_vals)
            
            m, i = 0, 0
            while m < self.size:
                if np.random.rand() < probs[i]:
                    m += 1
                    j = np.random.choice([idx for idx in range(self.size) if idx != i])
                    d = np.random.randint(0, dim)
                    phi = np.random.uniform(-1, 1)
                    
                    trial = pop[i].copy()
                    trial[d] = np.clip(trial[d] + phi * (trial[d] - pop[j][d]), l_bound[d], r_bound[d])
                    
                    f = problem.evaluate(trial)
                    if f < fitness[i]:
                        pop[i], fitness[i], trials[i] = trial, f, 0
                    else:
                        trials[i] += 1
                i = (i + 1) % self.size

            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_scr:
                best_scr, best_sol = fitness[current_best_idx], pop[current_best_idx].copy()

            # Scout Bees
            for i in range(self.size):
                if trials[i] > self.limit:
                    pop[i] = l_bound + np.random.rand(dim) * (r_bound - l_bound)
                    fitness[i], trials[i] = problem.evaluate(pop[i]), 0

            yield {
                'generation': gen + 1, 'population': pop.copy(), 'fitness': fitness.copy(),
                'best_solution': best_sol.copy(), 'best_score': best_scr, 'population_scores': list(fitness)
            }
        return best_sol, best_scr