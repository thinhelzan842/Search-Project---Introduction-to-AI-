import numpy as np
from core import AlgorithmBase, ContinuousProblem

class FireflyAlgorithm(AlgorithmBase):
    def __init__(self, alpha=0.5, beta0=1.0, gamma=1.0, popsize=20, gen=1000):
        self.alpha, self.beta0, self.gamma = alpha, beta0, gamma
        self.size, self.gen = popsize, gen

    def name(self) -> str: return "Firefly Algorithm"

    def run(self, problem):
        if not isinstance(problem, ContinuousProblem):
            raise TypeError("Firefly Algorithm requires a ContinuousProblem.")
            
        bounds = problem.get_bounds()
        l_bound, r_bound = np.asarray(bounds).T
        dim = len(bounds)
        pop = np.random.rand(self.size, dim) * (r_bound - l_bound) + l_bound

        fitness = np.asarray([problem.evaluate(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        best_sol, best_scr = pop[best_idx].copy(), fitness[best_idx]

        yield {
            'generation': 0, 'population': pop.copy(), 'fitness': fitness.copy(),
            'best_solution': best_sol.copy(), 'best_score': best_scr, 'population_scores': list(fitness)
        }

        scale = np.abs(r_bound - l_bound)

        for gen in range(self.gen):
            for i in range(self.size):
                for j in range(self.size):
                    if fitness[j] < fitness[i]:
                        r = np.linalg.norm(pop[i] - pop[j])
                        beta = self.beta0 * np.exp(-self.gamma * (r ** 2))
                        rand_step = self.alpha * (np.random.rand(dim) - 0.5) * scale
                        
                        pop[i] = np.clip(pop[i] + beta * (pop[j] - pop[i]) + rand_step, l_bound, r_bound)
                        fitness[i] = problem.evaluate(pop[i])
                        
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_scr:
                best_scr, best_sol = fitness[current_best_idx], pop[current_best_idx].copy()
                
            yield {
                'generation': gen + 1, 'population': pop.copy(), 'fitness': fitness.copy(),
                'best_solution': best_sol.copy(), 'best_score': best_scr, 'population_scores': list(fitness)
            }
        return best_sol, best_scr