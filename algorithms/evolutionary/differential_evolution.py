import numpy as np

from core import *

class DifferentialEvolution(AlgorithmBase):
    def __init__(self, mut=0.8, crossp=0.7, popsize=20, gen=1000):
        self.mut = mut
        self.crossp = crossp
        self.size = popsize
        self.gen = gen

    def name(self) -> str:
        return "Differential Evolution"
    
    def run(self, problem):
        #initialize population
        bounds = problem.get_bounds()
        l_bound, r_bound = np.asarray(bounds).T
        dim = len(bounds)
        pop = np.random.rand(self.size, dim)
        for i in range(self.size):
            pop[i] = l_bound + pop[i] * (r_bound - l_bound)

        #first evaluation
        fitness = np.asarray([problem.evaluate(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        best_sol = pop[best_idx]
        best_scr = fitness[best_idx]
        yield {
            'generation'    :0,
            'population'    :pop.copy(),
            'fitness'       :fitness.copy(),
            'best_solution' :best_sol.copy(),
            'best_score'    :best_scr
        }

        for gen in range(self.gen):
            for i in range(self.size):
                #select three random indices that are not the current index
                idxs = [idx for idx in range(self.size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]

                #mutation and crossover
                mutant = np.clip(a + self.mut * (b - c), l_bound, r_bound)
                cross_points = np.random.rand(dim) < self.crossp
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, dim)] = True

                #evaluate the trial vector
                trial = np.where(cross_points, mutant, pop[i])
                f = problem.evaluate(trial)
                if f < fitness[i]:
                    pop[i] = trial
                    fitness[i] = f
                    if f < best_scr:
                        best_scr = f
                        best_sol = trial
            yield {
                'generation'    :gen + 1,
                'population'    :pop.copy(),
                'fitness'       :fitness.copy(),
                'best_solution' :best_sol.copy(),
                'best_score'    :best_scr
            }


        return best_sol, best_scr