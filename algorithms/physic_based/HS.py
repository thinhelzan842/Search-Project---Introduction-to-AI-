import random
import numpy as np
from core import AlgorithmBase, ContinuousProblem

class HarmonySearch(AlgorithmBase):
    def __init__(self, hms=30, max_iters=1000, hmcr=0.9, par=0.3, bw=0.1):
        self.hms, self.max_iters = hms, max_iters
        self.hmcr, self.par, self.bw = hmcr, par, bw              

    def name(self) -> str: return "Harmony Search"

    def run(self, problem):
        if not isinstance(problem, ContinuousProblem):
            raise TypeError("Harmony Search requires a ContinuousProblem.")
            
        bounds = problem.get_bounds()
        dim = len(bounds)
        l_bound, r_bound = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])

        HM = np.random.uniform(l_bound, r_bound, (self.hms, dim))
        scores = np.array([problem.evaluate(ind) for ind in HM])

        for iteration in range(self.max_iters):
            new_harmony = np.zeros(dim)
            for j in range(dim):
                if random.random() < self.hmcr:
                    new_harmony[j] = HM[random.randint(0, self.hms - 1), j]
                    if random.random() < self.par:
                        new_harmony[j] += (random.random() * 2 - 1) * self.bw * (bounds[j][1] - bounds[j][0])
                else:
                    new_harmony[j] = random.uniform(bounds[j][0], bounds[j][1])

            new_harmony = np.clip(new_harmony, l_bound, r_bound)
            new_score = problem.evaluate(new_harmony)

            worst_idx = np.argmax(scores)
            if new_score < scores[worst_idx]:
                HM[worst_idx], scores[worst_idx] = new_harmony, new_score

            best_idx = np.argmin(scores)
            best_sol, best_score = HM[best_idx].copy(), scores[best_idx]

            yield {
                'iteration': iteration + 1, 'current_solution': list(new_harmony),
                'best_solution': list(best_sol), 'best_score': best_score, 'population_scores': list(scores),
                'population': HM.copy()
            }
