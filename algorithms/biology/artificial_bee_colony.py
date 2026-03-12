import numpy as np
import random

class ArtificialBeeColony(AlgorithmBase):
    def __init__(self, n_bees=30, limit=20):
        self.n_bees = n_bees
        self.limit = limit  # Abandonment threshold

    def name(self) -> str:
        return "Artificial Bee Colony"

    def run(self, problem):
        # Initialize sources (half the bees are employed)
        n_sources = self.n_bees // 2
        sources = [problem.get_random_solution() for _ in range(n_sources)]
        fitness = [problem.evaluate(s) for s in sources]
        trials = [0] * n_sources # Track failed improvements

        for _ in range(problem.max_iter):
            # 1. Employed Bees
            for i in range(n_sources):
                new_sol = problem.local_search(sources[i])
                if problem.evaluate(new_sol) < fitness[i]:
                    sources[i], fitness[i], trials[i] = new_sol, problem.evaluate(new_sol), 0
                else:
                    trials[i] += 1

            # 2. Onlooker Bees (Selection based on fitness probability)
            probs = [1/f if f != 0 else 1 for f in fitness]
            total = sum(probs)
            probs = [p/total for p in probs]
            
            for _ in range(n_sources):
                i = np.random.choice(range(n_sources), p=probs)
                new_sol = problem.local_search(sources[i])
                if problem.evaluate(new_sol) < fitness[i]:
                    sources[i], fitness[i], trials[i] = new_sol, problem.evaluate(new_sol), 0
                else:
                    trials[i] += 1

            # 3. Scout Bees
            for i in range(n_sources):
                if trials[i] > self.limit:
                    sources[i] = problem.get_random_solution()
                    fitness[i] = problem.evaluate(sources[i])
                    trials[i] = 0
                    
        return sources[np.argmin(fitness)]