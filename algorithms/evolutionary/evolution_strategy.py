import numpy as np
from core import AlgorithmBase, ContinuousProblem

class EvolutionStrategies(AlgorithmBase):
    def __init__(self, mu=20, lambda_=140, max_iters=300):
        self.mu = mu
        self.lambda_ = lambda_
        self.max_iters = max_iters

    def name(self) -> str:
        return f"Evolution Strategies ({self.mu} + {self.lambda_})"

    def run(self, problem):
        if not isinstance(problem, ContinuousProblem):
            raise TypeError("Evolution Strategies requires a ContinuousProblem.")

        bounds = problem.get_bounds()
        l_bound, r_bound = np.asarray(bounds).T
        dim = len(bounds)

        # Learning rates for self-adaptation
        tau_prime = 1.0 / np.sqrt(2.0 * dim)
        tau = 1.0 / np.sqrt(2.0 * np.sqrt(dim))
        max_sigma = (r_bound - l_bound) / 2.0  # Cap maximum step size

        # 1. Initialize Population (x) and Strategy Parameters (sigma)
        pop_x = np.random.uniform(l_bound, r_bound, (self.mu, dim))
        # Initialize sigma to ~10% of the search space bounds
        pop_sigma = np.random.uniform(0.01, 0.1, (self.mu, dim)) * (r_bound - l_bound)

        # Evaluate initial population
        fitness = np.array([problem.evaluate(ind) for ind in pop_x])

        best_idx = np.argmin(fitness)
        best_sol, best_score = pop_x[best_idx].copy(), fitness[best_idx]

        yield {
            'iteration': 0, 'current_solution': best_sol.tolist(), 'current_score': best_score,
            'best_solution': best_sol.tolist(), 'best_score': best_score
        }

        for iteration in range(self.max_iters):
            # 2. Generate Offspring (vectorized)
            # Randomly select parents (uniform selection is standard in basic ES)
            parent_indices = np.random.choice(self.mu, self.lambda_, replace=True)
            parents_x = pop_x[parent_indices]
            parents_sigma = pop_sigma[parent_indices]

            # Self-Adaptation of Strategy Parameters (Mutation step sizes)
            global_N = np.random.normal(0, 1, (self.lambda_, 1))
            local_N = np.random.normal(0, 1, (self.lambda_, dim))

            offspring_sigma = parents_sigma * np.exp(tau_prime * global_N + tau * local_N)
            offspring_sigma = np.clip(offspring_sigma, 1e-8,
                                      max_sigma)  # Prevent sigma from collapsing to 0 or exploding

            # Mutate Solutions
            offspring_x = parents_x + offspring_sigma * np.random.normal(0, 1, (self.lambda_, dim))
            offspring_x = np.clip(offspring_x, l_bound, r_bound)

            # Evaluate Offspring
            offspring_fitness = np.array([problem.evaluate(ind) for ind in offspring_x])

            # 3. Selection (μ + λ)
            # Combine parents and offspring
            combined_x = np.vstack((pop_x, offspring_x))
            combined_sigma = np.vstack((pop_sigma, offspring_sigma))
            combined_fitness = np.concatenate((fitness, offspring_fitness))

            # Select the top μ individuals for the next generation
            survivor_indices = np.argsort(combined_fitness)[:self.mu]

            pop_x = combined_x[survivor_indices]
            pop_sigma = combined_sigma[survivor_indices]
            fitness = combined_fitness[survivor_indices]

            # 4. Update Global Best
            if fitness[0] < best_score:
                best_score = fitness[0]
                best_sol = pop_x[0].copy()

            yield {
                'iteration': iteration + 1, 'current_solution': best_sol.tolist(), 'current_score': best_score,
                'best_solution': best_sol.tolist(), 'best_score': best_score
            }

        return best_sol, best_score