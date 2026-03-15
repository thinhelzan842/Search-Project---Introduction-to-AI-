import numpy as np
from core import AlgorithmBase, ContinuousProblem

class PSO(AlgorithmBase):
    def __init__(self, num_particles=30, max_iters=300, w=0.7, c1=1.5, c2=1.5):
        self.num_particles, self.max_iters = num_particles, max_iters
        self.w, self.c1, self.c2 = w, c1, c2

    def name(self) -> str: return "PSO"

    def run(self, problem):
        if not isinstance(problem, ContinuousProblem):
            raise TypeError("PSO requires a ContinuousProblem.")
            
        bounds = problem.get_bounds()
        positions = np.array([problem.random_solution_generate() for _ in range(self.num_particles)])
        velocities = np.random.uniform(-1, 1, positions.shape)
        
        fitness = np.array([problem.evaluate(pos) for pos in positions])
        pbest_positions, pbest_fitness = positions.copy(), fitness.copy()
        
        best_idx = np.argmin(fitness)
        gbest_position, gbest_fitness = positions[best_idx].copy(), fitness[best_idx]
        
        yield {
            'iteration': 0, 'current_solution': gbest_position.tolist(), 'current_score': gbest_fitness,
            'best_solution': gbest_position.tolist(), 'best_score': gbest_fitness, 'population_scores': list(fitness),
            'population_positions': positions.tolist()
        }
        
        for iteration in range(self.max_iters):
            for i in range(self.num_particles):
                r1, r2 = np.random.random(2)
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (pbest_positions[i] - positions[i]) +
                                 self.c2 * r2 * (gbest_position - positions[i]))
                positions[i] += velocities[i]
                
                for j, (low, high) in enumerate(bounds):
                    positions[i][j] = np.clip(positions[i][j], low, high)
            
            fitness = np.array([problem.evaluate(pos) for pos in positions])
            
            # Strict Minimization Updates
            for i in range(self.num_particles):
                if fitness[i] < pbest_fitness[i]:
                    pbest_fitness[i], pbest_positions[i] = fitness[i], positions[i].copy()
            
            best_idx = np.argmin(pbest_fitness)
            if pbest_fitness[best_idx] < gbest_fitness:
                gbest_fitness, gbest_position = pbest_fitness[best_idx], pbest_positions[best_idx].copy()
            
            yield {
                'iteration': iteration + 1, 'current_solution': gbest_position.tolist(), 'current_score': gbest_fitness,
                'best_solution': gbest_position.tolist(), 'best_score': gbest_fitness, 'population_scores': list(fitness),
                'population_positions': positions.tolist()
            }