import numpy as np
from core.base import AlgorithmBase

class PSO(AlgorithmBase):
    def __init__(self, num_particles=30, max_iters=300, w=0.7, c1=1.5, c2=1.5):
        self.num_particles = num_particles
        self.max_iters = max_iters
        self.w = w  # inertia weight
        self.c1 = c1  # cognitive parameter
        self.c2 = c2  # social parameter

    def name(self) -> str:
        return "PSO"

    def run(self, problem):
        is_min = problem.is_min_optimization()
        bounds = problem.get_bounds()
        
        # Initialize particles
        positions = np.array([problem.random_solution_generate() for _ in range(self.num_particles)])
        velocities = np.random.uniform(-1, 1, positions.shape)
        
        # Evaluate initial positions
        fitness = np.array([problem.evaluate(pos) for pos in positions])
        
        # Track personal best
        pbest_positions = positions.copy()
        pbest_fitness = fitness.copy()
        
        # Track global best
        best_idx = np.argmin(fitness) if is_min else np.argmax(fitness)
        gbest_position = positions[best_idx].copy()
        gbest_fitness = fitness[best_idx]
        
        yield {
            'iteration': 0,
            'current_solution': gbest_position.tolist(),
            'current_score': gbest_fitness,
            'best_solution': gbest_position.tolist(),
            'best_score': gbest_fitness
        }
        
        for iteration in range(self.max_iters):
            # Update velocities and positions
            for i in range(self.num_particles):
                r1, r2 = np.random.random(2)
                
                # Velocity update
                velocities[i] = (self.w * velocities[i] +
                                     self.c1 * r1 * (pbest_positions[i] - positions[i]) +
                                     self.c2 * r2 * (gbest_position - positions[i]))
                
                # Position update
                positions[i] = positions[i] + velocities[i]
                
                # Boundary check
                for j, (low, high) in enumerate(bounds):
                    positions[i][j] = np.clip(positions[i][j], low, high)
            
            # Evaluate new positions
            fitness = np.array([problem.evaluate(pos) for pos in positions])
            
            # Update personal best
            for i in range(self.num_particles):
                if (is_min and fitness[i] < pbest_fitness[i]) or (not is_min and fitness[i] > pbest_fitness[i]):
                    pbest_fitness[i] = fitness[i]
                    pbest_positions[i] = positions[i].copy()
            
            # Update global best
            best_idx = np.argmin(pbest_fitness) if is_min else np.argmax(pbest_fitness)
            if (is_min and pbest_fitness[best_idx] < gbest_fitness) or (not is_min and pbest_fitness[best_idx] > gbest_fitness):
                gbest_fitness = pbest_fitness[best_idx]
                gbest_position = pbest_positions[best_idx].copy()
            
            yield {
                'iteration': iteration + 1,
                'current_solution': gbest_position.tolist(),
                'current_score': gbest_fitness,
                'best_solution': gbest_position.tolist(),
                'best_score': gbest_fitness
            }