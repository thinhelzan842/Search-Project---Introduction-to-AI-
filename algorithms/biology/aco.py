import numpy as np
from core.base import AlgorithmBase

class ACO(AlgorithmBase):
    def __init__(self, num_ants=30, max_iters=300, alpha=1.0, beta=2.0, evaporation=0.1):
        self.num_ants = num_ants
        self.max_iters = max_iters
        self.alpha = alpha  # pheromone importance
        self.beta = beta    # heuristic importance
        self.evaporation = evaporation

    def name(self) -> str:
        return "ACO"

    def run(self, problem):
        is_min = problem.is_min_optimization()
        bounds = problem.get_bounds()
        dim = len(bounds)
        
        # Initialize pheromone
        pheromone = np.ones((dim, 2)) * 0.1
        
        best_solution = None
        best_score = float('inf') if is_min else float('-inf')
        
        yield {
            'iteration': 0,
            'current_solution': best_solution if best_solution is not None else problem.random_solution_generate(),
            'current_score': best_score,
            'best_solution': best_solution if best_solution is not None else problem.random_solution_generate(),
            'best_score': best_score
        }
        
        for iteration in range(self.max_iters):
            solutions = []
            scores = []
            
            # Construct solutions
            for _ in range(self.num_ants):
                solution = []
                for d in range(dim):
                    # Probabilistic choice based on pheromone
                    prob = pheromone[d] / pheromone[d].sum()
                    choice = np.random.choice([0, 1], p=prob)
                    low, high = bounds[d]
                    value = low + choice * (high - low) / 2 + np.random.uniform(-0.1, 0.1) * (high - low)
                    solution.append(np.clip(value, low, high))
                
                solutions.append(solution)
                scores.append(problem.evaluate(solution))
            
            # Find iteration best
            scores = np.array(scores)
            if is_min:
                iter_best_idx = np.argmin(scores)
            else:
                iter_best_idx = np.argmax(scores)
            
            iter_best_solution = solutions[iter_best_idx]
            iter_best_score = scores[iter_best_idx]
            
            # Update global best
            if (is_min and iter_best_score < best_score) or (not is_min and iter_best_score > best_score):
                best_score = iter_best_score
                best_solution = iter_best_solution.copy()
            
            # Evaporate pheromone
            pheromone *= (1 - self.evaporation)
            
            # Add pheromone from best solution
            for d in range(dim):
                low, high = bounds[d]
                choice = 0 if best_solution[d] < (low + high) / 2 else 1
                pheromone[d][choice] += 1.0 / best_score if is_min else best_score
            
            yield {
                'iteration': iteration + 1,
                'current_solution': best_solution,
                'current_score': best_score,
                'best_solution': best_solution,
                'best_score': best_score
            }