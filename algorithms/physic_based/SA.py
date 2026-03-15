import math
import random
from core import AlgorithmBase, ContinuousProblem, DiscreteSearchProblem

class SimulatedAnnealing(AlgorithmBase):
    def __init__(self, max_epochs=1000, initial_temp=100.0, cooling_rate=0.99, step_size=0.5, markov_chain_length=20, step_decay=0.99):
        self.max_epochs = max_epochs
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.step_size = step_size
        self.markov_chain_length = markov_chain_length
        self.step_decay = step_decay

    def name(self) -> str: 
        return "Simulated Annealing"

    def _get_neighbor(self, current, problem, current_step_size):
        if isinstance(problem, ContinuousProblem):
            neighbor = list(current)
            bounds = problem.get_bounds()
            for i in range(len(neighbor)):
                val = neighbor[i] + random.gauss(0, current_step_size)
                neighbor[i] = max(bounds[i][0], min(bounds[i][1], val))
            return neighbor
            
        elif isinstance(problem, DiscreteSearchProblem):
            # Cấu trúc lại không gian lân cận (Neighborhood) theo dạng đột biến (Perturbation)
            neighbor = list(current)
            if not neighbor:
                return neighbor
                
            # Phân loại để dùng chiến lược đột biến phù hợp
            is_binary = all(x in (0, 1) for x in neighbor)
            
            if is_binary:
                # Đột biến lật bit (Bit-flip) cho các bài toán như Knapsack
                idx = random.randrange(len(neighbor))
                neighbor[idx] = 1 - neighbor[idx]
            elif len(neighbor) > 1:
                # Đột biến hoán vị (Swap) cho các bài toán như TSP, Graph Coloring
                idx1, idx2 = random.sample(range(len(neighbor)), 2)
                neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
                
            return neighbor
            
        raise TypeError("Problem type not supported by SA.")

    def run(self, problem):
        current_sol = problem.random_solution_generate()
        current_score = problem.evaluate(current_sol)
        best_sol = list(current_sol) if isinstance(current_sol, list) else current_sol
        best_score = current_score
        
        temp = self.initial_temp
        current_step_size = self.step_size

        yield {
            'iteration': 0, 'current_solution': current_sol, 'current_score': current_score,
            'best_solution': best_sol, 'best_score': best_score, 'temperature': temp
        }

        for epoch in range(self.max_epochs):
            for _ in range(self.markov_chain_length):
                neighbor_sol = self._get_neighbor(current_sol, problem, current_step_size)
                neighbor_score = problem.evaluate(neighbor_sol)
                
                # Hàm evaluate luôn được thiết kế theo hướng Minimization (Tìm Min)
                delta_e = neighbor_score - current_score
                
                if delta_e < 0:
                    current_sol, current_score = neighbor_sol, neighbor_score
                    if current_score < best_score:
                        best_sol, best_score = neighbor_sol, current_score
                else: 
                    if temp > 1e-8:
                        try:
                            # Xác suất chấp nhận giải pháp tồi hơn
                            if random.random() < math.exp(-delta_e / temp):
                                current_sol, current_score = neighbor_sol, neighbor_score
                        except OverflowError:
                            pass
            
            temp *= self.cooling_rate
            
            yield {
                'iteration': epoch + 1, 'current_solution': current_sol, 'current_score': current_score,
                'best_solution': best_sol, 'best_score': best_score, 'temperature': temp
            }

            if isinstance(problem, ContinuousProblem):
                current_step_size *= self.step_decay
            elif hasattr(problem, 'get_optimal_value') and best_score <= problem.get_optimal_value() + 1e-9: 
                break # Early stop nếu đã đạt mục tiêu

        return best_sol, best_score