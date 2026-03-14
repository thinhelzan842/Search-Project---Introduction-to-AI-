import numpy as np
import random
from collections import defaultdict
from core import AlgorithmBase, DiscreteSearchProblem

class ACO(AlgorithmBase):
    def __init__(self, num_ants=30, max_iters=300, alpha=1.0, beta=2.0, evaporation=0.1):
        self.num_ants = num_ants
        self.max_iters = max_iters
        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation

    def name(self) -> str: 
        return "ACO"

    def run(self, problem):
        if not isinstance(problem, DiscreteSearchProblem):
            raise TypeError("This ACO implementation requires a DiscreteSearchProblem.")
            
        # Dùng dictionary để lưu Pheromone linh hoạt: (u, v) -> mức độ mùi
        # u: context (node hiện tại hoặc index lựa chọn), v: quyết định tiếp theo
        tau = defaultdict(lambda: 0.1)
        
        best_solution = None
        best_score = float('inf')
        
        yield {
            'iteration': 0, 'current_solution': problem.get_initial_state(), 'current_score': best_score,
            'best_solution': best_solution, 'best_score': best_score
        }
        
        for iteration in range(self.max_iters):
            solutions = []
            scores = []
            
            # 1. Giai đoạn Kiến đi tìm đường (Construct Solutions)
            for _ in range(self.num_ants):
                state = problem.get_initial_state()
                # Phân biệt bài toán Dạng Tuyến (TSP/Path) hay Dạng Gán (Coloring/Knapsack)
                is_path_problem = len(state) > 0  
                
                while not problem.is_goal(state):
                    neighbors = problem.get_neighbors(state)
                    if not neighbors:
                        break
                        
                    valid_neighbors = []
                    heuristic_vals = []
                    
                    for nb in neighbors:
                        # evaluate() dùng để chặn rẽ vào các nhánh vi phạm luật
                        if problem.evaluate(nb) != float('inf'):
                            valid_neighbors.append(nb)
                            # Ưu tiên các trạng thái có Heuristic estimate nhỏ (càng nhỏ càng tốt)
                            h = problem.get_heuristic(nb)
                            heuristic_vals.append(1.0 / (1.0 + max(0, h)) if h != float('inf') else 0.0)
                    
                    if not valid_neighbors:
                        state = random.choice(neighbors) # Ngõ cụt, chọn bừa để kết thúc
                        break
                        
                    probs = []
                    for i, nb in enumerate(valid_neighbors):
                        v = nb[-1]
                        u = state[-1] if is_path_problem else len(state)
                        
                        p_pheromone = tau[(u, v)] ** self.alpha
                        p_heuristic = heuristic_vals[i] ** self.beta
                        probs.append(p_pheromone * p_heuristic)
                        
                    probs = np.array(probs)
                    sum_probs = probs.sum()
                    if sum_probs == 0 or np.isnan(sum_probs):
                        probs = np.ones(len(valid_neighbors)) / len(valid_neighbors)
                    else:
                        probs /= sum_probs
                        
                    # Kiến đưa ra quyết định dựa trên xác suất
                    state = valid_neighbors[np.random.choice(len(valid_neighbors), p=probs)]
                    
                solutions.append(state)
                scores.append(problem.evaluate(state))
            
            # 2. Cập nhật Global Best
            valid_indices = [i for i, s in enumerate(scores) if s != float('inf')]
            if valid_indices:
                iter_best_idx = min(valid_indices, key=lambda i: scores[i])
                if scores[iter_best_idx] < best_score:
                    best_score = scores[iter_best_idx]
                    best_solution = solutions[iter_best_idx]
            
            # 3. Mùi Pheromone bay hơi
            for key in tau:
                tau[key] *= (1 - self.evaporation)
                
            # 4. Rải thêm mùi Pheromone dựa trên Rank của Kiến
            if valid_indices:
                f_scores = [scores[i] for i in valid_indices]
                max_s = max(f_scores)
                min_s = min(f_scores)
                
                for i in valid_indices:
                    sol = solutions[i]
                    score = scores[i]
                    
                    # Kiến tìm được điểm càng nhỏ (min-optimization) thì nhận reward càng lớn
                    if max_s == min_s:
                        reward = 1.0
                    else:
                        reward = (max_s - score) / (max_s - min_s + 1e-8) + 0.1
                        
                    start_idx = len(problem.get_initial_state())
                    for step in range(start_idx, len(sol)):
                        v = sol[step]
                        if is_path_problem:
                            u = sol[step - 1]
                        else:
                            u = step - 1
                        tau[(u, v)] += reward

            yield {
                'iteration': iteration + 1, 'current_solution': best_solution, 'current_score': best_score,
                'best_solution': best_solution, 'best_score': best_score
            }