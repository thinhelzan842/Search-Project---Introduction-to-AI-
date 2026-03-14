import random
from core import AlgorithmBase, ContinuousProblem, DiscreteSearchProblem

class TabuSearch(AlgorithmBase):
    def __init__(self, max_iters=500, tabu_tenure=10, num_neighbors=20, step_size=0.1, step_decay=0.99):
        self.max_iters, self.tabu_tenure = max_iters, tabu_tenure
        self.num_neighbors, self.step_size, self.step_decay = num_neighbors, step_size, step_decay

    def name(self) -> str: return "Tabu Search"

    def _get_neighbor(self, current, problem, current_step_size):
        if isinstance(problem, ContinuousProblem):
            neighbor = list(current)
            bounds = problem.get_bounds()
            for i in range(len(neighbor)):
                val = neighbor[i] + random.gauss(0, current_step_size)
                neighbor[i] = max(bounds[i][0], min(bounds[i][1], val))
            return neighbor
            
        elif isinstance(problem, DiscreteSearchProblem):
            neighbor = list(current)
            if not neighbor:
                return neighbor
                
            # Copy logic phân loại từ SA.py sang
            is_binary = all(x in (0, 1) for x in neighbor)
            
            if is_binary:
                # Đột biến lật bit cho Knapsack
                idx = random.randrange(len(neighbor))
                neighbor[idx] = 1 - neighbor[idx]
            elif len(neighbor) > 1:
                # Đột biến hoán vị cho TSP, Graph Coloring
                idx1, idx2 = random.sample(range(len(neighbor)), 2)
                neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
                
            return neighbor
            
        raise TypeError("Unsupported problem type.")

    def _hash_sol(self, sol, is_cont):
        if is_cont:
            return tuple(round(x, 4) for x in sol)
        return tuple(sol) if isinstance(sol, list) else sol

    def run(self, problem):
        current_sol = problem.random_solution_generate()
        current_score = problem.evaluate(current_sol)
        best_sol, best_score = current_sol, current_score
        
        is_cont = isinstance(problem, ContinuousProblem)
        tabu_list = {self._hash_sol(current_sol, is_cont): self.tabu_tenure}
        current_step_size = self.step_size

        yield {
            'iteration': 0, 'current_solution': current_sol, 'current_score': current_score,
            'best_solution': best_sol, 'best_score': best_score
        }

        for iteration in range(self.max_iters):
            best_candidate_sol = None
            best_candidate_score = float('inf')
            
            for _ in range(self.num_neighbors):
                n_sol = self._get_neighbor(current_sol, problem, current_step_size)
                n_score = problem.evaluate(n_sol)
                n_hash = self._hash_sol(n_sol, is_cont)
                
                is_tabu = n_hash in tabu_list and tabu_list[n_hash] >= iteration
                aspiration = n_score < best_score
                
                if not is_tabu or aspiration:
                    if n_score < best_candidate_score:
                        best_candidate_score, best_candidate_sol = n_score, n_sol

            if best_candidate_sol is None:
                best_candidate_sol = self._get_neighbor(current_sol, problem, current_step_size)
                best_candidate_score = problem.evaluate(best_candidate_sol)

            current_sol, current_score = best_candidate_sol, best_candidate_score
            
            if current_score < best_score:
                best_sol, best_score = current_sol, current_score
                
            tabu_list[self._hash_sol(current_sol, is_cont)] = iteration + self.tabu_tenure

            yield {
                'iteration': iteration + 1, 'current_solution': current_sol, 'current_score': current_score,
                'best_solution': best_sol, 'best_score': best_score
            }

            if is_cont:
                current_step_size *= self.step_decay
            elif hasattr(problem, 'is_goal') and problem.is_goal(best_sol): 
                break

        return best_sol, best_score