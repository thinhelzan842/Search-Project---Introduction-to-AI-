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
            neighbors = problem.get_neighbors(current)
            return random.choice(neighbors) if neighbors else current
        raise TypeError("Unsupported problem type.")

    def _hash_sol(self, sol, is_continuous):
        if is_continuous:
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

"""import random
import numpy as np
from core.base import AlgorithmBase

class TabuSearch(AlgorithmBase):
    def __init__(self, max_iters=500, tabu_tenure=10, num_neighbors=20, step_size=0.1, step_decay=0.99):
        self.max_iters = max_iters
        self.tabu_tenure = tabu_tenure
        self.num_neighbors = num_neighbors
        self.step_size = step_size
        self.step_decay = step_decay

    def name(self) -> str:
        return "Tabu Search"

    def _get_neighbor(self, current, problem, current_step_size):
        neighbor = list(current)
        if not problem.is_discrete():
            bounds = problem.get_bounds()
            for i in range(len(neighbor)):
                val = neighbor[i] + random.gauss(0, current_step_size)
                neighbor[i] = max(bounds[i][0], min(bounds[i][1], val))
        else:
            if len(neighbor) > 1:
                idx1, idx2 = random.sample(range(len(neighbor)), 2)
                neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
        return neighbor

    def _hash_sol(self, sol, is_discrete):
        return tuple(sol) if is_discrete else tuple(round(x, 4) for x in sol)

    def run(self, problem):
        current_sol = problem.random_solution_generate()
        current_score = problem.evaluate(current_sol)
        best_sol, best_score = list(current_sol), current_score
        
        is_min = problem.is_min_optimization()
        is_discrete = problem.is_discrete()
        
        tabu_list = {self._hash_sol(current_sol, is_discrete): self.tabu_tenure}

        current_step_size = self.step_size

        yield {
            'iteration': 0,
            'current_solution': list(current_sol),
            'current_score': current_score,
            'best_solution': list(best_sol),
            'best_score': best_score
        }

        for iteration in range(self.max_iters):
            best_candidate_sol = None
            best_candidate_score = float('inf') if is_min else float('-inf')
            
            # Survey the neighborhood  
            for _ in range(self.num_neighbors):
                n_sol = self._get_neighbor(current_sol, problem, current_step_size)
                n_score = problem.evaluate(n_sol)
                n_hash = self._hash_sol(n_sol, is_discrete)
                
                is_tabu = n_hash in tabu_list and tabu_list[n_hash] >= iteration
                aspiration = (is_min and n_score < best_score) or (not is_min and n_score > best_score)
                
                if not is_tabu or aspiration:
                    if (is_min and n_score < best_candidate_score) or (not is_min and n_score > best_candidate_score):
                        best_candidate_score = n_score
                        best_candidate_sol = list(n_sol)

            # Choose the random neighbor not tabu 
            if best_candidate_sol is None:
                best_candidate_sol = self._get_neighbor(current_sol, problem, current_step_size)
                best_candidate_score = problem.evaluate(best_candidate_sol)

            current_sol, current_score = best_candidate_sol, best_candidate_score
            
            # Update Global Best
            if (is_min and current_score < best_score) or (not is_min and current_score > best_score):
                best_sol, best_score = list(current_sol), current_score
                
            # take note in the tabu list 
            tabu_list[self._hash_sol(current_sol, is_discrete)] = iteration + self.tabu_tenure

            yield {
                'iteration': iteration + 1,
                'current_solution': list(current_sol),
                'current_score': current_score,
                'best_solution': list(best_sol),
                'best_score': best_score
            }

            if not problem.is_discrete():
                current_step_size *= self.step_decay
                if problem.is_goal(best_sol): 
                    break

        return best_sol, best_score"""