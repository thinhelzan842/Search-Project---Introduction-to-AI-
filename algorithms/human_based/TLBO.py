import random
import numpy as np
from core import AlgorithmBase, ContinuousProblem

class TLBO(AlgorithmBase):
    def __init__(self, pop_size=20, max_iters=500):
        self.pop_size, self.max_iters = pop_size, max_iters

    def name(self) -> str: return "Teaching-Learning-Based Optimization"

    def run(self, problem):
        if not isinstance(problem, ContinuousProblem):
            raise TypeError("TLBO requires a ContinuousProblem.")
            
        bounds = problem.get_bounds()
        dim = len(bounds)
        l_bound, r_bound = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])

        pop = np.random.uniform(l_bound, r_bound, (self.pop_size, dim))
        scores = np.array([problem.evaluate(ind) for ind in pop])

        best_idx = np.argmin(scores)
        best_sol, best_score = pop[best_idx].copy(), scores[best_idx]

        yield {
            'iteration': 0, 'current_solution': list(best_sol), 'current_score': best_score,
            'best_solution': list(best_sol), 'best_score': best_score, 'population_scores': list(scores)
        }

        for iteration in range(self.max_iters):
            mean_learner = np.mean(pop, axis=0)
            
            for i in range(self.pop_size):
                # Teaching Phase
                T_F = random.randint(1, 2) 
                r_teacher = np.random.rand(dim)
                
                new_sol_teacher = np.clip(pop[i] + r_teacher * (best_sol - T_F * mean_learner), l_bound, r_bound)
                new_score_teacher = problem.evaluate(new_sol_teacher)
                
                if new_score_teacher < scores[i]:
                    pop[i], scores[i] = new_sol_teacher, new_score_teacher

                # Learning Phase
                j = random.choice([idx for idx in range(self.pop_size) if idx != i])
                r_learner = np.random.rand(dim)
                
                if scores[i] < scores[j]:
                    new_sol_learner = pop[i] + r_learner * (pop[i] - pop[j])
                else:
                    new_sol_learner = pop[i] + r_learner * (pop[j] - pop[i])
                    
                new_sol_learner = np.clip(new_sol_learner, l_bound, r_bound)
                new_score_learner = problem.evaluate(new_sol_learner)
                
                if new_score_learner < scores[i]:
                    pop[i], scores[i] = new_sol_learner, new_score_learner

            best_idx = np.argmin(scores)
            if scores[best_idx] < best_score:
                best_sol, best_score = pop[best_idx].copy(), scores[best_idx]

            yield {
                'iteration': iteration + 1, 'current_solution': list(best_sol), 'current_score': best_score,
                'best_solution': list(best_sol), 'best_score': best_score, 'population_scores': list(scores)
            }

"""import random
import numpy as np
from core.base import AlgorithmBase

class TLBO(AlgorithmBase):
    def __init__(self, pop_size=20, max_iters=500):
        self.pop_size = pop_size
        self.max_iters = max_iters

    def name(self) -> str:
        return "Teaching-Learning-Based Optimization"

    def run(self, problem):
        if problem.is_discrete():
            raise NotImplementedError("TLBO standard version requires a Continuous Problem.")
            
        is_min = problem.is_min_optimization()
        bounds = problem.get_bounds()
        dim = len(bounds)
        l_bound = np.array([b[0] for b in bounds])
        r_bound = np.array([b[1] for b in bounds])

        # Initialize population
        pop = np.random.uniform(l_bound, r_bound, (self.pop_size, dim))
        scores = np.array([problem.evaluate(ind) for ind in pop])

        # Find Teacher
        best_idx = np.argmin(scores) if is_min else np.argmax(scores)
        best_sol = pop[best_idx].copy()
        best_score = scores[best_idx]

        yield {
            'iteration': 0,
            'current_solution': list(best_sol),
            'current_score': best_score,
            'best_solution': list(best_sol),
            'best_score': best_score
        }

        
        for iteration in range(self.max_iters):
            # Calculate average knowledge level of the class
            mean_learner = np.mean(pop, axis=0)
            
            for i in range(self.pop_size):
                T_F = random.randint(1, 2) 
                r_teacher = np.random.rand(dim)
                
                # Update T & M
                new_sol_teacher = pop[i] + r_teacher * (best_sol - T_F * mean_learner)
                new_sol_teacher = np.clip(new_sol_teacher, l_bound, r_bound) 
                
                new_score_teacher = problem.evaluate(new_sol_teacher)
                
                if (is_min and new_score_teacher < scores[i]) or (not is_min and new_score_teacher > scores[i]):
                    pop[i] = new_sol_teacher
                    scores[i] = new_score_teacher

                
                # choose random learner
                j = i
                while j == i:
                    j = random.randint(0, self.pop_size - 1)
                
                r_learner = np.random.rand(dim)
                
                if (is_min and scores[i] < scores[j]) or (not is_min and scores[i] > scores[j]):
                    new_sol_learner = pop[i] + r_learner * (pop[i] - pop[j])
                else:
                    new_sol_learner = pop[i] + r_learner * (pop[j] - pop[i])
                    
                new_sol_learner = np.clip(new_sol_learner, l_bound, r_bound)
                new_score_learner = problem.evaluate(new_sol_learner)
                
                if (is_min and new_score_learner < scores[i]) or (not is_min and new_score_learner > scores[i]):
                    pop[i] = new_sol_learner
                    scores[i] = new_score_learner

            # Update global best 
            best_idx = np.argmin(scores) if is_min else np.argmax(scores)
            if (is_min and scores[best_idx] < best_score) or (not is_min and scores[best_idx] > best_score):
                best_sol = pop[best_idx].copy()
                best_score = scores[best_idx]

            yield {
                'iteration': iteration + 1,
                'current_solution': list(best_sol),
                'current_score': best_score,
                'best_solution': list(best_sol),
                'best_score': best_score
            }

            if problem.is_goal(best_sol):
                break

        return best_sol, best_score"""