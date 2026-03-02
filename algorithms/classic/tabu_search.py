import numpy as np
from core.problem import Problem
from core.neighborhood import NeighborhoodStrategy
from utils.tabu_memory import TabuList

class TabuSearch:
    def __init__(self, problem: Problem, neighborhood: NeighborhoodStrategy, 
                 tabu_size: int = 50, max_iters: int = 500):
        self.problem = problem
        self.neighborhood = neighborhood
        self.max_iters = max_iters
        self.tabu_list = TabuList(max_size=tabu_size)

    def run(self):
        curr_state = self.problem.random_state()
        curr_energy = self.problem.evaluate(curr_state)
        
        best_state, best_energy = np.copy(curr_state), curr_energy
        history = [best_energy]

        for _ in range(self.max_iters):
            neighbors = self.neighborhood.get_neighbors(curr_state)
            best_c_state, best_c_energy = None, float('inf')

            for n_state in neighbors:
                n_state = self.problem.clip(n_state)
                e = self.problem.evaluate(n_state)
                
                is_aspirated = (e < best_energy)
                is_tabu = self.tabu_list.is_tabu(n_state)

                if (not is_tabu or is_aspirated) and (e < best_c_energy):
                    best_c_energy, best_c_state = e, n_state

            if best_c_state is None:
                break 

            curr_state, curr_energy = best_c_state, best_c_energy
            self.tabu_list.add(curr_state)

            if curr_energy < best_energy:
                best_state, best_energy = np.copy(curr_state), curr_energy

            history.append(best_energy)

        return best_state, best_energy, history