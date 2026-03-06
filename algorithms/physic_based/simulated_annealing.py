import numpy as np
from core.problem import Problem
from core.neighborhood import NeighborhoodStrategy
from core.cooling import CoolingSchedule

class SimulatedAnnealing:
    def __init__(self, problem: Problem, neighborhood: NeighborhoodStrategy, 
                 cooling: CoolingSchedule, t_init: float = 100.0, 
                 t_min: float = 1e-3, l_epochs: int = 20):
        self.problem = problem
        self.neighborhood = neighborhood
        self.cooling = cooling
        self.t_init = t_init
        self.t_min = t_min
        self.l_epochs = l_epochs # Số bước Markov Chain cân bằng nhiệt

    def run(self):
        curr_state = self.problem.random_state()
        curr_energy = self.problem.evaluate(curr_state)
        
        best_state, best_energy = np.copy(curr_state), curr_energy
        T = self.t_init
        history = [best_energy]

        while T > self.t_min:
            for _ in range(self.l_epochs):
                new_state = self.neighborhood.get_neighbors(curr_state, temp=1.0)[0]
                new_state = self.problem.clip(new_state)
                new_energy = self.problem.evaluate(new_state)
                
                delta_e = new_energy - curr_energy
                
                # Metropolis Acceptance Criterion
                if delta_e < 0 or np.random.rand() < np.exp(-delta_e / T):
                    curr_state, curr_energy = new_state, new_energy
                    
                    if curr_energy < best_energy:
                        best_state, best_energy = np.copy(curr_state), curr_energy
                        
            history.append(best_energy)
            T = self.cooling.update(T) # Hạ nhiệt độ
            
        return best_state, best_energy, history