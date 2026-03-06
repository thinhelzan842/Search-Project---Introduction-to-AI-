import numpy as np
from algorithms.classic.problem import Problem
from algorithms.classic.neighborhood import NeighborhoodStrategy

class HillClimbing:
    def __init__(self, problem: Problem, neighborhood: NeighborhoodStrategy, max_iters: int = 1000):
        self.problem = problem
        self.neighborhood = neighborhood
        self.max_iters = max_iters

    def run(self):
        curr_state = self.problem.random_state()
        curr_energy = self.problem.evaluate(curr_state)
        history = [curr_energy]

        for _ in range(self.max_iters):
            neighbors = self.neighborhood.get_neighbors(curr_state)
            best_n_state, best_n_energy = None, float('inf')

            for n_state in neighbors:
                n_state = self.problem.clip(n_state)
                e = self.problem.evaluate(n_state)
                if e < best_n_energy:
                    best_n_energy, best_n_state = e, n_state

            if best_n_energy < curr_energy:
                curr_state, curr_energy = best_n_state, best_n_energy
            else:
                break 

            history.append(curr_energy)

        return curr_state, curr_energy, history