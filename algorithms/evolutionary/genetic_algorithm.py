import numpy as np
from core import *

class GeneticAlgorithm(AlgorithmBase):
    def __init__(self, size, gen, des, enc, xor, mut, sel):
        self.size = size
        self.gen = gen
        self.pop = []
        self.des = des
        self.enc = enc
        self.xor = xor
        self.mut = mut
        self.sel = sel

    def name(self) -> str:
        return "Genetic Algorithm"

    """
    def random_initialize(self, problem):
        for _ in range(self.size):
            self.pop.append(problem.random_solution_generate())
    """

    def run(self, problem):
        #initialize population
        for _ in range(self.size):
            self.pop.append(problem.random_solution_generate())

        """
        if problem.is_discrete():
            return GeneticAlgorithm_Discrete()
        else:
            return GeneticAlgorithm_Continuous()
        """

        scr_list = [problem.evaluate(v) for v in self.pop]
        best_scr = float('inf')
        best_sol = self.pop[0]
        for i,v in scr_list:
            if v < best_scr:
                best_scr = v
                best_sol = self.pop[i]


        diff = float('inf') #difference over generations
        gen = 0
        while best_scr > desire and gen < self.gen and diff > 0.5:
            scr_list = [problem.evaluate(v) for v in self.pop]
            for i, v in scr_list:
                if v < best_scr:
                    best_scr = v
                    best_sol = self.pop[i]

            #tournament selection

class GeneticAlgorithm_Discrete(GeneticAlgorithm):
    def __init__(self):
        pass

    def run(self, problem): #edit here
        s = self.super()
        pass

class GeneticAlgorithm_Continuous(GeneticAlgorithm):
    def __init__(self):
        pass

    def run(self, problem): #edit here
        pass