import numpy as np
import random

from core import *

class FireflyAlgorithm(AlgorithmBase):
    def __init__(self, n_fireflies=20, alpha=0.5, beta0=1.0, gamma=1.0):
        self.n_fireflies = n_fireflies
        self.alpha = alpha  # Randomness factor
        self.beta0 = beta0  # Attractiveness at distance 0
        self.gamma = gamma  # Light absorption coefficient

    def name(self) -> str:
        return "Firefly Algorithm"

    def run(self, problem):
        # Initialize population
        pop = [problem.get_random_solution() for _ in range(self.n_fireflies)]
        
        for _ in range(problem.max_iter):
            for i in range(self.n_fireflies):
                for j in range(self.n_fireflies):
                    # Firefly i moves toward firefly j if j is better
                    if problem.evaluate(pop[j]) < problem.evaluate(pop[i]):
                        dist = np.linalg.norm(pop[i] - pop[j])
                        # Calculate Attractiveness (Beta)
                        beta = self.beta0 * np.exp(-self.gamma * dist**2)
                        
                        # Move firefly i: [current] + [attraction] + [random step]
                        pop[i] += beta * (pop[j] - pop[i]) + self.alpha * (np.random.rand() - 0.5)
                        
                        # Constraints (Keep in bounds)
                        pop[i] = problem.clip(pop[i])
                        
        return min(pop, key=problem.evaluate)