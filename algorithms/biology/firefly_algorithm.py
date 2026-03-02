from core import *

class FireflyAlgorithm(AlgorithmBase):
    def __init__(self):
        pass

    def name(self) -> str:
        return "Firefly Algorithm"
    
    def run(self, problem):
        pass

    def move_firefly(self, firefly_i, firefly_j, beta0, gamma):
        # Calculate the distance between firefly_i and firefly_j
        r = np.linalg.norm(firefly_i - firefly_j)
        # Calculate the attractiveness
        beta = beta0 * np.exp(-gamma * r**2)
        # Move firefly_i towards firefly_j
        firefly_i += beta * (firefly_j - firefly_i) + 0.01 * np.random.rand(*firefly_i.shape) 

