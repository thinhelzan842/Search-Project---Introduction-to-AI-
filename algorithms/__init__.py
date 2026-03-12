from .classic import HillClimbing, TabuSearch
from .physic_based.simulated_annealing import SimulatedAnnealing
from .evolutionary import GeneticAlgorithm, DifferentialEvolution
from .biology import PSO, ACO, CuckooOptimization, ArtificialBeeColony, FireflyAlgorithm
__all__ = [
    'HillClimbing',
    'TabuSearch',
    'SimulatedAnnealing',
    'GeneticAlgorithm',
"""    'GeneticAlgorithm_Discrete',
    'GeneticAlgorithm_Continuous',"""
    'DifferentialEvolution',
    'PSO',
    'ACO',
    'CuckooOptimization',
    'ArtificialBeeColony',
    'FireflyAlgorithm'
]
