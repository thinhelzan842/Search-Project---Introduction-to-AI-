from .classic import HillClimbing, TabuSearch
from .physic_based.simulated_annealing import SimulatedAnnealing
from .evolutionary import GeneticAlgorithm, DifferentialEvolution
from .biology import PSO, ACO, CuckooOptimization, ArtificialBeeColony, FireflyAlgorithm
from .human_based.TLBO import TLBO
__all__ = [
    'HillClimbing',
    'TabuSearch',
    'SimulatedAnnealing',
    'GeneticAlgorithm',
    'DifferentialEvolution',
    'PSO',
    'ACO',
    'CuckooOptimization',
    'ArtificialBeeColony',
    'FireflyAlgorithm',
    "TLBO"
]
