from .classic import HillClimbing, TabuSearch
from .physic_based.simulated_annealing import SimulatedAnnealing
from .evolutionary import GeneticAlgorithm, DifferentialEvolution
from .biology import PSO, ACO, CuckooOptimization
__all__ = [
    'HillClimbing',
    'TabuSearch',
    'SimulatedAnnealing',
    'GeneticAlgorithm',
    'GeneticAlgorithm_Discrete',
    'GeneticAlgorithm_Continuous',
    'DifferentialEvolution',
    'PSO',
    'ACO',
    'CuckooOptimization'
]
#from .evolutionary.genetic_algorithm import GeneticAlgorithm
#from .evolutionary.differential_evolution import DifferentialEvolution
#from .evolutionary.ga import GA
#from .evolutionary import ga
#__all__ = ['GeneticAlgorithm', 'GA', 'DifferentialEvolution', 'ga']