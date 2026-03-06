from .classic.hill_climbing import HillClimbing
from .classic.tabu_search import TabuSearch
from .physic_based.simulated_annealing import SimulatedAnnealing
from .evolutionary.genetic_algorithm import GeneticAlgorithm, GeneticAlgorithm_Discrete, GeneticAlgorithm_Continuous
from .evolutionary.differential_evolution import DifferentialEvolution
__all__ = [
    'LocalSearch',
    'TabuSearch',
    'SimulatedAnnealing',
    'GeneticAlgorithm',
    'GeneticAlgorithm_Discrete',
    'GeneticAlgorithm_Continuous',
    'DifferentialEvolution'
]
#from .evolutionary.genetic_algorithm import GeneticAlgorithm
#from .evolutionary.differential_evolution import DifferentialEvolution
#from .evolutionary.ga import GA
#from .evolutionary import ga
#__all__ = ['GeneticAlgorithm', 'GA', 'DifferentialEvolution', 'ga']