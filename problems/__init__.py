from .continuous import Sphere
from .continuous import Ackley
from .continuous import Rastrigin
from .continuous import Rosenbrock
from .continuous import Griewank
from .discrete import ShortestPath
from .discrete import GraphColoring
from .discrete import Knapsack
from .discrete import TravelingSalesman

"""
from .discrete.graph_coloring import GraphColoring
from .continuous.ackley import Ackley
from .continuous.griewank import Griewank
from .discrete.knapsack import Knapsack
from .continuous.rastrigin import Rastrigin
from .continuous.rosenbrock import Rosenbrock
from .discrete.shortest_path import ShortestPath
from .continuous.sphere import Sphere
from .discrete.traveling_salesman import TravelingSalesman
"""
__all__ = ['GraphColoring', 'Ackley', 'Griewank', 'Knapsack', 'Rastrigin', 'Rosenbrock', 'ShortestPath', 'Sphere', 'TravelingSalesman']
