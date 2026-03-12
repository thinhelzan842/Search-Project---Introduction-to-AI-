from algorithms import *
from problems import *

from utils.evaluator import BenchmarkEngine

import webbrowser
import os

def main():
    ls = HillClimbing(max_iters=300, step_size=0.5)
    sa = SimulatedAnnealing(max_epochs=300, initial_temp=100.0, cooling_rate=0.95, step_size=0.5)
    ts = TabuSearch(max_iters=300, tabu_tenure=10, num_neighbors=10, step_size=0.5)

    ga_cont = GeneticAlgorithm(size=50, gen=150, desire=0.001, crossover_type='multi_point', mutation_type='gaussian')
    #crossover_type='one_point','multi_point','order'
    #mutation_type='bit_flip','swap','gaussian'
    ga_graph = GeneticAlgorithm(size=50, gen=150, desire=None, crossover_type='order', mutation_type='swap')
    ga_disc = GeneticAlgorithm(size=50, gen=150, desire=None, crossover_type='multi_point', mutation_type='bit_flip')
    de = DifferentialEvolution(popsize=50, gen=150)
    
    algorithms = [ls, sa, ts, de, ga_cont, ga_graph, ga_disc]

    problems = [
        Rastrigin(dim=2, bound=5.12),
        Sphere(dim=2, bound=5.12),
        Ackley(dim=2, bound=5.12),
        Rosenbrock(dim=2, bound=5.12),
        Griewank(dim=2, bound=5.12),
        TravelingSalesman(size=10, time_limit=2000, cost_limit=2000),
        Knapsack(size=15, limit=40),
        ShortestPath(size=10),
        GraphColoring(size=10)
    ]

    engine_cont = BenchmarkEngine(algorithms=algorithms, problems=problems, num_runs=1)
    engine_cont.run_all()
    engine_cont.generate_reports()
    # Path to the results folder
    results_folder = "results"

if __name__ == "__main__":
    main()