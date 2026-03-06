from algorithms.classic.hill_climbing import HillClimbing
from algorithms.classic.tabu_search import TabuSearch
from algorithms.physic_based.simulated_annealing import SimulatedAnnealing

from problems import *

from utils.evaluator import BenchmarkEngine

def main():
    ls = HillClimbing(max_iters=300, step_size=0.5)
    sa = SimulatedAnnealing(max_epochs=300, initial_temp=100.0, cooling_rate=0.95, step_size=0.5)
    ts = TabuSearch(max_iters=300, tabu_tenure=10, num_neighbors=10, step_size=0.5)
    
    algorithms = [ls, sa, ts]

    problems = [
        # Continuous 
        Rastrigin(dim=2, bound=5.12),
        Sphere(dim=2, bound=5.0),
        Ackley(dim=2, bound=5.0),
        # Discrete
        TravelingSalesman(size=10, time_limit=2000, cost_limit=2000),
        Knapsack(size=15, limit=40)
    ]

    engine = BenchmarkEngine(algorithms=algorithms, problems=problems, num_runs=1)
    engine.run_all()
    engine.generate_reports()

if __name__ == "__main__":
    main()