from algorithms.classic.hill_climbing import HillClimbing
from algorithms.classic.tabu_search import TabuSearch
from algorithms.physic_based.simulated_annealing import SimulatedAnnealing
from algorithms.biology import PSO, ACO, CuckooOptimization

from problems import *

from utils.evaluator import BenchmarkEngine

def main():
    ls = HillClimbing(max_iters=300, step_size=0.5)
    sa = SimulatedAnnealing(max_epochs=300, initial_temp=100.0, cooling_rate=0.95, step_size=0.5)
    ts = TabuSearch(max_iters=300, tabu_tenure=10, num_neighbors=10, step_size=0.5)
    pso = PSO(num_particles=30, max_iters=300)
    aco = ACO(num_ants=30, max_iters=300)
    co = CuckooOptimization(num_nests=30, max_iters=300)
    
    # Algorithms for continuous problems
    continuous_algorithms = [ls, sa, ts, pso, aco, co]
    
    # Algorithms for discrete problems
    discrete_algorithms = [ls, sa, ts]

    # Continuous problems
    continuous_problems = [
        Rastrigin(dim=2, bound=5.12),
        Sphere(dim=2, bound=5.0),
        Ackley(dim=2, bound=5.0),
    ]
    
    # Discrete problems
    discrete_problems = [
        TravelingSalesman(size=10, time_limit=2000, cost_limit=2000),
        Knapsack(size=15, limit=40)
    ]

    # Run continuous benchmark
    print("=" * 50)
    print("CONTINUOUS PROBLEMS BENCHMARK")
    print("=" * 50)
    engine_continuous = BenchmarkEngine(algorithms=continuous_algorithms, problems=continuous_problems, num_runs=1)
    engine_continuous.run_all()
    
    # Run discrete benchmark
    print("\n" + "=" * 50)
    print("DISCRETE PROBLEMS BENCHMARK")
    print("=" * 50)
    engine_discrete = BenchmarkEngine(algorithms=discrete_algorithms, problems=discrete_problems, num_runs=1)
    engine_discrete.run_all()
    
    # Generate reports
    engine_continuous.generate_reports()
    engine_discrete.generate_reports()

if __name__ == "__main__":
    main()