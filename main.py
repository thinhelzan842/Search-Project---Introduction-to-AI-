from algorithms.classic import HillClimbing, TabuSearch
from algorithms.physic_based.simulated_annealing import SimulatedAnnealing
from algorithms.biology import PSO, ACO, CuckooOptimization
from algorithms.evolutionary import GeneticAlgorithm
from algorithms.human_based.TLBO import TLBO

from problems import *

from utils.evaluator import BenchmarkEngine

def main():
    hb = HillClimbing(max_iters=1300, step_size=0.5, num_neighbors=15)
    sa = SimulatedAnnealing(max_epochs=1000, initial_temp=100.0, cooling_rate=0.99, step_size=0.5, markov_chain_length=20)
    ts = TabuSearch(max_iters=300, tabu_tenure=10, num_neighbors=10, step_size=0.5)
    pso = PSO(num_particles=30, max_iters=300)
    aco = ACO(num_ants=30, max_iters=300)
    co = CuckooOptimization(num_nests=30, max_iters=300)
    tlbo = TLBO(pop_size=20, max_iters=300)
    
    # Algorithms for continuous problems
    continuous_algorithms = [hb, tlbo, sa, ts, pso, aco, co]
    
    # Algorithms for discrete problems
    discrete_algorithms = [hb, sa, ts]

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
    engine_continuous.generate_reports(prefix="continuous")
    engine_discrete.generate_reports(prefix="discrete")

if __name__ == "__main__":
    main()