from algorithms import *
from problems import *

from utils.evaluator import BenchmarkEngine
from utils.visualization import plot_scalability, plot_heatmap, plot_fitness_degradation
import os
import numpy as np

def main():
    # -----------------------------------------------------------------------
    # Algorithm instances - Reduce max_iters/gen/max_epochs to 50 to save some times
    # -----------------------------------------------------------------------
    hb      = HillClimbing(max_iters=50, step_size=0.5, num_neighbors=15)
    ts      = TabuSearch(max_iters=50, tabu_tenure=10, num_neighbors=20, step_size=0.5)
    sa      = SimulatedAnnealing(max_epochs=50, initial_temp=100.0, cooling_rate=0.99,
                                 step_size=0.5, markov_chain_length=20)
    gsa     = GravitationalSearchAlgorithm(pop_size=30, max_iters=50, G0=100.0, alpha=20.0)
    hs      = HarmonySearch(max_iters=50, hmcr=0.9, par=0.3, bw=0.1)
    pso     = PSO(num_particles=20, max_iters=50)
    aco     = ACO(num_ants=20, max_iters=50)
    co      = CuckooOptimization(num_nests=20, max_iters=50)
    tlbo    = TLBO(pop_size=20, max_iters=50)
    ff      = FireflyAlgorithm(popsize=20, gen=50)
    abc     = ArtificialBeeColony(popsize=20, gen=50)
    de      = DifferentialEvolution(popsize=20, gen=50)
    ga_cont = GeneticAlgorithm(size=20, gen=50,
                               crossover_type='multi_point', mutation_type='gaussian')
    ga_disc = GeneticAlgorithm(size=20, gen=50,
                               crossover_type='multi_point', mutation_type='bit_flip')

    bfs    = BFS()
    dfs    = DFS()
    ucs    = UCS()
    greedy = GreedyBFS()
    astar  = AStar()

    # -----------------------------------------------------------------------
    # 1. CONTINUOUS BENCHMARK (num_runs = 30)
    # -----------------------------------------------------------------------
    continuous_algorithms = [hb, tlbo, sa, gsa, hs, ts, pso, co, ff, abc, de, ga_cont]
    continuous_problems   = [
        Rastrigin(dim=2, bound=5.12), Sphere(dim=2, bound=5.0),
        Ackley(dim=2, bound=5.0), Rosenbrock(dim=2, bound=5.0),
        Griewank(dim=2, bound=5.0)
    ]

    print("=" * 60)
    print("BENCHMARK 1 — CONTINUOUS PROBLEMS")
    print("=" * 60)
    engine_cont = BenchmarkEngine(algorithms=continuous_algorithms,
                                  problems=continuous_problems, num_runs=30)
    engine_cont.run_all()
    engine_cont.generate_reports(prefix="continuous")

    # -----------------------------------------------------------------------
    # 2. DISCRETE BENCHMARK
    # -----------------------------------------------------------------------
    discrete_algorithms = [hb, sa, ts, ga_disc, aco]
    discrete_problems   = [
        TravelingSalesman(size=10, time_limit=2000, cost_limit=2000),
        Knapsack(size=15, limit=40), GraphColoring(size=10)
    ]

    print("\n" + "=" * 60)
    print("BENCHMARK 2 — DISCRETE PROBLEMS")
    print("=" * 60)
    engine_disc = BenchmarkEngine(algorithms=discrete_algorithms,
                                  problems=discrete_problems, num_runs=30)
    engine_disc.run_all()
    engine_disc.generate_reports(prefix="discrete")

    # -----------------------------------------------------------------------
    # 3. GRAPH-SEARCH BENCHMARK
    # -----------------------------------------------------------------------
    graph_algorithms = [bfs, dfs, ucs, greedy, astar, aco]
    sp_problem       = ShortestPath(size=12)

    print("\n" + "=" * 60)
    print("BENCHMARK 3 — GRAPH SEARCH vs METAHEURISTICS")
    print("=" * 60)
    engine_graph = BenchmarkEngine(algorithms=graph_algorithms,
                                   problems=[sp_problem], num_runs=30)
    engine_graph.run_all()
    engine_graph.generate_reports(prefix="graph_search")

    # -----------------------------------------------------------------------
    # 4. SCALABILITY & COMPUTATIONAL COMPLEXITY (Time & Space)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BENCHMARK 4 — SCALABILITY & COMPLEXITY (Varying Problem Size)")
    print("=" * 60)
    
    dimensions = [2, 10, 30, 50, 100]
    times, spaces, fitnesses = [], [], []
    
    for dim in dimensions:
        prob_scale = Sphere(dim=dim, bound=5.0) 
        algo_scale = PSO(num_particles=30, max_iters=50) 
        
        engine_scale = BenchmarkEngine(algorithms=[algo_scale], problems=[prob_scale], num_runs=5) 
        engine_scale.run_all()
        
        avg_time = np.mean([r['time'] for r in engine_scale.results])
        avg_space = np.mean([r['space_peak_kb'] for r in engine_scale.results])
        avg_score = np.mean([r['best_score'] for r in engine_scale.results])
        
        times.append(avg_time)
        spaces.append(avg_space)
        fitnesses.append(avg_score)
        
    plot_scalability(dimensions, times, spaces, fitnesses, algo_name="PSO on Sphere", filename="scalability_pso.html")
    print(">> Scalability report generated: results/scalability_pso.html")

    # -----------------------------------------------------------------------
    # 5. CURSE OF DIMENSIONALITY (Sự suy thoái do số chiều cao trên Rastrigin)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BENCHMARK 5 — CURSE OF DIMENSIONALITY (RASTRIGIN)")
    print("=" * 60)
    
    dims_deg = [2, 10, 30, 50, 100]
    deg_results = {'PSO': [], 'Genetic Algorithm': [], 'Simulated Annealing': []}
    
    for dim in dims_deg:
        prob = Rastrigin(dim=dim, bound=5.12)
        algos = [PSO(num_particles=30, max_iters=50), 
                 GeneticAlgorithm(size=30, gen=50, crossover_type='multi_point', mutation_type='gaussian'),
                 SimulatedAnnealing(max_epochs=50, initial_temp=100.0, cooling_rate=0.95, step_size=0.5, markov_chain_length=10)]
        
        engine_deg = BenchmarkEngine(algorithms=algos, problems=[prob], num_runs=3)
        engine_deg.run_all()
        
        for key in deg_results.keys():
            scores = [r['best_score'] for r in engine_deg.results if r['algo'] == key]
            deg_results[key].append(np.mean(scores) if scores else float('inf'))
            
    plot_fitness_degradation(dims_deg, deg_results, title="Fitness Degradation on Rastrigin", filename="curse_of_dimensionality.html")
    print(">> Curse of dimensionality report generated: results/curse_of_dimensionality.html")

    # -----------------------------------------------------------------------
    # 6. PARAMETER SENSITIVITY ANALYSIS (Grid Search on GA)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BENCHMARK 6 — GA PARAMETER SENSITIVITY")
    print("=" * 60)
    
    mut_probs = [0.01, 0.05, 0.1, 0.3]
    cross_probs = [0.6, 0.7, 0.8, 0.9]
    sensitivity_matrix = np.zeros((len(mut_probs), len(cross_probs)))
    prob_sens = Griewank(dim=5, bound=5.0)
    
    for i, mut in enumerate(mut_probs):
        for j, cross in enumerate(cross_probs):
            print(f"  >> Testing GA with Mut={mut}, Cross={cross}...")
            ga_sens = GeneticAlgorithm(size=20, gen=50, crossover_prob=cross, mutate_prob=mut,
                                       crossover_type='multi_point', mutation_type='gaussian')
            engine_sens = BenchmarkEngine(algorithms=[ga_sens], problems=[prob_sens], num_runs=3)
            engine_sens.run_all()
            
            avg_score = np.mean([r['best_score'] for r in engine_sens.results])
            sensitivity_matrix[i, j] = avg_score
            
    plot_heatmap(sensitivity_matrix, 
                 algos=[f"MutRate={m}" for m in mut_probs], 
                 problems=[f"CrossRate={c}" for c in cross_probs], 
                 title="GA Parameter Sensitivity Analysis (Best Fitness)", 
                 filename="ga_parameter_sensitivity.html")
    print(">> GA Parameter sensitivity report generated: results/ga_parameter_sensitivity.html")

    # -----------------------------------------------------------------------
    # 7. PARAMETER SENSITIVITY ANALYSIS (Grid Search on SA)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BENCHMARK 7 — SA PARAMETER SENSITIVITY")
    print("=" * 60)
    
    cooling_rates = [0.8, 0.9, 0.95, 0.99]
    step_sizes = [0.1, 0.5, 1.0, 2.0]
    sa_sens_matrix = np.zeros((len(cooling_rates), len(step_sizes)))
    prob_sens_sa = Ackley(dim=5, bound=5.0)
    
    for i, cool in enumerate(cooling_rates):
        for j, step in enumerate(step_sizes):
            print(f"  >> Testing SA with CoolRate={cool}, Step={step}...")
            sa_sens = SimulatedAnnealing(max_epochs=50, initial_temp=100.0, cooling_rate=cool, step_size=step, markov_chain_length=10)
            engine_sens_sa = BenchmarkEngine(algorithms=[sa_sens], problems=[prob_sens_sa], num_runs=3)
            engine_sens_sa.run_all()
            sa_sens_matrix[i, j] = np.mean([r['best_score'] for r in engine_sens_sa.results])
            
    plot_heatmap(sa_sens_matrix, algos=[f"Cool={c}" for c in cooling_rates], problems=[f"Step={s}" for s in step_sizes], 
                 title="SA Parameter Sensitivity Analysis (Best Fitness)", filename="sa_parameter_sensitivity.html")
    print(">> SA Parameter sensitivity report generated: results/sa_parameter_sensitivity.html")

    print("\n[Done] All amazing HTML reports saved to the results/ directory.")

if __name__ == "__main__":
    main()