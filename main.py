from algorithms import *
from problems import *

from utils.evaluator import BenchmarkEngine

import os

def main():
    # -----------------------------------------------------------------------
    # Algorithm instances
    # -----------------------------------------------------------------------
    hb      = HillClimbing(max_iters=600, step_size=0.5, num_neighbors=15)
    ts      = TabuSearch(max_iters=600, tabu_tenure=10, num_neighbors=20, step_size=0.5)
    sa      = SimulatedAnnealing(max_epochs=600, initial_temp=100.0, cooling_rate=0.99,
                                 step_size=0.5, markov_chain_length=20)
    gsa = GravitationalSearchAlgorithm(pop_size=30, max_iters=600, G0=100.0, alpha=20.0)
    hs  = HarmonySearch(max_iters=600, hmcr=0.9, par=0.3, bw=0.1)
    pso     = PSO(num_particles=20, max_iters=600)
    aco     = ACO(num_ants=20, max_iters=600)
    co      = CuckooOptimization(num_nests=20, max_iters=600)
    tlbo    = TLBO(pop_size=20, max_iters=600)
    ff      = FireflyAlgorithm(popsize=20, gen=600)
    abc     = ArtificialBeeColony(popsize=20, gen=600)
    de      = DifferentialEvolution(popsize=20, gen=600)
    ga_cont = GeneticAlgorithm(size=20, gen=600, desire=0.001,
                               crossover_type='multi_point', mutation_type='gaussian')
    ga_disc = GeneticAlgorithm(size=20, gen=600, desire=None,
                               crossover_type='multi_point', mutation_type='bit_flip')

    # Graph-search algorithms (only valid on graph/path problems)
    bfs    = BFS()
    dfs    = DFS()
    ucs    = UCS()
    greedy = GreedyBFS()
    astar  = AStar()

    # -----------------------------------------------------------------------
    # 1. CONTINUOUS BENCHMARK
    # -----------------------------------------------------------------------
    continuous_algorithms = [hb, tlbo, sa, gsa, hs, ts, pso, aco, co, ff, abc, de, ga_cont]
    continuous_problems   = [
        #Rastrigin(dim=2, bound=5.12)
        #Sphere(dim=2, bound=5.0),
        #Ackley(dim=2, bound=5.0),
        #Rosenbrock(dim=2, bound=5.0),
        Griewank(dim=2, bound=5.0)
    ]

    print("=" * 60)
    print("BENCHMARK 1 — CONTINUOUS PROBLEMS")
    print("=" * 60)
    engine_cont = BenchmarkEngine(algorithms=continuous_algorithms,
                                  problems=continuous_problems, num_runs=5)
    engine_cont.run_all()
    engine_cont.generate_reports(prefix="continuous")
    engine_cont.generate_animations()

    # -----------------------------------------------------------------------
    # 2. DISCRETE BENCHMARK  (metaheuristics on combinatorial problems)
    # -----------------------------------------------------------------------
    discrete_algorithms = [hb, sa, ts, ga_disc]
    discrete_problems   = [
        #TravelingSalesman(size=10, time_limit=2000, cost_limit=2000),
        #Knapsack(size=15, limit=40),
        GraphColoring(size=10)
    ]

    print("\n" + "=" * 60)
    print("BENCHMARK 2 — DISCRETE PROBLEMS (metaheuristics)")
    print("=" * 60)
    engine_disc = BenchmarkEngine(algorithms=discrete_algorithms,
                                  problems=discrete_problems, num_runs=5)
    engine_disc.run_all()
    engine_disc.generate_reports(prefix="discrete")
    #engine_cont.generate_animations()

    # -----------------------------------------------------------------------
    # 3. GRAPH-SEARCH BENCHMARK
    #    Classical search (BFS / DFS / UCS / Greedy / A*) vs metaheuristics
    #    (Hill Climbing, SA, Tabu Search) on the Shortest Path problem.
    #    This is the head-to-head comparison required by the PDF spec.
    # -----------------------------------------------------------------------
    graph_algorithms = [bfs, dfs, ucs, greedy, astar, hb, sa, ts]
    sp_problem       = ShortestPath(size=12)

    print("\n" + "=" * 60)
    print("BENCHMARK 3 — GRAPH SEARCH vs METAHEURISTICS (Shortest Path)")
    print("=" * 60)
    engine_graph = BenchmarkEngine(algorithms=graph_algorithms,
                                   problems=[sp_problem], num_runs=5)
    engine_graph.run_all()
    engine_graph.generate_reports(prefix="graph_search")
   # engine_cont.generate_animations()

    print("\n[Done] All HTML reports saved to the results/ directory.")

if __name__ == "__main__":
    main()