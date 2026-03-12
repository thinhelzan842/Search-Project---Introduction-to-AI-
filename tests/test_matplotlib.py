import os 
import sys
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from algorithms import HillClimbing , SimulatedAnnealing, TabuSearch
from problems import *

def run_algo_and_get_history(algo, problem):
    history = []
    print(f"Đang chạy {algo.name()} trên {problem.name()}...")
    
    for state in algo.run(problem):
        score = state['best_score']
        if score != float('inf'):
            history.append(score)
            
    return history

def plot_problem_comparison(problem, algorithms):
    plt.figure(figsize=(10, 6))
    
    for algo in algorithms:
        history = run_algo_and_get_history(algo, problem)
        plt.plot(history, label=algo.name())

    plt.title(f"{problem.name()} Problem Optimization")
    plt.xlabel("Iterations")
    plt.ylabel("Energy (Loss)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    ls = HillClimbing(max_iters=1300, step_size=0.5, num_neighbors=15)
    sa = SimulatedAnnealing(max_epochs=1000, initial_temp=100.0, cooling_rate=0.99, step_size=0.5, markov_chain_length=20)
    ts = TabuSearch(max_iters=150, tabu_tenure=10, num_neighbors=10, step_size=0.5)
    
    algos_to_test = [ls, sa, ts]

    # TEST 1: Ackley
    ackley_prob = Ackley(dim=10, bound=5.0)
    plot_problem_comparison(ackley_prob, algos_to_test)


    # TEST 2: (TSP)

    tsp_prob = TravelingSalesman(size=10, time_limit=2000, cost_limit=2000)
    plot_problem_comparison(tsp_prob, algos_to_test)