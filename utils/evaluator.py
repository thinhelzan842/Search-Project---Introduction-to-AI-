import time
import numpy as np
import pandas as pd
from core import ContinuousProblem  # Imported for type checking
from utils.visualization import (
    plot_convergence_robust, 
    plot_boxplot_performance, 
    plot_exploration_exploitation,
    plot_3d_landscape, 
    plot_heatmap, 
    plot_animation_2d
)

class BenchmarkEngine:
    def __init__(self, algorithms, problems, num_runs=1):
        self.algorithms = algorithms
        self.problems = problems
        self.num_runs = num_runs
        self.results = []

    def run_all(self):
        print(f"BENCHMARK (Algorithm: {len(self.algorithms)}, Problem: {len(self.problems)})")
        
        for problem in self.problems:
            print(f"\n{'='*50}\n Problem: {problem.name().upper()}\n{'='*50}")
            
            for algo in self.algorithms:
                print(f"  >> Running: [ {algo.name()} ]...")
                
                for run_idx in range(self.num_runs):
                    start_time = time.time()
                    
                    history = []
                    trajectory = []
                    best_score_final = None
                    best_sol_final = None
                    
                    for state in algo.run(problem):
                        score = state['best_score']
                        # Safeguard against infinity for plotting purposes
                        clean_score = score if score != float('inf') else 1e6 
                        history.append(clean_score)
                        trajectory.append(state['best_solution'])
                        best_score_final = clean_score
                        best_sol_final = state['best_solution']
                        if 'population_scores' in state:
                            # Đo độ phân tán (std) của điểm số trong quần thể
                            if 'diversity_history' not in locals(): diversity_history = []
                            diversity = np.std(state['population_scores'])
                            diversity_history.append(diversity)
                        
                    exec_time = time.time() - start_time
                    
                    # Update lại dict kết quả
                    run_result = {
                        'algo': algo.name(), 'problem': problem.name(),
                        'best_score': best_score_final, 'history': history,
                        'trajectory': trajectory, 'time': exec_time
                    }
                    if 'diversity_history' in locals():
                        run_result['diversity_history'] = diversity_history
                    self.results.append(run_result)
                    
                    print(f"    - Num {run_idx+1}/{self.num_runs} | Loss: {best_score_final:.4f} | Time: {exec_time:.3f}s")

    def get_best_run(self, algo_name, problem_name):
        runs = [r for r in self.results if r['algo'] == algo_name and r['problem'] == problem_name]
        if not runs: return None
        return min(runs, key=lambda r: r['best_score'])

    def generate_reports(self, prefix="all"):
        for problem in self.problems:
            all_histories = {algo.name(): [] for algo in self.algorithms}
            final_scores = {algo.name(): [] for algo in self.algorithms}
            trajectories = {}
            avg_diversities = {algo.name(): [] for algo in self.algorithms}
            
            # Gom dữ liệu từ TẤT CẢ các lần chạy
            for run in self.results:
                if run['problem'] == problem.name():
                    algo_name = run['algo']
                    all_histories[algo_name].append(run['history'])
                    final_scores[algo_name].append(run['best_score'])
                    
                    # Lấy trajectory của lần chạy tốt nhất để vẽ 3D
                    if algo_name not in trajectories or run['best_score'] < min(final_scores[algo_name]):
                        trajectories[algo_name] = run['trajectory']
                        
                    # Lấy diversity history (nếu thuật toán có trả về population_scores)
                    if 'diversity_history' in run:
                        avg_diversities[algo_name] = run['diversity_history']

            safe_name = problem.name().replace(" ", "_").lower()
            
            # Gọi các hàm vẽ mới
            plot_convergence_robust(all_histories, title=f"Convergence - {problem.name()}", filename=f"convergence_{safe_name}.html")
            plot_boxplot_performance(final_scores, title=f"Robustness (Boxplot) - {problem.name()}", filename=f"boxplot_{safe_name}.html")
            
            # Vẽ Exploration/Exploitation cho 1 thuật toán đại diện (VD: ACO)
            if "ACO" in avg_diversities and len(avg_diversities["ACO"]) > 0:
                plot_exploration_exploitation(avg_diversities["ACO"], title=f"ACO: Explore vs Exploit - {problem.name()}", filename=f"ee_aco_{safe_name}.html")

    def generate_animations(self):
        for problem in self.problems:
            # Updated check for Continuous 2D problems
            if isinstance(problem, ContinuousProblem) and hasattr(problem, 'dim') and problem.dim == 2:
                safe_prob = problem.name().replace(" ", "_").lower()
                for algo in self.algorithms:
                    best_run = self.get_best_run(algo.name(), problem.name())
                    if best_run:
                        safe_algo = algo.name().replace(" ", "_").lower()
                        filename = f"anim_{safe_algo}_{safe_prob}.gif"
                        plot_animation_2d(problem, best_run['trajectory'], algo.name(), filename)