import time
import tracemalloc
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
import scipy.stats as stats

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
            true_optimum = problem.get_optimal_value()
            print(f"  [Info] True Optimal Value for {problem.name()}: {true_optimum}")
            
            for algo in self.algorithms:
                print(f"  >> Running: [ {algo.name()} ]...")
                
                for run_idx in range(self.num_runs):
                    try:
                        # Bắt đầu đo lường Space Complexity
                        tracemalloc.start()
                        start_time = time.time()
                        
                        history = []
                        trajectory = []
                        best_score_final = None
                        diversity_history = []
                        
                        # Chạy thuật toán
                        for state in algo.run(problem):
                            score = state['best_score']
                            clean_score = score if score != float('inf') else 1e6 
                            history.append(clean_score)
                            trajectory.append(state['best_solution'])
                            best_score_final = clean_score
                            
                            if 'population_scores' in state:
                                diversity = np.std(state['population_scores'])
                                diversity_history.append(diversity)
                            
                        exec_time = time.time() - start_time
                        current_mem, peak_mem = tracemalloc.get_traced_memory()
                        tracemalloc.stop()
                        peak_mem_kb = peak_mem / 1024.0

                        # ---> NEW: Calculate Optimality Gap
                        gap = abs(best_score_final - true_optimum)
                        if best_score_final == float('inf'):
                            gap = float('inf')  # Keep infinity if no solution found

                        run_result = {
                            'algo': algo.name(), 'problem': problem.name(),
                            'best_score': best_score_final,
                            'gap': gap,  # ---> NEW: Store gap
                            'history': history,
                            'trajectory': trajectory, 'time': exec_time,
                            'space_peak_kb': peak_mem_kb
                        }

                        if diversity_history:
                            run_result['diversity_history'] = diversity_history
                        self.results.append(run_result)
                        
                        print(f"    - Num {run_idx+1:02d}/{self.num_runs} | Loss: {best_score_final:.4e} | Time: {exec_time:.3f}s")

                    except Exception as e:
                        # Nếu lỗi, dừng đo bộ nhớ, in thông báo và tiếp tục run/algo khác
                        if tracemalloc.is_tracing():
                            tracemalloc.stop()
                        print(f"    [ERROR] Algorithm {algo.name()} failed on {problem.name()}: {e}")
                        continue

    def get_best_run(self, algo_name, problem_name):
        runs = [r for r in self.results if r['algo'] == algo_name and r['problem'] == problem_name]
        if not runs: return None
        return min(runs, key=lambda r: r['best_score'])

    def generate_reports(self, prefix="all"):
        for problem in self.problems:
            all_histories = {algo.name(): [] for algo in self.algorithms}
            final_scores = {algo.name(): [] for algo in self.algorithms}
            final_gaps = {algo.name(): [] for algo in self.algorithms}
            trajectories = {}
            all_diversities = {algo.name(): [] for algo in self.algorithms}
            
            # Gom dữ liệu từ TẤT CẢ các lần chạy
            for run in self.results:
                if run['problem'] == problem.name():
                    algo_name = run['algo']
                    all_histories[algo_name].append(run['history'])
                    final_gaps[algo_name].append(run['gap'])
                    final_scores[algo_name].append(run['best_score'])
                    
                    # Lấy trajectory của lần chạy tốt nhất để vẽ 3D
                    if algo_name not in trajectories or run['best_score'] < min(final_scores[algo_name]):
                        trajectories[algo_name] = run['trajectory']
                        
                    # Thu thập toàn bộ diversity history để tính trung bình
                    if 'diversity_history' in run:
                        all_diversities[algo_name].append(run['diversity_history'])

            safe_name = problem.name().replace(" ", "_").lower()
            
            # Gọi các hàm vẽ mới
            plot_convergence_robust(all_histories, title=f"Convergence - {problem.name()}", filename=f"convergence_{safe_name}.html")
            plot_boxplot_performance(final_gaps, title=f"Optimality Gap (Boxplot) - {problem.name()}",
                                     filename=f"boxplot_gap_{safe_name}.html")

            # ---> NEW: Call the dedicated gap visualization you already have in visualization.py
            from utils.visualization import plot_solution_quality
            plot_solution_quality(final_gaps, title=f"Solution Quality (Optimality Gap) - {problem.name()}",
                                  filename=f"quality_{safe_name}.html")
            
            # Vẽ Exploration/Exploitation bằng cách tính TRUNG BÌNH các lần chạy
            # Fix lỗi tính Diversity trung bình khi các lần chạy có độ dài khác nhau
            for algo_name, div_runs in all_diversities.items():
                if len(div_runs) > 0:
                    # Truncate về độ dài ngắn nhất để đảm bảo tính Explore/Exploit là thực tế cho đến khi kết thúc
                    min_len = min(len(r) for r in div_runs)
                    truncated_runs = [r[:min_len] for r in div_runs]
                    avg_div_history = np.mean(truncated_runs, axis=0)
                    
                    safe_algo = algo_name.replace(" ", "_").lower()
                    plot_exploration_exploitation(avg_div_history, title=f"{algo_name}: Explore vs Exploit - {problem.name()}", 
                                                   filename=f"ee_{safe_algo}_{safe_name}.html")
            
            print(f"\n[STATISTICS] Hypothesis Testing for {problem.name()}:")
            if final_scores:
                mean_scores = {algo: np.mean(scores) for algo, scores in final_scores.items() if scores}
                if mean_scores:
                    best_algo = min(mean_scores, key=mean_scores.get)
                    print(f"  Best Algorithm (Mean): {best_algo} (Mean = {mean_scores[best_algo]:.4e})")
                    
                    for algo, scores in final_scores.items():
                        if algo != best_algo and scores:
                            stat, p_val = stats.ranksums(final_scores[best_algo], scores)
                            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                            print(f"    vs {algo:<25}: p-value = {p_val:.4e} [{significance}]")
            
    def generate_animations(self):
        for problem in self.problems:
            if isinstance(problem, ContinuousProblem) and hasattr(problem, 'dim') and problem.dim == 2:
                safe_prob = problem.name().replace(" ", "_").lower()
                for algo in self.algorithms:
                    best_run = self.get_best_run(algo.name(), problem.name())
                    if best_run:
                        safe_algo = algo.name().replace(" ", "_").lower()
                        filename = f"anim_{safe_algo}_{safe_prob}.gif"
                        plot_animation_2d(problem, best_run['trajectory'], algo.name(), filename)