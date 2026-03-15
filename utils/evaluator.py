import time
import tracemalloc
import numpy as np
import scipy.stats as stats
from core import ContinuousProblem
from utils.visualization import (
    plot_convergence_robust, plot_boxplot_performance, plot_exploration_exploitation,
    plot_3d_landscape, plot_pvalue_heatmap, plot_first_hitting_time
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
                    try:
                        tracemalloc.start()
                        start_time = time.time()
                        
                        history, trajectory, diversity_history = [], [], []
                        best_score_final = float('inf')
                        
                        for state in algo.run(problem):
                            score = state['best_score']
                            clean_score = score if score != float('inf') else 1e6 
                            history.append(clean_score)
                            trajectory.append(state['best_solution'])
                            best_score_final = clean_score
                            
                            if 'population_scores' in state:
                                diversity_history.append(np.std(state['population_scores']))
                            
                        exec_time = time.time() - start_time
                        current_mem, peak_mem = tracemalloc.get_traced_memory()
                        tracemalloc.stop()
                        
                        run_result = {
                            'algo': algo.name(), 'problem': problem.name(),
                            'best_score': best_score_final, 'history': history,
                            'trajectory': trajectory, 'time': exec_time,
                            'space_peak_kb': peak_mem / 1024.0
                        }
                        if diversity_history: run_result['diversity_history'] = diversity_history
                        self.results.append(run_result)
                        print(f"    - Num {run_idx+1:02d}/{self.num_runs} | Loss: {best_score_final:.4e} | Time: {exec_time:.3f}s")

                    except Exception as e:
                        if tracemalloc.is_tracing(): tracemalloc.stop()
                        print(f"    [ERROR] Algorithm {algo.name()} failed: {e}")
                        continue

    def generate_reports(self, prefix="all"):
        for problem in self.problems:
            all_histories = {algo.name(): [] for algo in self.algorithms}
            final_scores = {algo.name(): [] for algo in self.algorithms}
            trajectories, all_diversities = {}, {algo.name(): [] for algo in self.algorithms}
            
            for run in self.results:
                if run['problem'] == problem.name():
                    algo_name = run['algo']
                    all_histories[algo_name].append(run['history'])
                    final_scores[algo_name].append(run['best_score'])
                    
                    if algo_name not in trajectories or run['best_score'] < min(final_scores[algo_name]):
                        trajectories[algo_name] = run['trajectory']
                    if 'diversity_history' in run:
                        all_diversities[algo_name].append(run['diversity_history'])

            safe_name = problem.name().replace(" ", "_").lower()
            
            # 1. Base Plots
            plot_convergence_robust(all_histories, title=f"Convergence - {problem.name()}", filename=f"convergence_{safe_name}.html")
            plot_boxplot_performance(final_scores, title=f"Robustness (Boxplot) - {problem.name()}", filename=f"boxplot_{safe_name}.html")
            
            # 2. Explore / Exploit
            for algo_name, div_runs in all_diversities.items():
                if len(div_runs) > 0:
                    min_len = min(len(r) for r in div_runs)
                    avg_div_history = np.mean([r[:min_len] for r in div_runs], axis=0)
                    plot_exploration_exploitation(avg_div_history, title=f"{algo_name}: Explore vs Exploit - {problem.name()}", filename=f"ee_{algo_name.replace(' ', '_').lower()}_{safe_name}.html")

            # 3. 3D Landscape (Dành cho bài toán liên tục 2 chiều)
            if isinstance(problem, ContinuousProblem) and hasattr(problem, 'dim') and problem.dim == 2:
                plot_3d_landscape(problem, trajectories, filename=f"3d_landscape_{safe_name}.html")
            
            # 4. First Hitting Time (Convergence Speed)
            valid_scores = [s for scores in final_scores.values() for s in scores if s != 1e6]
            if valid_scores:
                best_overall = min(valid_scores)
                threshold = best_overall + 0.05 * abs(best_overall) if best_overall != 0 else 1e-3
                
                # Lấy độ dài history lớn nhất làm hình phạt cho các thuật toán không chạm ngưỡng
                max_len = max([len(h) for histories in all_histories.values() for h in histories] + [0])
                
                hitting_times = {}
                for algo_name, histories in all_histories.items():
                    if not histories: continue
                    # Thay len(h) bằng max_len
                    times = [next((i for i, val in enumerate(h) if val <= threshold), max_len) for h in histories]
                    hitting_times[algo_name] = np.mean(times)

            # 5. Statistical Significance Matrix (P-Value Heatmap)
            algos_list = [a for a in final_scores.keys() if final_scores[a]]
            n_algos = len(algos_list)
            if n_algos > 1:
                p_matrix = np.ones((n_algos, n_algos))
                for i in range(n_algos):
                    for j in range(n_algos):
                        if i != j:
                            try:
                                stat, p_val = stats.ranksums(final_scores[algos_list[i]], final_scores[algos_list[j]])
                                p_matrix[i, j] = p_val
                            except: pass
                plot_pvalue_heatmap(p_matrix, algos_list, title=f"Significance P-Value Matrix - {problem.name()}", filename=f"pvalue_matrix_{safe_name}.html")