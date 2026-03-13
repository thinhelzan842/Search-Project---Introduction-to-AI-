import time
import numpy as np
import pandas as pd
import pickle
import tracemalloc
from core import ContinuousProblem  # Imported for type checking
from utils.visualization import plot_convergence, plot_performance_bar, plot_3d_landscape, plot_heatmap, plot_animation_2d

# --- THÊM CLASS NÀY VÀO ---
class TrackedProblem:
    """Vỏ bọc theo dõi mọi nghiệm được thuật toán đánh giá"""
    def __init__(self, problem):
        self._problem = problem
        self.current_epoch_solutions = []

    def evaluate(self, solution):
        # Ghi chú lại nghiệm đang được đánh giá
        self.current_epoch_solutions.append(solution.copy() if isinstance(solution, (list, np.ndarray)) else solution)
        return self._problem.evaluate(solution)

    def pop_epoch_solutions(self):
        # Lấy danh sách nghiệm đã đánh giá và reset cho epoch tiếp theo
        sols = self.current_epoch_solutions.copy()
        self.current_epoch_solutions = []
        return sols

    def __getattr__(self, attr):
        # Chuyển tiếp các hàm khác (name, get_bounds...) về cho problem gốc
        return getattr(self._problem, attr)
    
# --------------------------
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
                    tracemalloc.start()  # Bắt đầu theo dõi RAM
                    start_time = time.time()
                    
                    history = []
                    trajectory = []
                    diversity_history = []
                    best_score_final = None
                    best_sol_final = None
                    
                    for state in algo.run(problem):
                        score = state['best_score']
                        clean_score = score if score != float('inf') else 1e6 
                        history.append(clean_score)
                        trajectory.append(state['best_solution'])
                        
                        div = state.get('diversity', 0)
                        diversity_history.append(div)
                        
                        best_score_final = clean_score
                        best_sol_final = state['best_solution']
                        
                    exec_time = time.time() - start_time
                    current_mem, peak_mem = tracemalloc.get_traced_memory()  # Chốt mức RAM đỉnh điểm
                    tracemalloc.stop()  # Dừng theo dõi
                    
                    # Thêm 'peak_memory' vào kết quả lưu trữ
                    self.results.append({
                        'algo': algo.name(), 'problem': problem.name(),
                        'best_score': best_score_final, 'history': history,
                        'trajectory': trajectory, 'diversity': diversity_history, 
                        'time': exec_time, 'peak_memory': peak_mem / 1024.0  # Lưu theo đơn vị KB
                    })
                    print(f"    - Num {run_idx+1}/{self.num_runs} | Loss: {best_score_final:.4f} | Time: {exec_time:.3f}s | RAM: {peak_mem / 1024.0:.1f}KB")
                    
    def save_results(self, filepath):
        """Lưu toàn bộ kết quả đánh giá ra file .pkl"""
        import os
        import pickle
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.results, f)
        print(f"    [+] Đã lưu dữ liệu thô (.pkl) tại: {filepath}")

    def get_best_run(self, algo_name, problem_name):
        runs = [r for r in self.results if r['algo'] == algo_name and r['problem'] == problem_name]
        if not runs: return None
        return min(runs, key=lambda r: r['best_score'])

    def generate_reports(self, prefix="all"):
        for problem in self.problems:
            histories, scores, trajectories = {}, {}, {}
            for algo in self.algorithms:
                best_run = self.get_best_run(algo.name(), problem.name())
                if best_run:
                    histories[algo.name()] = best_run['history']
                    scores[algo.name()] = best_run['best_score']
                    trajectories[algo.name()] = best_run['trajectory']

            safe_name = problem.name().replace(" ", "_").lower()
            plot_convergence(histories, title=f"Convergence - {problem.name()}", filename=f"convergence_{safe_name}.html")
            plot_performance_bar(scores, title=f"Best Fitness - {problem.name()}", filename=f"bar_{safe_name}.html")
            
            # Updated check for Continuous 2D problems
            if isinstance(problem, ContinuousProblem) and hasattr(problem, 'dim') and problem.dim == 2:
                plot_3d_landscape(problem, trajectories, filename=f"3d_{safe_name}.html")

        algo_names = [a.name() for a in self.algorithms]
        prob_names = [p.name() for p in self.problems]
        df = pd.DataFrame(index=algo_names, columns=prob_names, dtype=float)
        
        for p_name in prob_names:
            for a_name in algo_names:
                best_run = self.get_best_run(a_name, p_name)
                df.at[a_name, p_name] = best_run['best_score'] if best_run else float('nan')
                
        plot_heatmap(np.log1p(df).values, algo_names, prob_names, title=f"Heatmap (Log Scale) - {prefix.upper()}", filename=f"heatmap_{prefix}.html")

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