import time
import numpy as np
import pandas as pd
import tracemalloc
from core import ContinuousProblem 
from utils.visualization import plot_convergence, plot_performance_bar, plot_3d_landscape, plot_heatmap, plot_animation_2d

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
                    # Start tracking RAM right before execution
                    tracemalloc.start()  
                    start_time = time.time()
                    
                    history = []
                    trajectory = []
                    best_score_final = None
                    best_sol_final = None
                    
                    # Iterate directly through the standardized yielded states
                    for state in algo.run(problem):
                        score = state['best_score']
                        # Map infinity to a high finite value so Plotly doesn't crash
                        clean_score = score if score != float('inf') else 1e6 
                        
                        history.append(clean_score)
                        trajectory.append(state['best_solution'])
                        
                        best_score_final = clean_score
                        best_sol_final = state['best_solution']
                        
                    exec_time = time.time() - start_time
                    current_mem, peak_mem = tracemalloc.get_traced_memory()  
                    tracemalloc.stop() 
                    
                    self.results.append({
                        'algo': algo.name(), 
                        'problem': problem.name(),
                        'best_score': best_score_final, 
                        'history': history,
                        'trajectory': trajectory, 
                        'time': exec_time, 
                        'peak_memory': peak_mem / 1024.0  # Save in KB
                    })
                    print(f"    - Num {run_idx+1}/{self.num_runs} | Loss: {best_score_final:.4f} | Time: {exec_time:.3f}s | RAM: {peak_mem / 1024.0:.1f}KB")
                    
    def save_results(self, filepath):
        import os
        import pickle
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.results, f)
        print(f"    [+] Saved raw data (.pkl) to: {filepath}")

    def get_best_run(self, algo_name, problem_name):
        runs = [r for r in self.results if r['algo'] == algo_name and r['problem'] == problem_name]
        if not runs: return None
        return min(runs, key=lambda r: r['best_score'])

    def generate_reports(self, prefix="all"):
        for problem in self.problems:
            histories, scores, trajectories = {}, {}, {}
            max_iters = 0
            best_runs = {}
            
            # 1. Find the best run for each algorithm and determine max iterations
            for algo in self.algorithms:
                best_run = self.get_best_run(algo.name(), problem.name())
                if best_run:
                    best_runs[algo.name()] = best_run
                    if len(best_run['history']) > max_iters:
                        max_iters = len(best_run['history'])

            # 2. Pad histories to prevent cut-off graphs in Plotly
            for algo_name, run_data in best_runs.items():
                scores[algo_name] = run_data['best_score']
                trajectories[algo_name] = run_data['trajectory']
                
                hist = run_data['history'].copy()
                if len(hist) < max_iters:
                    # Extend the array by repeating the final known score
                    padding = [hist[-1]] * (max_iters - len(hist))
                    hist.extend(padding)
                
                histories[algo_name] = hist

            safe_name = problem.name().replace(" ", "_").lower()
            plot_convergence(histories, title=f"Convergence - {problem.name()}", filename=f"convergence_{safe_name}.html")
            plot_performance_bar(scores, title=f"Best Fitness - {problem.name()}", filename=f"bar_{safe_name}.html")
            
            # Only plot 3D landscape if the problem is strictly Continuous 2D
            if isinstance(problem, ContinuousProblem) and hasattr(problem, 'dim') and problem.dim == 2:
                plot_3d_landscape(problem, trajectories, filename=f"3d_{safe_name}.html")

        # Generate comparative heatmap
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
            if isinstance(problem, ContinuousProblem) and hasattr(problem, 'dim') and problem.dim == 2:
                safe_prob = problem.name().replace(" ", "_").lower()
                for algo in self.algorithms:
                    best_run = self.get_best_run(algo.name(), problem.name())
                    if best_run:
                        safe_algo = algo.name().replace(" ", "_").lower()
                        filename = f"anim_{safe_algo}_{safe_prob}.gif"
                        plot_animation_2d(problem, best_run['trajectory'], algo.name(), filename)