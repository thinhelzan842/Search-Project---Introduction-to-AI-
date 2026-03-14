import os
import pickle
import numpy as np
import plotly.graph_objects as go
from scipy import stats

def load_data(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def generate_statistical_report(results, problem_name):
    print(f"\n{'=' * 60}")
    print(f"STATISTICAL REPORT & ROBUSTNESS: {problem_name.upper()}")
    print(f"{'=' * 60}")

    prob_data = [r for r in results if r['problem'] == problem_name]
    algos = list(set(r['algo'] for r in prob_data))

    algo_scores = {}
    algo_times = {}

    for algo in algos:
        # 1. FILTER: Ignore 'None' values from algorithms that failed to yield
        scores = [r['best_score'] for r in prob_data if r['algo'] == algo and r.get('best_score') is not None]
        times = [r['time'] for r in prob_data if r['algo'] == algo and r.get('time') is not None]

        # 2. SKIP algorithms that completely failed all runs
        if not scores:
            print(f"Algorithm: {algo:<20} | FAILED (No valid iterations/scores recorded)")
            continue

        algo_scores[algo] = scores
        algo_times[algo] = times

        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(
            f"Algorithm: {algo:<20} | Robustness (Mean ± Std): {mean_score:.4e} ± {std_score:.4e} | Avg Time: {np.mean(times):.3f}s")

    # 3. Use ONLY valid algorithms for plotting
    valid_algos = list(algo_scores.keys())

    fig_box = go.Figure()
    for algo in valid_algos:
        safe_scores = [s for s in algo_scores[algo] if s != float('inf')]
        fig_box.add_trace(go.Box(y=safe_scores, name=algo, boxpoints='all', jitter=0.3, pointpos=-1.8))

    fig_box.update_layout(
        title=f"Comparative Performance (Solution Quality) - {problem_name}",
        yaxis_title="Final Best Score (Log Scale)",
        yaxis_type="log",
        template="plotly_white",
        showlegend=False
    )

    box_filename = f"results/boxplot_{problem_name.replace(' ', '_').lower()}.html"
    fig_box.write_html(box_filename)
    print(f"   [+] Exported Boxplot: {box_filename}")

    # 4. Use ONLY valid algorithms for T-Test
    print("\n--- HYPOTHESIS TESTING (T-Test) ---")
    if len(valid_algos) >= 2:
        best_algo = min(valid_algos, key=lambda a: np.mean(algo_scores[a]))
        print(f"Best performing algorithm (by mean): {best_algo}")

        for other_algo in valid_algos:
            if other_algo == best_algo: continue

            if np.std(algo_scores[best_algo]) == 0 and np.std(algo_scores[other_algo]) == 0:
                p_val = 1.0
            else:
                t_stat, p_val = stats.ttest_ind(algo_scores[best_algo], algo_scores[other_algo])

            sig = "SIGNIFICANT (p < 0.05)" if p_val < 0.05 else "NOT significant"
            print(f"{best_algo} vs {other_algo:<15} | p-value: {p_val:.4e} -> {sig}")


def plot_exploration_exploitation(results, problem_name):
    """Vẽ biểu đồ đo lường hành vi Khám phá (Exploration) và Khai thác (Exploitation) bằng Plotly"""
    prob_data = [r for r in results if r['problem'] == problem_name]
    algos = list(set(r['algo'] for r in prob_data))

    fig_line = go.Figure()

    for algo in algos:
        # Lấy mảng diversity của tất cả các runs
        runs_diversity = [r.get('diversity', []) for r in prob_data if r['algo'] == algo]

        # 1. SANITIZE DATA: Convert whatever is in 'diversity' to a single scalar number
        valid_runs = []
        for run in runs_diversity:
            if isinstance(run, list) and len(run) > 0:
                clean_run = []
                for val in run:
                    if val is None:
                        clean_run.append(0.0)
                    elif isinstance(val, (list, tuple, np.ndarray)):
                        # If the data is a population or coordinate, calculate its variance
                        # to get a single number representing "spread" (diversity)
                        try:
                            clean_run.append(float(np.var(val)))
                        except:
                            clean_run.append(0.0)  # Fallback just in case
                    else:
                        # If it is already a single number
                        clean_run.append(float(val))
                valid_runs.append(clean_run)

        # Skip this algorithm entirely if there is no valid diversity data
        if not valid_runs:
            continue

        # 2. ALIGN LENGTHS: Tìm chiều dài ngắn nhất để cắt mảng
        min_len = min(len(run) for run in valid_runs)
        trimmed_runs = [run[:min_len] for run in valid_runs]

        # 3. CALCULATE MEAN: Now NumPy only sees clean, 1D arrays of numbers!
        mean_diversity = np.mean(trimmed_runs, axis=0)

        # Chuẩn hóa về % (so với giá trị max trong lịch sử của chính nó hoặc tổng thể)
        max_div = np.max(mean_diversity) if np.max(mean_diversity) > 0 else 1
        normalized_div = (mean_diversity / max_div) * 100

        iterations = list(range(1, len(normalized_div) + 1))

        fig_line.add_trace(go.Scatter(
            x=iterations,
            y=normalized_div,
            mode='lines',
            name=algo,
            line=dict(width=2)
        ))

    fig_line.update_layout(
        title=f"Exploration vs Exploitation Behavior - {problem_name}",
        xaxis_title="Iterations",
        yaxis_title="Population Diversity (%) <br><sup>Low = Exploitation | High = Exploration</sup>",
        template="plotly_white",
        legend=dict(x=1.02, y=1, bordercolor="Black", borderwidth=1)
    )

    exp_filename = f"results/explore_exploit_{problem_name.replace(' ', '_').lower()}.html"
    fig_line.write_html(exp_filename)
    print(f"   [+] Exported Exp/Exp chart: {exp_filename}")

def plot_space_complexity(results, problem_name):
    prob_data = [r for r in results if r['problem'] == problem_name]
    algos = list(set(r['algo'] for r in prob_data))
    
    # Bỏ qua nếu data cũ chưa có trường peak_memory
    if not prob_data or 'peak_memory' not in prob_data[0]: return
    
    memories = {algo: np.mean([r.get('peak_memory', 0) for r in prob_data if r['algo'] == algo]) for algo in algos}
        
    fig_bar = go.Figure(data=[go.Bar(x=list(memories.keys()), y=list(memories.values()), marker_color='teal')])
    fig_bar.update_layout(title=f"Space Complexity (Peak Memory) - {problem_name}", yaxis_title="Peak Memory (KB)", template="plotly_white")
    
    mem_filename = f"results/space_{problem_name.replace(' ', '_').lower()}.html"
    fig_bar.write_html(mem_filename)
    print(f"   [+] Exported Space Complexity: {mem_filename}")

def plot_sensitivity_and_scalability():
    if os.path.exists("results/sensitivity_results.pkl"):
        data = load_data("results/sensitivity_results.pkl")
        fig = go.Figure(go.Scatter(x=[r['param'] for r in data], y=[r['score'] for r in data], mode='lines+markers'))
        fig.update_layout(title="Parameter Sensitivity Analysis (PSO: num_particles vs Loss)", xaxis_title="Number of Particles", yaxis_title="Best Final Score")
        fig.write_html("results/sensitivity_analysis.html")
        print("   [+] Exported Sensitivity Analysis: results/sensitivity_analysis.html")

    if os.path.exists("results/scalability_results.pkl"):
        data = load_data("results/scalability_results.pkl")
        fig = go.Figure(go.Scatter(x=[r['dim'] for r in data], y=[r['time'] for r in data], mode='lines+markers', marker_color='green'))
        fig.update_layout(title="Scalability Analysis (Problem Dimension vs Execution Time)", xaxis_title="Dimension Size", yaxis_title="Time (seconds)")
        fig.write_html("results/scalability_analysis.html")
        print("   [+] Exported Scalability Analysis: results/scalability_analysis.html")

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    
    # Danh sách các file dữ liệu mà main.py sẽ xuất ra
    target_files = [
        "results/continuous_results.pkl",
        "results/discrete_results.pkl",
        "results/graph_results.pkl"
    ]
    
    for filepath in target_files:
        if not os.path.exists(filepath):
            print(f"\n[CẢNH BÁO] Không tìm thấy file {filepath}. Bỏ qua...")
            continue
            
        print(f"\n{'='*80}")
        print(f"ĐANG XỬ LÝ DỮ LIỆU TỪ FILE: {filepath}")
        print(f"{'='*80}")
        
        data = load_data(filepath)
        problems = list(set(r['problem'] for r in data))
        
        for p in problems:
            generate_statistical_report(data, p)
            plot_exploration_exploitation(data, p)
            plot_space_complexity(data, p)
        
    plot_sensitivity_and_scalability()        
    print("\n[Hoàn thành] Tất cả báo cáo đã được tạo trong thư mục results/")