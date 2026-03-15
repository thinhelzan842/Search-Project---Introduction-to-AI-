import os
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
from core import ContinuousProblem 
from plotly.subplots import make_subplots

os.makedirs("results", exist_ok=True)

# --- 1. HỘI TỤ ROBUST (Chỉ giữ đường Mean) ---
def plot_convergence_robust(histories_dict, title="Convergence (Mean)", filename="convergence.html"):
    fig = go.Figure()
    for algo_name, runs in histories_dict.items():
        if not runs: continue
        
        max_len = max(len(r) for r in runs)
        padded_runs = []
        for r in runs:
            if len(r) < max_len:
                padded_runs.append(r + [r[-1]] * (max_len - len(r)))
            else:
                padded_runs.append(r)
        
        runs_array = np.array(padded_runs)
        mean_hist = np.mean(runs_array, axis=0) + 1e-12
        x_vals = list(range(len(mean_hist)))

        fig.add_trace(go.Scatter(x=x_vals, y=mean_hist, mode='lines', name=algo_name))

    fig.update_layout(title=title, template="plotly_white", yaxis_type="log", 
                      xaxis_title="Iterations", yaxis_title="Fitness (Log Scale)")
    fig.write_html(os.path.join("results", filename))
        
# --- 2. BOXPLOT (QUALITY & ROBUSTNESS) ---
def plot_boxplot_performance(scores_dict, title="Robustness & Quality (Boxplot)", filename="boxplot.html"):
    import pandas as pd
    data = []
    for algo, scores in scores_dict.items():
        for s in scores: data.append({"Algorithm": algo, "Best Fitness": s})
    df = pd.DataFrame(data)
    fig = px.box(df, x="Algorithm", y="Best Fitness", color="Algorithm", points="all", title=title)
    fig.update_layout(template="plotly_white", yaxis_type="log")
    fig.write_html(os.path.join("results", filename))

# --- 3. EXPLORATION VS EXPLOITATION ---
def plot_exploration_exploitation(diversity_history, title="Exploration vs Exploitation", filename="explore_exploit.html"):
    if len(diversity_history) == 0: return
    div_array = np.array(diversity_history)
    max_div = np.max(div_array) if np.max(div_array) > 0 else 1
    exploration = (div_array / max_div) * 100
    exploitation = 100 - exploration
    
    fig = go.Figure()
    x = list(range(len(div_array)))
    fig.add_trace(go.Scatter(x=x, y=exploration, stackgroup='one', name='Exploration (%)', line=dict(color='cyan')))
    fig.add_trace(go.Scatter(x=x, y=exploitation, stackgroup='one', name='Exploitation (%)', line=dict(color='royalblue')))
    fig.update_layout(title=title, template="plotly_white", yaxis_range=[0, 100],
                      xaxis_title="Iterations", yaxis_title="Percentage (%)")
    fig.write_html(os.path.join("results", filename))

# --- 4. 3D LANDSCAPE ---
def plot_3d_landscape(problem, trajectories_dict, filename="3d_landscape.html"):
    if not isinstance(problem, ContinuousProblem) or not hasattr(problem, 'dim') or problem.dim != 2:
        return 
    bounds = problem.get_bounds()[0]
    x = np.linspace(bounds[0], bounds[1], 100)
    y = np.linspace(bounds[0], bounds[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[problem.evaluate([xi, yi]) for xi in x] for yi in y])
            
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', opacity=0.6, showscale=False)])
    for algo_name, trajectory in trajectories_dict.items():
        if not trajectory: continue
        traj_x = [pt[0] for pt in trajectory]
        traj_y = [pt[1] for pt in trajectory]
        traj_z = [problem.evaluate(pt) for pt in trajectory]
        fig.add_trace(go.Scatter3d(x=traj_x, y=traj_y, z=traj_z, mode='lines', name=algo_name))
        
    fig.update_layout(title=f"3D Landscape: {problem.name()}", scene=dict(xaxis_title='X1', yaxis_title='X2', zaxis_title='Fitness'))
    fig.write_html(os.path.join("results", filename))

# --- 5. HEATMAP ---
def plot_heatmap(data_matrix, algos, problems, title="Heatmap", filename="heatmap.html"):
    fig = px.imshow(data_matrix, x=problems, y=algos, text_auto=True, color_continuous_scale='RdBu_r')
    fig.update_layout(title=title)
    fig.write_html(os.path.join("results", filename))

# --- 6. SCALABILITY & COMPLEXITY ---
def plot_scalability(sizes, times, spaces, fitnesses, algo_name="Algorithm", filename="scalability.html"):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(go.Scatter(x=sizes, y=times, name="Time (s)", mode='lines+markers', 
                             line=dict(color='red', width=3), marker=dict(size=8)), secondary_y=False)
    fig.add_trace(go.Scatter(x=sizes, y=spaces, name="Space (KB)", mode='lines+markers', 
                             line=dict(color='green', width=3, dash='dot'), marker=dict(size=8, symbol='square')), secondary_y=False)
    
    fig.add_trace(go.Scatter(x=sizes, y=fitnesses, name="Best Fitness", mode='lines+markers', 
                             line=dict(color='blue', width=3), marker=dict(size=8)), secondary_y=True)
    
    fig.update_layout(title=f"Scalability & Complexity: {algo_name}", 
                      xaxis_title="Problem Size (Dimensions)", template="plotly_white")
    fig.update_yaxes(title_text="Execution Time (s) & Space Peak (KB)", secondary_y=False)
    fig.update_yaxes(title_text="Best Fitness (Log Scale)", type="log", secondary_y=True)
    fig.write_html(os.path.join("results", filename))

# --- 7. P-VALUE HEATMAP (STATISTICAL SIGNIFICANCE) ---
def plot_pvalue_heatmap(p_matrix, algos, title="Statistical Significance (P-Value)", filename="pvalue.html"):
    text_matrix = []
    for row in p_matrix:
        text_row = []
        for val in row:
            if val == 1.0: text_row.append("-")
            elif val < 0.001: text_row.append(f"{val:.1e}<br>(***)")
            elif val < 0.01: text_row.append(f"{val:.1e}<br>(**)")
            elif val < 0.05: text_row.append(f"{val:.1e}<br>(*)")
            else: text_row.append(f"{val:.2f}<br>(ns)")
        text_matrix.append(text_row)

    fig = go.Figure(data=go.Heatmap(
        z=p_matrix, x=algos, y=algos,
        text=text_matrix, texttemplate="%{text}",
        colorscale='RdYlGn_r', zmin=0, zmax=0.05
    ))
    fig.update_layout(title=title, template="plotly_white")
    fig.write_html(os.path.join("results", filename))

# --- 8. FIRST HITTING TIME (CONVERGENCE SPEED) ---
def plot_first_hitting_time(hitting_times_dict, title="First Hitting Time", filename="hitting_time.html"):
    algos = list(hitting_times_dict.keys())
    times = list(hitting_times_dict.values())
    fig = go.Figure(data=[go.Bar(x=algos, y=times, text=[f"{t:.1f}" for t in times], textposition='auto', marker_color='indianred')])
    fig.update_layout(title=title, template="plotly_white", xaxis_title="Algorithms", yaxis_title="Iterations to Threshold")
    fig.write_html(os.path.join("results", filename))

# --- 9. FITNESS DEGRADATION (CURSE OF DIMENSIONALITY) ---
def plot_fitness_degradation(dims, fitness_dict, title="Fitness Degradation", filename="degradation.html"):
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    for i, (algo, fitnesses) in enumerate(fitness_dict.items()):
        fig.add_trace(go.Scatter(x=dims, y=fitnesses, mode='lines+markers', name=algo, 
                                 line=dict(color=colors[i%len(colors)], width=3), marker=dict(size=8)))
    fig.update_layout(title=title, template="plotly_white", xaxis_title="Dimensions", yaxis_title="Best Fitness (Log Scale)", yaxis_type="log")
    fig.write_html(os.path.join("results", filename))