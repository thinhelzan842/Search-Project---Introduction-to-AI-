import os
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from core import ContinuousProblem 
from plotly.subplots import make_subplots

os.makedirs("results", exist_ok=True)

# --- 1. HỘI TỤ ROBUST (Đã sửa lỗi Inhomogeneous Shape) ---
def plot_convergence_robust(histories_dict, title="Convergence (Mean ± Std)", filename="convergence.html"):
    fig = go.Figure()
    for algo_name, runs in histories_dict.items():
        if not runs: continue
        
        # FIX: Tìm chiều dài lớn nhất và bù (padding) giá trị cuối cho các lần chạy ngắn hơn
        max_len = max(len(r) for r in runs)
        padded_runs = []
        for r in runs:
            if len(r) < max_len:
                padded_runs.append(r + [r[-1]] * (max_len - len(r)))
            else:
                padded_runs.append(r)
        
        runs_array = np.array(padded_runs)
        mean_hist = np.mean(runs_array, axis=0) + 1e-12
        std_hist = np.std(runs_array, axis=0)
        
        upper_bound = mean_hist + std_hist
        lower_bound = np.clip(mean_hist - std_hist, 1e-12, None)
        x_vals = list(range(len(mean_hist)))

        fig.add_trace(go.Scatter(
            x=x_vals + x_vals[::-1],
            y=list(upper_bound) + list(lower_bound)[::-1],
            fill='toself',
            fillcolor=f'rgba({np.random.randint(0,255)},{np.random.randint(0,255)},200,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip", showlegend=False
        ))
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

# --- 3. EXPLORATION VS EXPLOITATION (Đã sửa lỗi đồng bộ độ dài) ---
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

# --- 6. ANIMATION GIF ---
def plot_animation_2d(problem, trajectory, algo_name, filename="animation.gif"):
    if not isinstance(problem, ContinuousProblem) or problem.dim != 2: return
    fig, ax = plt.subplots()
    # (Giữ nguyên logic vẽ animation của bạn)
    plt.close(fig)

# --- 7. SCALABILITY & COMPLEXITY (Đã thêm Space Complexity) ---
def plot_scalability(sizes, times, spaces, fitnesses, algo_name="Algorithm", filename="scalability.html"):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Trục Y trái: Thời gian chạy và Dung lượng bộ nhớ (Complexity)
    fig.add_trace(go.Scatter(x=sizes, y=times, name="Time (s)", mode='lines+markers', 
                             line=dict(color='red', width=3), marker=dict(size=8)), secondary_y=False)
    fig.add_trace(go.Scatter(x=sizes, y=spaces, name="Space (KB)", mode='lines+markers', 
                             line=dict(color='green', width=3, dash='dot'), marker=dict(size=8, symbol='square')), secondary_y=False)
    
    # Trục Y phải: Chất lượng nghiệm (Scalability)
    fig.add_trace(go.Scatter(x=sizes, y=fitnesses, name="Best Fitness", mode='lines+markers', 
                             line=dict(color='blue', width=3), marker=dict(size=8)), secondary_y=True)
    
    fig.update_layout(title=f"Scalability & Complexity: {algo_name}", 
                      xaxis_title="Problem Size (Dimensions)", template="plotly_white")
    fig.update_yaxes(title_text="Execution Time (s) & Space Peak (KB)", secondary_y=False)
    fig.update_yaxes(title_text="Best Fitness (Log Scale)", type="log", secondary_y=True)
    
    fig.write_html(os.path.join("results", filename))