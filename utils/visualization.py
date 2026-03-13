import os
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from core import ContinuousProblem  # Imported for type checking

os.makedirs("results", exist_ok=True)

def plot_convergence(histories_dict, title="Convergence Curve", filename="convergence.html"):
    fig = go.Figure()
    for algo_name, history in histories_dict.items():
        safe_history = [val + 1e-10 for val in history]
        fig.add_trace(go.Scatter(y=safe_history, mode='lines', name=algo_name))
    fig.update_layout(title=title, xaxis_title="Iterations", yaxis_title="Best Fitness (Log Scale)", template="plotly_white", yaxis_type="log")
    
    filepath = os.path.join("results", filename)
    fig.write_html(filepath)
    print(f"   [+] Exported chart: {filepath}")

def plot_performance_bar(scores_dict, title="Performance Comparison", filename="bar_chart.html"):
    algos = list(scores_dict.keys())
    scores = list(scores_dict.values())
    fig = px.bar(x=algos, y=scores, color=algos, title=title, labels={'x': 'Algorithms', 'y': 'Final Best Fitness'})
    fig.write_html(os.path.join("results", filename))
    print(f"   [+] Exported chart: results/{filename}")

def plot_3d_landscape(problem, trajectories_dict, filename="3d_landscape.html"):
    # Robust check to ensure we only plot 2D continuous problems
    if not isinstance(problem, ContinuousProblem) or not hasattr(problem, 'dim') or problem.dim != 2:
        return 
    
    bounds = problem.get_bounds()[0]
    x = np.linspace(bounds[0], bounds[1], 100)
    y = np.linspace(bounds[0], bounds[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = problem.evaluate([X[i, j], Y[i, j]])
            
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', opacity=0.6, showscale=False)])
    
    for algo_name, trajectory in trajectories_dict.items():
        traj_x = [pt[0] for pt in trajectory]
        traj_y = [pt[1] for pt in trajectory]
        traj_z = [problem.evaluate(pt) for pt in trajectory]
        
        fig.add_trace(go.Scatter3d(
            x=traj_x, y=traj_y, z=traj_z, 
            mode='lines+markers',
            marker=dict(size=3), 
            line=dict(width=3),
            name=algo_name 
        ))
        
    fig.update_layout(
        title=f"3D Loss Landscape & Trajectories: {problem.name()}", 
        scene=dict(xaxis_title='X1', yaxis_title='X2', zaxis_title='Fitness'),
        legend=dict(x=0, y=1) 
    )
    
    fig.write_html(os.path.join("results", filename))
    print(f"   [+] Exported chart: results/{filename}")

def plot_heatmap(data_matrix, algos, problems, title="Performance Heatmap", filename="heatmap.html"):
    fig = px.imshow(data_matrix, x=problems, y=algos, text_auto=True, 
                    title=title, aspect="auto", color_continuous_scale='RdBu_r')
    fig.write_html(os.path.join("results", filename))
    print(f"   [+] Exported chart: results/{filename}")

def plot_animation_2d(problem, trajectory, algo_name, filename="animation.gif"):
    # Robust check to ensure we only animate 2D continuous problems
    if not isinstance(problem, ContinuousProblem) or not hasattr(problem, 'dim') or problem.dim != 2:
        return 
        
    fig, ax = plt.subplots(figsize=(8, 8))
    bounds = problem.get_bounds()[0]
    x_val = np.linspace(bounds[0], bounds[1], 100)
    y_val = np.linspace(bounds[0], bounds[1], 100)
    X, Y = np.meshgrid(x_val, y_val)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = problem.evaluate([X[i, j], Y[i, j]])
            
    ax.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.8)
    ax.set_title(f"Trajectory: {algo_name} on {problem.name()}", fontsize=14, fontweight='bold')
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    
    ax.plot([0], [0], marker='*', color='yellow', markersize=15, label="Global Optimum")
    
    agent, = ax.plot([], [], marker='o', color='red', markersize=8, label="Search Agent")
    tail, = ax.plot([], [], color='white', linewidth=1.5, alpha=0.5)
    ax.legend(loc="upper right")

    history_x = [pt[0] for pt in trajectory]
    history_y = [pt[1] for pt in trajectory]

    def update(frame):
        agent.set_data([history_x[frame]], [history_y[frame]])
        start_tail = max(0, frame - 20)
        tail.set_data(history_x[start_tail:frame+1], history_y[start_tail:frame+1])
        return agent, tail

    ani = animation.FuncAnimation(fig, update, frames=len(history_x), interval=20, blit=True, repeat=False)
    
    filepath = os.path.join("results", filename)
    try:
        ani.save(filepath, writer='pillow', fps=30)
        print(f"   [+] Exported animation: {filepath}")
    except Exception as e:
        print(f"   [-] Lỗi xuất Animation: {e}")
    finally:
        plt.close(fig)