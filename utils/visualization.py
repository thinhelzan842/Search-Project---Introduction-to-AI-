import os
import sys 
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

os.makedirs("results", exist_ok=True)

def plot_convergence(histories_dict, title="Convergence Curve", filename="convergence.html"):
    fig = go.Figure()
    for algo_name, history in histories_dict.items():
        fig.add_trace(go.Scatter(y=history, mode='lines', name=algo_name))
    fig.update_layout(title=title, xaxis_title="Iterations", yaxis_title="Best Fitness", template="plotly_white", yaxis_type="log")
    
    # export to HTML file 
    filepath = os.path.join("results", filename)
    fig.write_html(filepath)
    print(f"   [+] Exported chart: {filepath}")

def plot_performance_bar(scores_dict, title="Performance Comparison", filename="bar_chart.html"):
    algos = list(scores_dict.keys())
    scores = list(scores_dict.values())
    fig = px.bar(x=algos, y=scores, color=algos, title=title, labels={'x': 'Algorithms', 'y': 'Final Best Fitness'})
    fig.write_html(os.path.join("results", filename))
    print(f"   [+] Exported chart: results/{filename}")

def plot_3d_landscape(problem, trajectory, algo_name="Algorithm", filename="3d_landscape.html"):
    if problem.is_discrete() or problem.dim != 2:
        return 
    bounds = problem.get_bounds()[0]
    x = np.linspace(bounds[0], bounds[1], 100)
    y = np.linspace(bounds[0], bounds[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = problem.evaluate([X[i, j], Y[i, j]])
            
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', opacity=0.8)])
    traj_x = [pt[0] for pt in trajectory]
    traj_y = [pt[1] for pt in trajectory]
    traj_z = [problem.evaluate(pt) for pt in trajectory]
    
    fig.add_trace(go.Scatter3d(x=traj_x, y=traj_y, z=traj_z, mode='lines+markers',
                               marker=dict(size=4, color='red'), line=dict(color='red', width=2),
                               name=f'{algo_name} Trajectory'))
    fig.update_layout(title=f"Loss Landscape: {problem.name()} - {algo_name}", 
                      scene=dict(xaxis_title='X1', yaxis_title='X2', zaxis_title='Fitness'))
    fig.write_html(os.path.join("results", filename))
    print(f"   [+] Exported chart: results/{filename}")

def plot_heatmap(data_matrix, algos, problems, title="Performance Heatmap", filename="heatmap.html"):
    fig = px.imshow(data_matrix, x=problems, y=algos, text_auto=True, 
                    title=title, aspect="auto", color_continuous_scale='RdBu_r')
    fig.write_html(os.path.join("results", filename))
    print(f"   [+] Exported chart: results/{filename}")