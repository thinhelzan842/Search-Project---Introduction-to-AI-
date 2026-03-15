import os
import plotly.graph_objects as go
import numpy as np
from problems import *
from algorithms import *

def visualize_pathfinding(problem, algorithms, filename):
    """
    Standalone visualizer for graph problems with explicit edge drawing,
    distinct start/end markers, and separated node IDs.
    """
    trajectories = {}
    print(f"\nEvaluating Pathfinding on: {problem.name()}")
    
    for algo in algorithms:
        print(f"  >> Solving with {algo.name()}...")
        final_state = None
        for state in algo.run(problem):
            final_state = state
            
        if final_state and final_state['best_score'] != float('inf'):
            trajectories[algo.name()] = final_state['best_solution']
        else:
            print(f"     [!] {algo.name()} found no valid path.")

    fig = go.Figure()

    edge_x, edge_y = [], []
    for u in range(problem.size):
        for v in problem.adj[u]:
            edge_x.extend([problem.coords[u][0], problem.coords[v][0], None])
            edge_y.extend([problem.coords[u][1], problem.coords[v][1], None])
        
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode='lines',
        line=dict(color='rgba(200, 200, 200, 0.5)', width=1),
        name="Allowed Edges", hoverinfo='skip'
    ))

    # --- 2. DRAW NODES (With separated IDs) ---
    coords = np.array(problem.coords)
    # Main nodes
    fig.add_trace(go.Scatter(
        x=coords[:, 0], y=coords[:, 1], mode='markers+text',
        text=[f"  <b>ID: {i}</b>" for i in range(len(coords))],
        textposition="top right", # Offsets text to avoid overriding markers
        marker=dict(size=8, color='darkslategrey'),
        name="Nodes"
    ))
        
        # Start Node (ID 0)
    fig.add_trace(go.Scatter(
        x=[coords[0, 0]], y=[coords[0, 1]], mode='markers',
        marker=dict(size=15, color='lime', line=dict(width=2, color='black')),
        name="START"
    ))
        
        # End Node (Last ID)
    fig.add_trace(go.Scatter(
        x=[coords[-1, 0]], y=[coords[-1, 1]], mode='markers',
        marker=dict(size=15, color='red', symbol='star', line=dict(width=2, color='black')),
        name="END"
    ))

    # --- 3. DRAW TRAJECTORIES ---
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']
    for i, (algo_name, path) in enumerate(trajectories.items()):
        if isinstance(problem, GridShortestPath):
            px = [p[0] for p in path]
            py = [p[1] for p in path]
        else:
            px = [problem.coords[node][0] for node in path]
            py = [problem.coords[node][1] for node in path]

        fig.add_trace(go.Scatter(
            x=px, y=py, mode='lines+markers',
            name=algo_name, 
            line=dict(width=4, color=colors[i % len(colors)]),
            marker=dict(size=6)
        ))

    fig.update_layout(
        title=f"Pathfinding Map: {problem.name()}",
        template="plotly_white",
        xaxis_title="X Coordinate", yaxis_title="Y Coordinate",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    os.makedirs("results", exist_ok=True)
    full_path = os.path.join("results", filename)
    fig.write_html(full_path)
    print(f"Visualization saved to: {full_path}")

if __name__ == "__main__":
    # Test with the specific Euclidean problem you requested
    euclidean = EuclideanShortestPath(size=25, edge_probability=0.2)
    algos = [AStar(), UCS(), GreedyBFS(), BFS(), DFS()]
    
    visualize_pathfinding(euclidean, algos, "recolored_pathfinding.html")