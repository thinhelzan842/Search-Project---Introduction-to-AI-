import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
import math

# -----------------------------------------------------------------------
# Synchronize Imports with your main.py / structures
# -----------------------------------------------------------------------
from algorithms import *
from problems import *

st.set_page_config(page_title="Optimization Visualizer", layout="wide")
st.title("Optimization Algorithms: Step-by-Step Visualization")

# ==========================================
# 1. SIDEBAR - SELECT PROBLEM CATEGORY
# ==========================================
st.sidebar.header("1. Problem Settings")
problem_type = st.sidebar.radio("Problem Type", ["Continuous", "Discrete"])

if problem_type == "Continuous":
    problem_name = st.sidebar.selectbox(
        "Select Problem", 
        ["Sphere", "Rastrigin", "Ackley", "Rosenbrock", "Griewank"]
    )

    problems_dict = {
        "Sphere": Sphere,
        "Rastrigin": Rastrigin,
        "Ackley": Ackley,
        "Rosenbrock": Rosenbrock,
        "Griewank": Griewank
    }

    # Sync bound with main.py
    bound_val = 5.12 if problem_name == "Rastrigin" else 5.0
    problem = problems_dict[problem_name](dim=2, bound=bound_val)

    st.sidebar.header("2. Algorithm Settings")
    algo_name = st.sidebar.selectbox("Select Algorithm", ["PSO", "SimulatedAnnealing", "HillClimbing", "ArtificialBeeColony"])

    # Render dynamic sliders matching main.py defaults
    if algo_name == "PSO":
        num_particles = st.sidebar.slider("Number of particles", 5, 100, 20)
        max_iters = st.sidebar.slider("Max iterations", 10, 200, 50)
        algo = PSO(num_particles=num_particles, max_iters=max_iters)

    elif algo_name == "SimulatedAnnealing":
        max_epochs = st.sidebar.slider("Max epochs", 10, 200, 50)
        initial_temp = st.sidebar.slider("Initial temp", 10.0, 500.0, 100.0)
        cooling_rate = st.sidebar.slider("Cooling rate", 0.8, 0.999, 0.99)
        step_size = st.sidebar.slider("Step size", 0.1, 2.0, 0.5)
        markov_chain_length = st.sidebar.slider("Markov Chain Length", 5, 50, 20)
        algo = SimulatedAnnealing(
            max_epochs=max_epochs, initial_temp=initial_temp, 
            cooling_rate=cooling_rate, step_size=step_size, 
            markov_chain_length=markov_chain_length
        )

    elif algo_name == "HillClimbing":
        max_iters = st.sidebar.slider("Max iterations", 10, 200, 50)
        step_size = st.sidebar.slider("Step size", 0.1, 2.0, 0.5)
        num_neighbors = st.sidebar.slider("Number of neighbors", 5, 50, 15)
        algo = HillClimbing(max_iters=max_iters, step_size=step_size, num_neighbors=num_neighbors)
        
    elif algo_name == "ArtificialBeeColony":
        popsize = st.sidebar.slider("Population size", 10, 100, 20)
        max_gens = st.sidebar.slider("Max generations", 10, 200, 50)
        algo = ArtificialBeeColony(popsize=popsize, gen=max_gens)

else:
    problem_name = st.sidebar.selectbox("Select Problem", 
        ["Shortest Path", "Traveling Salesman", "Knapsack", "Graph Coloring"]
    )
    
    # Sync problem instantiations with main.py
    if problem_name == "Shortest Path":
        graph_size = st.sidebar.slider("Graph Size", 5, 50, 12)
        problem = ShortestPath(size=graph_size)
    elif problem_name == "Traveling Salesman":
        tsp_size = st.sidebar.slider("Number of Cities", 5, 50, 10)
        problem = TravelingSalesman(size=tsp_size, time_limit=2000, cost_limit=2000)
    elif problem_name == "Knapsack":
        ks_size = st.sidebar.slider("Number of Items", 5, 50, 15)
        problem = Knapsack(size=ks_size, limit=40)
    elif problem_name == "Graph Coloring":
        gc_size = st.sidebar.slider("Graph Size", 5, 50, 10)
        problem = GraphColoring(size=gc_size)

    st.sidebar.header("2. Algorithm Settings")
    # Tạm thời hiển thị các thuật toán áp dụng cho Discrete
    algo_name = st.sidebar.selectbox("Select Algorithm", ["ACO", "HillClimbing", "SimulatedAnnealing"])
    
    if algo_name == "ACO":
        num_ants = st.sidebar.slider("Number of ants", 5, 100, 20)
        max_iters = st.sidebar.slider("Max iterations", 10, 200, 50)
        alpha = st.sidebar.slider("Alpha (Pheromone)", 0.1, 5.0, 1.0)
        beta = st.sidebar.slider("Beta (Heuristic)", 0.1, 5.0, 2.0)
        algo = ACO(num_ants=num_ants, max_iters=max_iters, alpha=alpha, beta=beta)
    elif algo_name == "HillClimbing":
        max_iters = st.sidebar.slider("Max iterations", 10, 200, 50)
        algo = HillClimbing(max_iters=max_iters, step_size=1, num_neighbors=15)
    elif algo_name == "SimulatedAnnealing":
        max_epochs = st.sidebar.slider("Max epochs", 10, 200, 50)
        algo = SimulatedAnnealing(max_epochs=max_epochs, initial_temp=100.0, cooling_rate=0.99, step_size=1, markov_chain_length=20)

st.sidebar.header("3. Interface Settings")
delay = st.sidebar.slider("Animation speed (seconds/frame)", 0.0, 0.5, 0.1)

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
@st.cache_data
def get_contour_data(prob_name, bound):
    x = np.linspace(-bound, bound, 100)
    y = np.linspace(-bound, bound, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    temp_prob = problems_dict[prob_name](dim=2, bound=bound)
    for i in range(100):
        for j in range(100):
            Z[i, j] = temp_prob.evaluate([X[i, j], Y[i, j]])
    return X, Y, Z

def get_circle_coords(n):
    """Sinh tọa độ vòng tròn cho các bài toán không có sẵn tọa độ như TSP, GraphColoring"""
    return [(50 + 40*math.cos(2*math.pi*i/n), 50 + 40*math.sin(2*math.pi*i/n)) for i in range(n)]

# ==========================================
# 4. RUN ALGORITHM & ANIMATION
# ==========================================
if st.button("Run Algorithm", type="primary"):
    generator = algo.run(problem)

    info_placeholder = st.empty()
    plot_placeholder = st.empty()

    fig, ax = plt.subplots(figsize=(10, 4)) 
    
    # --- SETUP PLOT DỰA THEO TYPE ---
    if problem_type == "Continuous":
        X, Y, Z = get_contour_data(problem_name, bound_val)
        contour = ax.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.8)
        fig.colorbar(contour, ax=ax, label="Fitness Value")
        
        scatter = ax.scatter([], [], c='red', edgecolors='white', s=50, zorder=5)
        best_scatter = ax.scatter([], [], marker='*', c='gold', edgecolors='black', s=250, zorder=10)
        
        ax.set_xlim(-bound_val, bound_val)
        ax.set_ylim(-bound_val, bound_val)
        ax.set_title(f"{algo_name} running on {problem_name}")
    
    else:
        ax.set_title(f"{algo_name} running on {problem_name}")
        
        if problem_name in ["Shortest Path", "Traveling Salesman", "Graph Coloring"]:
            # Lấy tọa độ (nếu có) hoặc sinh tự động
            if hasattr(problem, 'coords'):
                coords = problem.coords
            else:
                coords = get_circle_coords(problem.size)
                
            x_coords = [c[0] for c in coords]
            y_coords = [c[1] for c in coords]
            
            # Vẽ các cạnh nền
            if hasattr(problem, 'adj'):
                for i in range(len(problem.adj)):
                    for j in problem.adj[i]:
                        if i < j:
                            ax.plot([coords[i][0], coords[j][0]], [coords[i][1], coords[j][1]], c='lightgray', zorder=1, alpha=0.4)
            elif hasattr(problem, 'costs'):
                for i in range(problem.size):
                    for j in range(i+1, problem.size):
                        if problem.costs[i][j] != float('inf'):
                            ax.plot([coords[i][0], coords[j][0]], [coords[i][1], coords[j][1]], c='lightgray', zorder=1, alpha=0.2)
                            
            node_scatter = ax.scatter(x_coords, y_coords, c='skyblue', edgecolors='black', s=100, zorder=3)
            
            if problem_name == "Shortest Path":
                ax.scatter(coords[0][0], coords[0][1], c='green', s=150, zorder=4, label='Start')
                ax.scatter(coords[-1][0], coords[-1][1], c='red', s=150, zorder=4, label='Goal')
                
            path_line, = ax.plot([], [], c='blue', linewidth=3, zorder=2)
            
        elif problem_name == "Knapsack":
            # Bar chart cho bài toán cái túi
            bars = ax.bar(range(problem.size), problem.profits, color='lightgray', edgecolor='black')
            ax.set_xticks(range(problem.size))
            ax.set_ylabel('Profit')
            ax.set_xlabel('Items')

    # --- ANIMATION LOOP ---
    for step_data in generator:
        iteration = step_data.get('iteration', step_data.get('generation', 0))
        best_score = step_data.get('best_score', float('inf'))
        best_solution = step_data.get('best_solution', [])
        
        if problem_type == "Continuous":
            if 'population_positions' in step_data or 'population' in step_data:
                positions = np.array(step_data.get('population_positions', step_data.get('population')))
                scatter.set_offsets(positions[:, :2])
            elif 'current_solution' in step_data:
                current_sol = np.array(step_data['current_solution']).reshape(1, 2)
                scatter.set_offsets(current_sol[:, :2])
                
            if best_solution is not None and len(best_solution) >= 2:
                best_scatter.set_offsets([best_solution[0], best_solution[1]])
                
        else:
            if best_solution and best_score != float('inf'):
                if problem_name in ["Shortest Path", "Traveling Salesman"]:
                    path_x = [coords[node][0] for node in best_solution]
                    path_y = [coords[node][1] for node in best_solution]
                    if problem_name == "Traveling Salesman" and len(best_solution) > 1:
                        path_x.append(coords[best_solution[0]][0]) # Quay về điểm đầu
                        path_y.append(coords[best_solution[0]][1])
                    path_line.set_data(path_x, path_y)
                    
                elif problem_name == "Graph Coloring":
                    colors = plt.cm.get_cmap('tab20', problem.size)
                    node_colors = [colors(c) for c in best_solution]
                    # Nếu giải pháp chưa đầy đủ (đang search), fill màu xám cho các node còn lại
                    while len(node_colors) < problem.size:
                        node_colors.append((0.8, 0.8, 0.8, 1.0))
                    node_scatter.set_color(node_colors)
                    node_scatter.set_edgecolor('black')
                    
                elif problem_name == "Knapsack":
                    for i, bar in enumerate(bars):
                        if i < len(best_solution) and best_solution[i] == 1:
                            bar.set_color('green') # Chọn
                        else:
                            bar.set_color('lightgray') # Bỏ qua

        plot_placeholder.pyplot(fig)
        info_placeholder.info(f"Iteration: **{iteration}** | Best Score: **{best_score:.6f}**")
        time.sleep(delay)
        
    st.success(f"Successfully completed running {algo_name} algorithm!")