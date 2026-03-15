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

st.set_page_config(page_title="Visualizer", layout="wide")
st.sidebar.title("Algorithms Visualizer")

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
    
    # --- DANH SÁCH THUẬT TOÁN CONTINUOUS ĐẦY ĐỦ ---
    algo_name = st.sidebar.selectbox("Select Algorithm", [
        "PSO", "SimulatedAnnealing", "HillClimbing", "ArtificialBeeColony",
        "CuckooOptimization", "FireflyAlgorithm", "TabuSearch", 
        "GravitationalSearchAlgorithm", "HarmonySearch", "TLBO", 
        "DifferentialEvolution", "GeneticAlgorithm"
    ])

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
        algo = SimulatedAnnealing(max_epochs=max_epochs, initial_temp=initial_temp, cooling_rate=cooling_rate, step_size=step_size, markov_chain_length=markov_chain_length)

    elif algo_name == "HillClimbing":
        max_iters = st.sidebar.slider("Max iterations", 10, 200, 50)
        step_size = st.sidebar.slider("Step size", 0.1, 2.0, 0.5)
        num_neighbors = st.sidebar.slider("Number of neighbors", 5, 50, 15)
        algo = HillClimbing(max_iters=max_iters, step_size=step_size, num_neighbors=num_neighbors)
        
    elif algo_name == "ArtificialBeeColony":
        popsize = st.sidebar.slider("Population size", 10, 100, 20)
        max_gens = st.sidebar.slider("Max generations", 10, 200, 50)
        algo = ArtificialBeeColony(popsize=popsize, gen=max_gens)
        
    elif algo_name == "CuckooOptimization":
        num_nests = st.sidebar.slider("Number of nests", 5, 100, 20)
        max_iters = st.sidebar.slider("Max iterations", 10, 200, 50)
        algo = CuckooOptimization(num_nests=num_nests, max_iters=max_iters)
        
    elif algo_name == "FireflyAlgorithm":
        popsize = st.sidebar.slider("Population size", 10, 100, 20)
        gen = st.sidebar.slider("Max generations", 10, 200, 50)
        algo = FireflyAlgorithm(popsize=popsize, gen=gen)
        
    elif algo_name == "TabuSearch":
        max_iters = st.sidebar.slider("Max iterations", 10, 200, 50)
        tabu_tenure = st.sidebar.slider("Tabu tenure", 5, 50, 10)
        num_neighbors = st.sidebar.slider("Number of neighbors", 5, 50, 20)
        algo = TabuSearch(max_iters=max_iters, tabu_tenure=tabu_tenure, num_neighbors=num_neighbors, step_size=0.5)

    elif algo_name == "GravitationalSearchAlgorithm":
        pop_size = st.sidebar.slider("Population size", 10, 100, 30)
        max_iters = st.sidebar.slider("Max iterations", 10, 200, 50)
        algo = GravitationalSearchAlgorithm(pop_size=pop_size, max_iters=max_iters, G0=100.0, alpha=20.0)

    elif algo_name == "HarmonySearch":
        max_iters = st.sidebar.slider("Max iterations", 10, 200, 50)
        hmcr = st.sidebar.slider("HMCR", 0.1, 1.0, 0.9)
        algo = HarmonySearch(max_iters=max_iters, hmcr=hmcr, par=0.3, bw=0.1)

    elif algo_name == "TLBO":
        pop_size = st.sidebar.slider("Population size", 10, 100, 20)
        max_iters = st.sidebar.slider("Max iterations", 10, 200, 50)
        algo = TLBO(pop_size=pop_size, max_iters=max_iters)

    elif algo_name == "DifferentialEvolution":
        popsize = st.sidebar.slider("Population size", 10, 100, 20)
        gen = st.sidebar.slider("Max generations", 10, 200, 50)
        algo = DifferentialEvolution(popsize=popsize, gen=gen)

    elif algo_name == "GeneticAlgorithm":
        size = st.sidebar.slider("Population size", 10, 100, 20)
        gen = st.sidebar.slider("Max generations", 10, 200, 50)
        algo = GeneticAlgorithm(size=size, gen=gen, crossover_type='multi_point', mutation_type='gaussian')

else:
    problem_name = st.sidebar.selectbox("Select Problem", 
        ["Shortest Path", "Traveling Salesman", "Knapsack", "Graph Coloring"]
    )
    
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
    
    algo_name = st.sidebar.selectbox("Select Algorithm", [
        "ACO", "HillClimbing", "SimulatedAnnealing", 
        "TabuSearch", "BFS", "DFS", "UCS", "Greedy BFS", "A*"
    ])
    
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

    elif algo_name == "TabuSearch":
        max_iters = st.sidebar.slider("Max iterations", 10, 200, 50)
        tabu_tenure = st.sidebar.slider("Tabu tenure", 5, 50, 10)
        num_neighbors = st.sidebar.slider("Number of neighbors", 5, 50, 20)
        algo = TabuSearch(max_iters=max_iters, tabu_tenure=tabu_tenure, num_neighbors=num_neighbors, step_size=1)

    elif algo_name == "BFS": algo = BFS()
    elif algo_name == "DFS": algo = DFS()
    elif algo_name == "UCS": algo = UCS()
    elif algo_name == "Greedy BFS": algo = GreedyBFS()
    elif algo_name == "A*": algo = AStar()

st.sidebar.header("3. Interface Settings")
delay = st.sidebar.slider("Animation speed (seconds/frame)", 0.0, 0.5, 0.1)
run_clicked = st.sidebar.button("Run Algorithm", type="primary", use_container_width=True)

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
    return [(50 + 40*math.cos(2*math.pi*i/n), 50 + 40*math.sin(2*math.pi*i/n)) for i in range(n)]

# ==========================================
# 4. RUN ALGORITHM & ANIMATION
# ==========================================
if run_clicked:
    generator = algo.run(problem)

    info_placeholder = st.empty()
    plot_placeholder = st.empty()

    fig, ax = plt.subplots(figsize=(10, 3.5)) 
    
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
            coords = getattr(problem, 'coords', get_circle_coords(problem.size))
            x_coords, y_coords = zip(*coords)
            
            # Vẽ các cạnh nền (đường đi có thể đi được)
            if hasattr(problem, 'adj'):
                # Dành cho Shortest Path hoặc các đồ thị có danh sách kề cụ thể
                for i in range(len(problem.adj)):
                    for j in problem.adj[i]:
                        if i < j:
                            ax.plot([coords[i][0], coords[j][0]], [coords[i][1], coords[j][1]], 
                                    c='gray', alpha=0.15, linewidth=0.8, zorder=1)
            elif problem_name == "Traveling Salesman":
                # TSP thường là đồ thị đầy đủ, vẽ tất cả các cặp nối với nhau
                for i in range(problem.size):
                    for j in range(i + 1, problem.size):
                        ax.plot([coords[i][0], coords[j][0]], [coords[i][1], coords[j][1]], 
                                c='dimgray', alpha=0.3, linewidth=1, zorder=1)
            
            node_scatter = ax.scatter(x_coords, y_coords, c='darkslategrey', s=100, zorder=3, edgecolors='white')
            for i, (x, y) in enumerate(coords):
                ax.text(x + 1.2, y + 1.2, f"{i}", fontsize=9, fontweight='bold', zorder=6)
            
            if problem_name == "Shortest Path":
                ax.scatter(coords[0][0], coords[0][1], c='lime', s=180, edgecolors='black', 
                           linewidth=1.5, zorder=5, label='START')
                ax.scatter(coords[-1][0], coords[-1][1], c='red', s=250, marker='*', 
                           edgecolors='black', linewidth=1.5, zorder=5, label='END')
                ax.legend(loc='upper right')
                
            path_line, = ax.plot([], [], c='#FF4B4B', linewidth=3, marker='o', markersize=5, zorder=4)
            
        elif problem_name == "Knapsack":
            bars = ax.bar(range(problem.size), problem.profits, color='lightgray', edgecolor='black', alpha=0.8)
            for i, (p, w) in enumerate(zip(problem.profits, problem.weights)):
                ax.text(i, p + 0.1, f"w:{w}", ha='center', fontsize=8, fontweight='bold')
            ax.set_xticks(range(problem.size))
            ax.set_ylabel('Profit')
            ax.set_xlabel('Items (Weight is shown above bars)')

    # --- ANIMATION LOOP ---
    for step_data in generator:
        iteration = step_data.get('iteration', step_data.get('generation', 0))
        best_score = step_data.get('best_score', float('inf'))
        best_solution = step_data.get('best_solution', [])
        
        extra_info = ""
        
        if problem_type == "Continuous":
            pos = np.array(step_data.get('population_positions', step_data.get('population', [])))
            if len(pos) > 0: scatter.set_offsets(pos[:, :2])
            if best_solution is not None and len(best_solution) >= 2:
                best_scatter.set_offsets([best_solution[0], best_solution[1]])
        else:
            if best_solution and len(best_solution) > 0:
                if problem_name in ["Shortest Path", "Traveling Salesman"]:
                    try:
                        # 1. Chuyển đổi ID node sang tọa độ x, y
                        if isinstance(best_solution[0], (list, tuple, np.ndarray)):
                            px = [p[0] for p in best_solution]
                            py = [p[1] for p in best_solution]
                        else:
                            # Ép kiểu n sang int để tránh lỗi chỉ số nếu algo trả về float
                            px = [coords[int(n)][0] for n in best_solution]
                            py = [coords[int(n)][1] for n in best_solution]

                        # 2. Xử lý khép kín chu trình cho TSP
                        if problem_name == "Traveling Salesman" and len(best_solution) > 1:
                            # Chỉ khép kín nếu điểm cuối khác điểm đầu
                            if best_solution[0] != best_solution[-1]:
                                px.append(px[0])
                                py.append(py[0])

                        # 3. Cập nhật dữ liệu cho đường kẻ
                        path_line.set_data(px, py)

                        # Cập nhật thông tin bổ sung: số lượng thành phố đã đi qua
                        extra_info = f" | Visited: **{len(best_solution)}/{problem.size}**"
                    except Exception as e:
                        st.error(f"Error mapping path: {e}")
                    
                elif problem_name == "Graph Coloring":
                    num_colors = len(set(best_solution))
                    extra_info = f" | Colors Used: **{num_colors}**"
                    cmap = plt.get_cmap('gist_ncar')
                    max_c = max(max(best_solution), 1)
                    node_colors = [cmap(c / max_c) for c in best_solution]
                    while len(node_colors) < problem.size: node_colors.append((0.8, 0.8, 0.8, 1.0))
                    node_scatter.set_color(node_colors)
                    
                elif problem_name == "Knapsack":
                    current_w = sum(problem.weights[i] for i, v in enumerate(best_solution) if v == 1)
                    extra_info = f" | Total Weight: **{current_w}/{problem.limit}**"
                    for i, bar in enumerate(bars):
                        bar.set_color('forestgreen' if i < len(best_solution) and best_solution[i] == 1 else 'lightgray')

        plot_placeholder.pyplot(fig)
        info_placeholder.info(f"Iteration: **{iteration}** | Best Score: **{best_score:.6f}**" + extra_info)
        time.sleep(delay)
        
    st.success(f"Successfully completed running {algo_name} algorithm!")