import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

# -----------------------------------------------------------------------
# Synchronize Imports with your main.py
# -----------------------------------------------------------------------
from algorithms import *
from problems import *

# Streamlit page configuration
st.set_page_config(page_title="Optimization Visualizer", layout="wide")
st.title("Optimization Algorithms: Step-by-Step Visualization")

# ==========================================
# 1. SIDEBAR - SELECT PROBLEM
# ==========================================
st.sidebar.header("1. Problem Settings (Continuous)")
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

# Fix dimension to dim=2 to draw 2D contour plots
bound_val = 5.0
problem = problems_dict[problem_name](dim=2, bound=bound_val)


# ==========================================
# 2. SIDEBAR - SELECT ALGORITHM & PARAMETERS
# ==========================================
st.sidebar.header("2. Algorithm Settings")
algo_name = st.sidebar.selectbox("Select Algorithm", ["PSO", "SimulatedAnnealing", "HillClimbing"])

# Render dynamic sliders based on the selected algorithm
if algo_name == "PSO":
    num_particles = st.sidebar.slider("Number of particles (num_particles)", 10, 100, 20)
    max_iters = st.sidebar.slider("Max iterations (max_iters)", 10, 200, 50)
    algo = PSO(num_particles=num_particles, max_iters=max_iters)

elif algo_name == "SimulatedAnnealing":
    max_epochs = st.sidebar.slider("Max epochs (max_epochs)", 10, 200, 50)
    initial_temp = st.sidebar.slider("Initial temperature (initial_temp)", 10.0, 500.0, 100.0)
    cooling_rate = st.sidebar.slider("Cooling rate (cooling_rate)", 0.8, 0.999, 0.99)
    step_size = st.sidebar.slider("Step size (step_size)", 0.1, 2.0, 0.5)
    markov_chain_length = st.sidebar.slider("Markov Chain Length", 5, 50, 20)
    algo = SimulatedAnnealing(
        max_epochs=max_epochs, initial_temp=initial_temp, 
        cooling_rate=cooling_rate, step_size=step_size, 
        markov_chain_length=markov_chain_length
    )

elif algo_name == "HillClimbing":
    max_iters = st.sidebar.slider("Max iterations (max_iters)", 10, 200, 50)
    step_size = st.sidebar.slider("Step size (step_size)", 0.1, 2.0, 0.5)
    num_neighbors = st.sidebar.slider("Number of neighbors (num_neighbors)", 5, 50, 15)
    algo = HillClimbing(max_iters=max_iters, step_size=step_size, num_neighbors=num_neighbors)

st.sidebar.header("3. Interface Settings")
delay = st.sidebar.slider("Animation speed (seconds/frame)", 0.0, 0.5, 0.1)


# ==========================================
# 3. CONTOUR PLOT GENERATION FUNCTION
# ==========================================
@st.cache_data
def get_contour_data(prob_name, bound):
    """Contour calculation is time-consuming, caching it prevents UI lag."""
    x = np.linspace(-bound, bound, 100)
    y = np.linspace(-bound, bound, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    temp_prob = problems_dict[prob_name](dim=2, bound=bound)
    for i in range(100):
        for j in range(100):
            Z[i, j] = temp_prob.evaluate([X[i, j], Y[i, j]])
    return X, Y, Z

X, Y, Z = get_contour_data(problem_name, bound_val)


# ==========================================
# 4. RUN ALGORITHM & ANIMATION
# ==========================================
if st.button("Run Algorithm", type="primary"):
    generator = algo.run(problem)

    # Placeholders for updating the UI frame-by-frame
    info_placeholder = st.empty()
    plot_placeholder = st.empty()

    # TÙY CHỈNH TẠI ĐÂY: Dùng tỷ lệ 10:4 để đồ thị "dẹt" hơn, triệt tiêu thanh cuộn dọc
    fig, ax = plt.subplots(figsize=(10, 4)) 
    
    contour = ax.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.8)
    fig.colorbar(contour, ax=ax, label="Fitness Value")
    
    # Scatter plot for individuals (particles/neighbors) and the best individual
    scatter = ax.scatter([], [], c='red', edgecolors='white', s=50, zorder=5, label='Current Solutions')
    best_scatter = ax.scatter([], [], marker='*', c='gold', edgecolors='black', s=250, zorder=10, label='Global Best')
    
    ax.set_title(f"{algo_name} running on {problem_name} function (2D)")
    ax.set_xlim(-bound_val, bound_val)
    ax.set_ylim(-bound_val, bound_val)
    ax.legend(loc="upper right")

    # Iterate through each step (yield) of the algorithm
    for step_data in generator:
        iteration = step_data.get('iteration', 0)
        best_score = step_data.get('best_score', 0.0)
        best_solution = step_data.get('best_solution', [0.0, 0.0])
        
        # Flexible handling: Draw all if population (PSO, GA), draw 1 point if single (SA, HillClimbing)
        if 'population_positions' in step_data:
            positions = np.array(step_data['population_positions'])
            scatter.set_offsets(positions[:, :2])
        elif 'current_solution' in step_data:
            current_sol = np.array(step_data['current_solution']).reshape(1, 2)
            scatter.set_offsets(current_sol[:, :2])
            
        # Update coordinates of the best position
        best_scatter.set_offsets([best_solution[0], best_solution[1]])
        
        # Push plot and text to UI
        plot_placeholder.pyplot(fig)
        info_placeholder.info(f"Iteration: **{iteration}** | Best Fitness: **{best_score:.6f}**")
        
        time.sleep(delay)
        
    st.success(f"Successfully completed running {algo_name} algorithm!")