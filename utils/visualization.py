import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

os.makedirs("results", exist_ok=True)

def plot_convergence_robust(histories_dict, title="Convergence (Mean ± Std)", filename="convergence.html"):
    """Vẽ đồ thị hội tụ kèm vùng Standard Deviation (Robustness)"""
    fig = go.Figure()
    
    for algo_name, runs in histories_dict.items():
        if not runs: continue
        # runs là mảng 2D: [num_runs, max_iters]
        runs_array = np.array(runs)
        mean_hist = np.mean(runs_array, axis=0)
        std_hist = np.std(runs_array, axis=0)
        
        # Thêm 1e-10 để tránh lỗi log(0)
        mean_hist = mean_hist + 1e-10
        upper_bound = mean_hist + std_hist
        lower_bound = np.clip(mean_hist - std_hist, 1e-10, None)
        x_vals = list(range(len(mean_hist)))

        # Vẽ vùng mờ Std
        fig.add_trace(go.Scatter(
            x=x_vals + x_vals[::-1],
            y=list(upper_bound) + list(lower_bound)[::-1],
            fill='toself',
            fillcolor=f'rgba({np.random.randint(0,255)},{np.random.randint(0,255)},200,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        ))
        
        # Vẽ đường Mean
        fig.add_trace(go.Scatter(x=x_vals, y=mean_hist, mode='lines', name=algo_name, line=dict(width=2)))

    fig.update_layout(title=title, xaxis_title="Iterations", yaxis_title="Fitness (Log Scale)", 
                      template="plotly_white", yaxis_type="log")
    fig.write_html(os.path.join("results", filename))

def plot_boxplot_performance(scores_dict, title="Robustness & Quality (Boxplot)", filename="boxplot.html"):
    """Sử dụng Boxplot thay cho Bar chart để thể hiện tính ổn định (Robustness)"""
    data = []
    for algo, scores in scores_dict.items():
        for score in scores:
            data.append({"Algorithm": algo, "Best Fitness": score})
            
    df = pd.DataFrame(data)
    fig = px.box(df, x="Algorithm", y="Best Fitness", color="Algorithm", points="all", title=title)
    fig.update_layout(template="plotly_white", yaxis_type="log")
    fig.write_html(os.path.join("results", filename))

def plot_exploration_exploitation(diversity_history, title="Exploration vs Exploitation", filename="explore_exploit.html"):
    """Vẽ miền phân bổ % Khám phá và Khai thác"""
    fig = go.Figure()
    x_vals = list(range(len(diversity_history)))
    
    # Tính toán dựa trên độ lệch chuẩn của quần thể
    div_array = np.array(diversity_history)
    max_div = np.max(div_array) if np.max(div_array) > 0 else 1
    exploration_pct = (div_array / max_div) * 100
    exploitation_pct = 100 - exploration_pct

    fig.add_trace(go.Scatter(x=x_vals, y=exploration_pct, mode='lines', stackgroup='one', name='Exploration (%)', line=dict(width=0.5, color='rgb(111, 231, 219)')))
    fig.add_trace(go.Scatter(x=x_vals, y=exploitation_pct, mode='lines', stackgroup='one', name='Exploitation (%)', line=dict(width=0.5, color='rgb(131, 90, 241)')))
    
    fig.update_layout(title=title, xaxis_title="Iterations", yaxis_title="Percentage (%)", template="plotly_white")
    fig.write_html(os.path.join("results", filename))
    
# (Giữ nguyên plot_3d_landscape và plot_heatmap của bạn ở đây)