# Brusselator Model Solver Comparison GUI
# A Streamlit application for comparing different numerical solvers

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import time
import sys
import os
import pandas as pd

# Add parent directory to path for imports
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PARENT_DIR)

# Import Euler solver
from Regular_Euler.regular_euler import brusselator_euler as euler_solve

# Import Improved Euler solver
from Improved_Euler.improved_euler import brusselator_euler as improved_euler_solve

# Import RK4 solver
from RK4.RK4 import brusselator_rk4 as rk4_solve

# Import PINN solver
from PINN.pinn_model import BrusselatorPINNSolver

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Brusselator Solver Comparison",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM STYLING - Light/Dark Mode Compatible
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    
    /* Dark mode (default) */
    :root {
        --bg-primary: #0a0a0f;
        --bg-secondary: #12121a;
        --bg-card: #1a1a24;
        --bg-card-end: #1f1f2e;
        --accent-cyan: #00d4ff;
        --accent-magenta: #ff00aa;
        --accent-yellow: #ffdd00;
        --accent-green: #00ff88;
        --text-primary: #e8e8f0;
        --text-secondary: #8888a0;
        --border-color: #2a2a3a;
        --shadow-color: rgba(0, 0, 0, 0.3);
        --equation-bg: #1a1a2e;
        --equation-bg-end: #16162a;
    }
    
    /* Light mode overrides */
    @media (prefers-color-scheme: light) {
        :root {
            --bg-primary: #f8f9fa;
            --bg-secondary: #ffffff;
            --bg-card: #ffffff;
            --bg-card-end: #f0f2f5;
            --accent-cyan: #0099cc;
            --accent-magenta: #cc0088;
            --accent-yellow: #cc9900;
            --accent-green: #00aa55;
            --text-primary: #1a1a2e;
            --text-secondary: #5a5a7a;
            --border-color: #d0d0e0;
            --shadow-color: rgba(0, 0, 0, 0.1);
            --equation-bg: #f0f4f8;
            --equation-bg-end: #e8ecf0;
        }
    }
    
    /* Streamlit theme detection - override for light theme */
    [data-theme="light"], 
    .stApp[data-theme="light"],
    html[data-theme="light"] :root {
        --bg-primary: #f8f9fa;
        --bg-secondary: #ffffff;
        --bg-card: #ffffff;
        --bg-card-end: #f0f2f5;
        --accent-cyan: #0099cc;
        --accent-magenta: #cc0088;
        --accent-yellow: #cc9900;
        --accent-green: #00aa55;
        --text-primary: #1a1a2e;
        --text-secondary: #5a5a7a;
        --border-color: #d0d0e0;
        --shadow-color: rgba(0, 0, 0, 0.1);
        --equation-bg: #f0f4f8;
        --equation-bg-end: #e8ecf0;
    }
    
    .main-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, var(--accent-cyan) 0%, var(--accent-magenta) 50%, var(--accent-yellow) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .sub-header {
        font-family: 'JetBrains Mono', monospace;
        color: var(--text-secondary);
        text-align: center;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(145deg, var(--bg-card) 0%, var(--bg-card-end) 100%);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 20px var(--shadow-color);
    }
    
    .metric-label {
        font-family: 'JetBrains Mono', monospace;
        color: var(--text-secondary);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.3rem;
    }
    
    .metric-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text-primary);
    }
    
    .solver-euler { border-left: 4px solid #ff6b6b; }
    .solver-improved { border-left: 4px solid #4ecdc4; }
    .solver-rk4 { border-left: 4px solid #ffe66d; }
    .solver-pinn { border-left: 4px solid #c792ea; }
    .solver-ground { border-left: 4px solid #00ff88; }
    
    .solver-config-card {
        background: linear-gradient(145deg, var(--bg-card) 0%, var(--bg-card-end) 100%);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .equation-box {
        background: linear-gradient(145deg, var(--equation-bg) 0%, var(--equation-bg-end) 100%);
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 1.5rem;
        font-family: 'JetBrains Mono', monospace;
        text-align: center;
        margin: 1rem 0;
        color: var(--text-primary);
    }
    
    .accuracy-excellent { color: #00cc66; }
    .accuracy-good { color: #ccaa00; }
    .accuracy-moderate { color: #ff8800; }
    .accuracy-poor { color: #ff5555; }
    
    .stSelectbox > div > div {
        background-color: var(--bg-card);
        border-color: var(--border-color);
    }
    
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--bg-secondary) 0%, var(--bg-primary) 100%);
        border-right: 1px solid var(--border-color);
    }
    
    section[data-testid="stSidebar"] .stMarkdown h2 {
        color: var(--accent-cyan);
        font-family: 'Space Grotesk', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# THEME DETECTION FOR MATPLOTLIB
def get_plot_theme():
    """Detect theme and return appropriate plot colors"""
    # Try to detect Streamlit theme from config
    try:
        theme_base = st.get_option("theme.base")
        is_dark = theme_base != "light"
    except:
        # Default to dark theme if detection fails
        is_dark = True
    
    if is_dark:
        return {
            "bg_primary": "#0a0a0f",
            "bg_secondary": "#12121a",
            "text_color": "#e8e8f0",
            "text_secondary": "#8888a0",
            "grid_color": "#3a3a5a",
            "border_color": "#2a2a3a",
            "legend_bg": "#1a1a24"
        }
    else:
        return {
            "bg_primary": "#ffffff",
            "bg_secondary": "#f8f9fa",
            "text_color": "#1a1a2e",
            "text_secondary": "#5a5a7a",
            "grid_color": "#d0d0e0",
            "border_color": "#c0c0d0",
            "legend_bg": "#ffffff"
        }

PLOT_THEME = get_plot_theme()

# HEADER
st.markdown('<h1 class="main-header">🧪 Brusselator Solver Comparison</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Compare numerical methods for the Brusselator reaction-diffusion system</p>', unsafe_allow_html=True)

# Brusselator equations display
st.markdown("""
<div class="equation-box">
    <span style="color: #00d4ff;">dx/dt</span> = A + x²y - (B+1)x &nbsp;&nbsp;&nbsp;&nbsp;
    <span style="color: #ff00aa;">dy/dt</span> = Bx - x²y
</div>
""", unsafe_allow_html=True)

# SIDEBAR - PARAMETERS AND SOLVER SELECTION
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    st.markdown("### 📊 Model Parameters")
    
    # Parameter A
    A = st.slider(
        "Parameter A",
        min_value=0.5,
        max_value=2.5,
        value=1.0,
        step=0.1,
        help="Brusselator parameter A (valid range: 0.5 - 2.5)"
    )
    
    # Parameter B
    B = st.slider(
        "Parameter B",
        min_value=1.0,
        max_value=6.0,
        value=3.0,
        step=0.1,
        help="Brusselator parameter B (valid range: 1.0 - 6.0)"
    )
    
    st.markdown("### 🎯 Initial Conditions")
    
    # Initial x0
    x0 = st.slider(
        "Initial x₀",
        min_value=0.0,
        max_value=3.0,
        value=1.0,
        step=0.1,
        help="Initial condition for x (valid range: 0.0 - 3.0)"
    )
    
    # Initial y0
    y0 = st.slider(
        "Initial y₀",
        min_value=0.0,
        max_value=3.0,
        value=1.0,
        step=0.1,
        help="Initial condition for y (valid range: 0.0 - 3.0)"
    )
    
    st.markdown("### 🔬 Solver Selection")
    
    solvers_available = ["Euler", "Improved Euler", "RK4", "PINN"]
    selected_solvers = st.multiselect(
        "Select Solvers to Compare",
        options=solvers_available,
        default=["Euler", "RK4"],
        help="Select 2-4 solvers to compare"
    )
    
    # Validation
    if len(selected_solvers) < 2:
        st.warning("⚠️ Please select at least 2 solvers to compare")
    
    st.markdown("---")
    st.markdown("### 📌 Fixed Parameters")
    st.markdown("**Time Range:** 0 - 20 seconds")
    st.markdown("**Ground Truth:** RK4 @ dt=0.001")

# CONSTANTS AND HELPER FUNCTIONS
T = 20.0  # Fixed time range
GROUND_TRUTH_DT = 0.001  # High accuracy reference

# Color mapping for solvers
SOLVER_COLORS = {
    "Euler": "#ff6b6b",
    "Improved Euler": "#4ecdc4", 
    "RK4": "#ffe66d",
    "PINN": "#c792ea",
    "Ground Truth": "#00ff88"
}

SOLVER_STYLES = {
    "Euler": "-",
    "Improved Euler": "--",
    "RK4": "-.",
    "PINN": ":",
    "Ground Truth": "-"
}

DT_OPTIONS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]

@st.cache_resource
def load_pinn_model():
    """Load the PINN model (cached)"""
    model_path = os.path.join(PARENT_DIR, "PINN", "brusselator_pinn.pth")
    if os.path.exists(model_path):
        return BrusselatorPINNSolver(model_path)
    return None

def run_solver(solver_name, A, B, x0, y0, dt, T):
    """Run a solver and return results with timing info"""
    if solver_name == "Euler":
        start = time.perf_counter()
        t, x, y, step_times = euler_solve(A, B, x0, y0, dt, T, record_step_times=True)
        total_time = time.perf_counter() - start
        n_steps = len(t) - 1
        return {
            "name": solver_name,
            "t": t,
            "x": x,
            "y": y,
            "total_time": total_time,
            "n_steps": n_steps,
            "time_per_step": total_time / n_steps if n_steps > 0 else 0,
            "step_times": step_times,
            "dt": dt
        }
    
    elif solver_name == "Improved Euler":
        start = time.perf_counter()
        t, x, y, step_times = improved_euler_solve(A, B, x0, y0, dt, T, record_step_times=True)
        total_time = time.perf_counter() - start
        n_steps = len(t) - 1
        return {
            "name": solver_name,
            "t": t,
            "x": x,
            "y": y,
            "total_time": total_time,
            "n_steps": n_steps,
            "time_per_step": total_time / n_steps if n_steps > 0 else 0,
            "step_times": step_times,
            "dt": dt
        }
    
    elif solver_name == "RK4":
        start = time.perf_counter()
        t, x, y, step_times = rk4_solve(A, B, x0, y0, dt, T, record_step_times=True)
        total_time = time.perf_counter() - start
        n_steps = len(t) - 1
        return {
            "name": solver_name,
            "t": t,
            "x": x,
            "y": y,
            "total_time": total_time,
            "n_steps": n_steps,
            "time_per_step": total_time / n_steps if n_steps > 0 else 0,
            "step_times": step_times,
            "dt": dt
        }
    
    elif solver_name == "PINN":
        pinn = load_pinn_model()
        if pinn is None:
            return None
        t, x, y, elapsed = pinn.solve(A, B, x0, y0, T, n_points=1000)
        return {
            "name": solver_name,
            "t": t,
            "x": x,
            "y": y,
            "total_time": elapsed,
            "n_steps": None,
            "time_per_step": None,
            "step_times": None,
            "dt": None
        }
    
    return None

def compute_ground_truth(A, B, x0, y0, T):
    """Compute ground truth using RK4 at very small dt"""
    start = time.perf_counter()
    t, x, y, _ = rk4_solve(A, B, x0, y0, GROUND_TRUTH_DT, T, record_step_times=False)
    elapsed = time.perf_counter() - start
    return {
        "name": "Ground Truth",
        "t": t,
        "x": x,
        "y": y,
        "total_time": elapsed,
        "n_steps": len(t) - 1,
        "dt": GROUND_TRUTH_DT
    }

def compute_mse(result, ground_truth):
    """Compute MSE between a solver result and ground truth"""
    # Interpolate solver result to ground truth time points
    x_interp = np.interp(ground_truth['t'], result['t'], result['x'])
    y_interp = np.interp(ground_truth['t'], result['t'], result['y'])
    
    # Compute MSE
    mse_x = np.mean((x_interp - ground_truth['x']) ** 2)
    mse_y = np.mean((y_interp - ground_truth['y']) ** 2)
    mse_total = (mse_x + mse_y) / 2
    
    return {
        "mse_x": mse_x,
        "mse_y": mse_y,
        "mse_total": mse_total,
        "rmse_total": np.sqrt(mse_total)
    }

def get_accuracy_class(mse):
    """Get CSS class based on MSE value"""
    if mse < 1e-6:
        return "accuracy-excellent"
    elif mse < 1e-4:
        return "accuracy-good"
    elif mse < 1e-2:
        return "accuracy-moderate"
    else:
        return "accuracy-poor"

def get_accuracy_label(mse):
    """Get label based on MSE value"""
    if mse < 1e-6:
        return "Excellent"
    elif mse < 1e-4:
        return "Good"
    elif mse < 1e-2:
        return "Moderate"
    else:
        return "Poor"

# MAIN CONTENT
if len(selected_solvers) >= 2:
    
    st.markdown("---")
    st.markdown("## 🛠️ Solver Configuration")
    st.markdown("Configure the time step for each traditional solver:")
    
    # Create solver configuration cards on main page
    solver_dts = {}
    
    # Determine number of columns based on selected solvers
    n_traditional = sum(1 for s in selected_solvers if s != "PINN")
    n_pinn = 1 if "PINN" in selected_solvers else 0
    
    cols = st.columns(len(selected_solvers))
    
    for i, solver_name in enumerate(selected_solvers):
        with cols[i]:
            color = SOLVER_COLORS[solver_name]
            
            if solver_name == "PINN":
                st.markdown(f"""
                <div class="solver-config-card" style="border-left: 4px solid {color};">
                    <h4 style="color: {color}; margin: 0;">🧠 {solver_name}</h4>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("**Method:** Neural Network")
                st.markdown("**Points:** 1,000 (fixed)")
                st.markdown("*No time step needed*")
                solver_dts[solver_name] = None
            else:
                st.markdown(f"""
                <div class="solver-config-card" style="border-left: 4px solid {color};">
                    <h4 style="color: {color}; margin: 0;">📊 {solver_name}</h4>
                </div>
                """, unsafe_allow_html=True)
                
                dt = st.select_slider(
                    f"Time Step (dt)",
                    options=DT_OPTIONS,
                    value=0.01,
                    key=f"dt_{solver_name}",
                    help=f"Time step for {solver_name} method"
                )
                solver_dts[solver_name] = dt
                
                n_steps = int(np.ceil(T / dt))
                st.markdown(f"**Steps:** {n_steps:,}")
    
    st.markdown("---")
    
    # Run button
    if st.button("🚀 Run Comparison", type="primary", width="stretch"):
        
        results = []
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # First, compute ground truth
        status_text.text("Computing ground truth (RK4 @ dt=0.001)...")
        ground_truth = compute_ground_truth(A, B, x0, y0, T)
        progress_bar.progress(0.2)
        
        # Run each selected solver
        total_solvers = len(selected_solvers)
        for i, solver_name in enumerate(selected_solvers):
            status_text.text(f"Running {solver_name}...")
            dt = solver_dts.get(solver_name, 0.01)
            result = run_solver(solver_name, A, B, x0, y0, dt, T)
            if result is not None:
                # Compute accuracy metrics
                accuracy = compute_mse(result, ground_truth)
                result['accuracy'] = accuracy
                results.append(result)
            progress_bar.progress(0.2 + 0.8 * (i + 1) / total_solvers)
        
        status_text.text("Complete!")
        time.sleep(0.3)
        progress_bar.empty()
        status_text.empty()
        
        if results:
            # Store results in session state
            st.session_state['results'] = results
            st.session_state['ground_truth'] = ground_truth
            st.session_state['params'] = {'A': A, 'B': B, 'x0': x0, 'y0': y0}
            st.session_state['solver_dts'] = solver_dts
    
    # Display results if available
    if 'results' in st.session_state and 'ground_truth' in st.session_state:
        results = st.session_state['results']
        ground_truth = st.session_state['ground_truth']
        params = st.session_state['params']
        solver_dts = st.session_state['solver_dts']
        
        # PERFORMANCE METRICS
        st.markdown("## ⏱️ Performance Metrics")
        
        cols = st.columns(len(results))
        
        for i, result in enumerate(results):
            with cols[i]:
                solver_class = result['name'].lower().replace(" ", "-").split('-')[0]
                color = SOLVER_COLORS[result['name']]
                
                st.markdown(f"""
                <div class="metric-card solver-{solver_class}">
                    <div class="metric-label">{result['name']}</div>
                    <div class="metric-value">{result['total_time']*1000:.3f} ms</div>
                </div>
                """, unsafe_allow_html=True)
                
                if result['n_steps'] is not None:
                    st.markdown(f"**Steps:** {result['n_steps']:,}")
                    st.markdown(f"**Time/Step:** {result['time_per_step']*1e6:.2f} μs")
                    st.markdown(f"**dt:** {result['dt']}")
                else:
                    st.markdown("**Method:** Neural Network")
                    st.markdown("**Points:** 1,000")
        
        # ACCURACY METRICS
        st.markdown("---")
        st.markdown("## 🎯 Accuracy Metrics")
        st.markdown(f"*Compared against ground truth: RK4 @ dt={GROUND_TRUTH_DT} ({ground_truth['n_steps']:,} steps)*")
        
        cols = st.columns(len(results))
        
        for i, result in enumerate(results):
            with cols[i]:
                acc = result['accuracy']
                acc_class = get_accuracy_class(acc['mse_total'])
                acc_label = get_accuracy_label(acc['mse_total'])
                color = SOLVER_COLORS[result['name']]
                
                st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid {color};">
                    <div class="metric-label">{result['name']} - MSE</div>
                    <div class="metric-value {acc_class}">{acc['mse_total']:.2e}</div>
                    <div style="color: var(--text-secondary); font-size: 0.85rem;">Rating: <span class="{acc_class}">{acc_label}</span></div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"**MSE (x):** {acc['mse_x']:.2e}")
                st.markdown(f"**MSE (y):** {acc['mse_y']:.2e}")
                st.markdown(f"**RMSE:** {acc['rmse_total']:.2e}")
        
        # SOLUTION COMPARISON WITH GROUND TRUTH
        st.markdown("---")
        st.markdown("## 📈 Solution Comparison")
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📊 All Solvers", "🔍 Individual vs Ground Truth", "🌀 Phase Portrait"])
        
        with tab1:
            # All solvers comparison
            theme = get_plot_theme()
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.patch.set_facecolor(theme["bg_primary"])
            
            for ax in axes:
                ax.set_facecolor(theme["bg_secondary"])
                ax.tick_params(colors=theme["text_secondary"])
                for spine in ax.spines.values():
                    spine.set_color(theme["border_color"])
                ax.xaxis.label.set_color(theme["text_color"])
                ax.yaxis.label.set_color(theme["text_color"])
                ax.title.set_color(theme["text_color"])
                ax.grid(True, alpha=0.3, color=theme["grid_color"])
            
            # Plot ground truth
            axes[0].plot(ground_truth['t'], ground_truth['x'], 
                        color=SOLVER_COLORS['Ground Truth'], 
                        linewidth=2.5, 
                        label='Ground Truth',
                        alpha=0.5)
            axes[1].plot(ground_truth['t'], ground_truth['y'], 
                        color=SOLVER_COLORS['Ground Truth'], 
                        linewidth=2.5, 
                        label='Ground Truth',
                        alpha=0.5)
            
            # Plot each solver
            for result in results:
                axes[0].plot(result['t'], result['x'], 
                            color=SOLVER_COLORS[result['name']], 
                            linestyle=SOLVER_STYLES[result['name']],
                            linewidth=2, 
                            label=result['name'],
                            alpha=0.9)
                axes[1].plot(result['t'], result['y'], 
                            color=SOLVER_COLORS[result['name']], 
                            linestyle=SOLVER_STYLES[result['name']],
                            linewidth=2, 
                            label=result['name'],
                            alpha=0.9)
            
            axes[0].set_xlabel('Time (t)', fontsize=11)
            axes[0].set_ylabel('x(t)', fontsize=11)
            axes[0].set_title('Concentration x vs Time', fontsize=12, fontweight='bold')
            axes[0].legend(facecolor=theme["legend_bg"], edgecolor=theme["border_color"], labelcolor=theme["text_color"])
            
            axes[1].set_xlabel('Time (t)', fontsize=11)
            axes[1].set_ylabel('y(t)', fontsize=11)
            axes[1].set_title('Concentration y vs Time', fontsize=12, fontweight='bold')
            axes[1].legend(facecolor=theme["legend_bg"], edgecolor=theme["border_color"], labelcolor=theme["text_color"])
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with tab2:
            # Individual solver vs ground truth
            theme = get_plot_theme()
            n_results = len(results)
            fig, axes = plt.subplots(n_results, 2, figsize=(14, 4 * n_results))
            fig.patch.set_facecolor(theme["bg_primary"])
            
            if n_results == 1:
                axes = axes.reshape(1, -1)
            
            for idx, result in enumerate(results):
                for col in range(2):
                    ax = axes[idx, col]
                    ax.set_facecolor(theme["bg_secondary"])
                    ax.tick_params(colors=theme["text_secondary"])
                    for spine in ax.spines.values():
                        spine.set_color(theme["border_color"])
                    ax.xaxis.label.set_color(theme["text_color"])
                    ax.yaxis.label.set_color(theme["text_color"])
                    ax.title.set_color(theme["text_color"])
                    ax.grid(True, alpha=0.3, color=theme["grid_color"])
                
                # Plot x comparison
                axes[idx, 0].plot(ground_truth['t'], ground_truth['x'], 
                                 color=SOLVER_COLORS['Ground Truth'], 
                                 linewidth=2, label='Ground Truth', alpha=0.7)
                axes[idx, 0].plot(result['t'], result['x'], 
                                 color=SOLVER_COLORS[result['name']], 
                                 linewidth=2, label=result['name'], linestyle='--')
                axes[idx, 0].set_xlabel('Time (t)')
                axes[idx, 0].set_ylabel('x(t)')
                axes[idx, 0].set_title(f"{result['name']} - x(t) | MSE: {result['accuracy']['mse_x']:.2e}", 
                                       fontsize=11, fontweight='bold')
                axes[idx, 0].legend(facecolor=theme["legend_bg"], edgecolor=theme["border_color"], labelcolor=theme["text_color"])
                
                # Plot y comparison
                axes[idx, 1].plot(ground_truth['t'], ground_truth['y'], 
                                 color=SOLVER_COLORS['Ground Truth'], 
                                 linewidth=2, label='Ground Truth', alpha=0.7)
                axes[idx, 1].plot(result['t'], result['y'], 
                                 color=SOLVER_COLORS[result['name']], 
                                 linewidth=2, label=result['name'], linestyle='--')
                axes[idx, 1].set_xlabel('Time (t)')
                axes[idx, 1].set_ylabel('y(t)')
                axes[idx, 1].set_title(f"{result['name']} - y(t) | MSE: {result['accuracy']['mse_y']:.2e}", 
                                       fontsize=11, fontweight='bold')
                axes[idx, 1].legend(facecolor=theme["legend_bg"], edgecolor=theme["border_color"], labelcolor=theme["text_color"])
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with tab3:
            # Phase portrait
            theme = get_plot_theme()
            fig, ax = plt.subplots(figsize=(10, 8))
            fig.patch.set_facecolor(theme["bg_primary"])
            ax.set_facecolor(theme["bg_secondary"])
            ax.tick_params(colors=theme["text_secondary"])
            for spine in ax.spines.values():
                spine.set_color(theme["border_color"])
            ax.xaxis.label.set_color(theme["text_color"])
            ax.yaxis.label.set_color(theme["text_color"])
            ax.title.set_color(theme["text_color"])
            ax.grid(True, alpha=0.3, color=theme["grid_color"])
            
            # Plot ground truth
            ax.plot(ground_truth['x'], ground_truth['y'], 
                   color=SOLVER_COLORS['Ground Truth'], 
                   linewidth=3, label='Ground Truth', alpha=0.4)
            
            # Plot each solver
            for result in results:
                ax.plot(result['x'], result['y'], 
                       color=SOLVER_COLORS[result['name']], 
                       linestyle=SOLVER_STYLES[result['name']],
                       linewidth=2, 
                       label=result['name'],
                       alpha=0.9)
                ax.plot(result['x'][0], result['y'][0], 'o', 
                       color=SOLVER_COLORS[result['name']], 
                       markersize=10)
            
            ax.set_xlabel('x', fontsize=12)
            ax.set_ylabel('y', fontsize=12)
            ax.set_title('Phase Portrait (x vs y)', fontsize=14, fontweight='bold')
            ax.legend(facecolor=theme["legend_bg"], edgecolor=theme["border_color"], labelcolor=theme["text_color"], loc='best')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # DETAILED COMPARISON TABLE
        st.markdown("---")
        st.markdown("## 📊 Detailed Comparison Table")
        
        comparison_data = []
        for result in results:
            # Convert all values to strings for consistent DataFrame serialization
            row = {
                "Solver": str(result['name']),
                "dt": str(result['dt']) if result['dt'] is not None else "-",
                "Steps": str(f"{result['n_steps']:,}") if result['n_steps'] is not None else "-",
                "Total Time (ms)": str(f"{result['total_time']*1000:.4f}"),
                "Time/Step (μs)": str(f"{result['time_per_step']*1e6:.3f}") if result['time_per_step'] is not None else "-",
                "MSE (total)": str(f"{result['accuracy']['mse_total']:.2e}"),
                "MSE (x)": str(f"{result['accuracy']['mse_x']:.2e}"),
                "MSE (y)": str(f"{result['accuracy']['mse_y']:.2e}"),
                "RMSE": str(f"{result['accuracy']['rmse_total']:.2e}"),
                "Accuracy": str(get_accuracy_label(result['accuracy']['mse_total']))
            }
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, hide_index=True)
        
        # PARAMETER SUMMARY
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 📌 Model Parameters")
            st.markdown(f"- **A** = {params['A']}")
            st.markdown(f"- **B** = {params['B']}")
            st.markdown(f"- **x₀** = {params['x0']}")
            st.markdown(f"- **y₀** = {params['y0']}")
        
        with col2:
            st.markdown("### ⚙️ Simulation Settings")
            st.markdown(f"- **Time Range:** 0 - {T} s")
            st.markdown(f"- **Ground Truth dt:** {GROUND_TRUTH_DT}")
            st.markdown(f"- **Ground Truth Steps:** {ground_truth['n_steps']:,}")
        
        with col3:
            st.markdown("### 🎛️ Solver Time Steps")
            for solver, dt in solver_dts.items():
                if dt is not None:
                    st.markdown(f"- **{solver}:** dt = {dt}")
                else:
                    st.markdown(f"- **{solver}:** N/A (neural net)")

else:
    st.info("👈 Select at least 2 solvers from the sidebar to begin comparison")
    
    # Show some info about the solvers
    st.markdown("---")
    st.markdown("## 📚 Available Solvers")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🔢 Traditional Numerical Methods
        
        **Euler's Method**
        - First-order accuracy (O(h))
        - Simplest numerical integrator
        - Fast but less accurate
        
        **Improved Euler (Heun's Method)**
        - Second-order accuracy (O(h²))
        - Predictor-corrector approach
        - Better stability than basic Euler
        
        **RK4 (Runge-Kutta 4th Order)**
        - Fourth-order accuracy (O(h⁴))
        - Industry standard for ODEs
        - Excellent balance of speed and accuracy
        """)
    
    with col2:
        st.markdown("""
        ### 🧠 Machine Learning Method
        
        **PINN (Physics-Informed Neural Network)**
        - Neural network trained on Brusselator physics
        - Learns the solution manifold
        - Fast inference after training
        - Handles parameter variations
        
        **Model Details:**
        - 8 hidden layers (128 neurons each)
        - Inputs: t, A, B, x₀, y₀
        - Outputs: x(t), y(t)
        """)
    
    st.markdown("---")
    st.markdown("### 🎯 Accuracy Evaluation")
    st.markdown("""
    All solvers are compared against a **ground truth** solution computed using **RK4 at dt=0.001**.
    
    This provides ~20,000 steps with 4th-order accuracy, giving an essentially exact reference solution.
    
    **Accuracy Ratings:**
    - 🟢 **Excellent**: MSE < 10⁻⁶
    - 🟡 **Good**: MSE < 10⁻⁴
    - 🟠 **Moderate**: MSE < 10⁻²
    - 🔴 **Poor**: MSE ≥ 10⁻²
    """)
