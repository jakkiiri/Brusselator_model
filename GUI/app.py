# Brusselator PINN Solver - Physics-Informed Neural Network Showcase
# Test and validate a trained PINN against traditional numerical methods

import streamlit as st
import numpy as np
import warnings
# Suppress numpy overflow warnings (handled gracefully in code)
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*overflow.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value.*')
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
    page_title="Brusselator PINN Solver",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM STYLING - Neural Network / AI Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    /* Dark mode (default) - Neural Network Theme */
    :root {
        --bg-primary: #0d0d1a;
        --bg-secondary: #13132b;
        --bg-card: #1a1a3e;
        --bg-card-end: #242454;
        --accent-primary: #a855f7;
        --accent-secondary: #6366f1;
        --accent-glow: #c084fc;
        --accent-cyan: #22d3ee;
        --accent-green: #34d399;
        --accent-orange: #fb923c;
        --accent-red: #f87171;
        --text-primary: #f0f0ff;
        --text-secondary: #9898c8;
        --border-color: #3d3d6b;
        --shadow-color: rgba(168, 85, 247, 0.15);
        --pinn-gradient-start: #7c3aed;
        --pinn-gradient-end: #a855f7;
        --neural-glow: 0 0 40px rgba(168, 85, 247, 0.3);
    }
    
    /* Light mode overrides */
    @media (prefers-color-scheme: light) {
        :root {
            --bg-primary: #faf5ff;
            --bg-secondary: #ffffff;
            --bg-card: #ffffff;
            --bg-card-end: #f3e8ff;
            --accent-primary: #9333ea;
            --accent-secondary: #4f46e5;
            --accent-glow: #a855f7;
            --accent-cyan: #0891b2;
            --accent-green: #059669;
            --accent-orange: #ea580c;
            --accent-red: #dc2626;
            --text-primary: #1e1b4b;
            --text-secondary: #6b6b9d;
            --border-color: #e9d5ff;
            --shadow-color: rgba(147, 51, 234, 0.1);
            --pinn-gradient-start: #9333ea;
            --pinn-gradient-end: #c084fc;
            --neural-glow: 0 0 30px rgba(147, 51, 234, 0.15);
        }
    }
    
    /* Streamlit theme detection - override for light theme */
    [data-theme="light"], 
    .stApp[data-theme="light"],
    html[data-theme="light"] :root {
        --bg-primary: #faf5ff;
        --bg-secondary: #ffffff;
        --bg-card: #ffffff;
        --bg-card-end: #f3e8ff;
        --accent-primary: #9333ea;
        --accent-secondary: #4f46e5;
        --accent-glow: #a855f7;
        --accent-cyan: #0891b2;
        --accent-green: #059669;
        --accent-orange: #ea580c;
        --accent-red: #dc2626;
        --text-primary: #1e1b4b;
        --text-secondary: #6b6b9d;
        --border-color: #e9d5ff;
        --shadow-color: rgba(147, 51, 234, 0.1);
        --pinn-gradient-start: #9333ea;
        --pinn-gradient-end: #c084fc;
        --neural-glow: 0 0 30px rgba(147, 51, 234, 0.15);
    }
    
    .main-header {
        font-family: 'Outfit', sans-serif;
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #7c3aed 0%, #a855f7 40%, #22d3ee 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.3rem;
        letter-spacing: -0.03em;
        text-shadow: 0 0 60px rgba(168, 85, 247, 0.5);
    }
    
    .sub-header {
        font-family: 'Fira Code', monospace;
        color: var(--text-secondary);
        text-align: center;
        font-size: 1.05rem;
        margin-bottom: 1rem;
        letter-spacing: 0.02em;
    }
    
    .pinn-badge {
        display: inline-block;
        background: linear-gradient(135deg, var(--pinn-gradient-start) 0%, var(--pinn-gradient-end) 100%);
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        box-shadow: 0 4px 15px rgba(168, 85, 247, 0.4);
    }
    
    .metric-card {
        background: linear-gradient(145deg, var(--bg-card) 0%, var(--bg-card-end) 100%);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 1.4rem;
        margin: 0.5rem 0;
        box-shadow: 0 8px 32px var(--shadow-color);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px var(--shadow-color);
    }
    
    .metric-label {
        font-family: 'Fira Code', monospace;
        color: var(--text-secondary);
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        margin-bottom: 0.4rem;
    }
    
    .metric-value {
        font-family: 'Outfit', sans-serif;
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--text-primary);
    }
    
    .pinn-hero-card {
        background: linear-gradient(145deg, #1a1a3e 0%, #2d2d5a 50%, #1a1a3e 100%);
        border: 2px solid var(--accent-primary);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: var(--neural-glow), 0 8px 32px var(--shadow-color);
        position: relative;
        overflow: hidden;
    }
    
    .pinn-hero-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(168, 85, 247, 0.1) 0%, transparent 50%);
        animation: pulse 4s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 0.5; transform: scale(1); }
        50% { opacity: 1; transform: scale(1.1); }
    }
    
    .solver-pinn { border-left: 4px solid #a855f7; }
    .solver-euler { border-left: 4px solid #fb923c; }
    .solver-improved { border-left: 4px solid #22d3ee; }
    .solver-rk4 { border-left: 4px solid #34d399; }
    .solver-ground { border-left: 4px solid #6366f1; }
    
    .neural-icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .architecture-box {
        background: linear-gradient(145deg, var(--bg-card) 0%, var(--bg-card-end) 100%);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.2rem;
        font-family: 'Fira Code', monospace;
        font-size: 0.85rem;
        color: var(--text-secondary);
    }
    
    .layer-indicator {
        display: inline-block;
        background: linear-gradient(90deg, var(--accent-primary), var(--accent-secondary));
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.75rem;
        margin: 0.2rem;
    }
    
    .equation-box {
        background: linear-gradient(145deg, #1a1a3e 0%, #13132b 100%);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        font-family: 'Fira Code', monospace;
        text-align: center;
        margin: 1rem 0;
        color: var(--text-primary);
    }
    
    .accuracy-excellent { color: #34d399; }
    .accuracy-good { color: #fbbf24; }
    .accuracy-moderate { color: #fb923c; }
    .accuracy-poor { color: #f87171; }
    
    .comparison-toggle {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--bg-secondary) 0%, var(--bg-primary) 100%);
        border-right: 1px solid var(--border-color);
    }
    
    section[data-testid="stSidebar"] .stMarkdown h2 {
        color: var(--accent-primary);
        font-family: 'Outfit', sans-serif;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: var(--bg-card);
        border-radius: 8px;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--pinn-gradient-start), var(--pinn-gradient-end)) !important;
    }
    
    .info-banner {
        background: linear-gradient(90deg, rgba(168, 85, 247, 0.15) 0%, rgba(99, 102, 241, 0.15) 100%);
        border: 1px solid var(--accent-primary);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
    }
    
    .performance-highlight {
        background: linear-gradient(135deg, rgba(34, 211, 238, 0.1) 0%, rgba(52, 211, 153, 0.1) 100%);
        border: 1px solid var(--accent-cyan);
        border-radius: 12px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# THEME DETECTION FOR MATPLOTLIB
def get_plot_theme():
    # Detect theme and return appropriate plot colors
    try:
        theme_base = st.get_option("theme.base")
        is_dark = theme_base != "light"
    except:
        is_dark = True
    
    if is_dark:
        return {
            "bg_primary": "#0d0d1a",
            "bg_secondary": "#13132b",
            "text_color": "#f0f0ff",
            "text_secondary": "#9898c8",
            "grid_color": "#3d3d6b",
            "border_color": "#3d3d6b",
            "legend_bg": "#1a1a3e"
        }
    else:
        return {
            "bg_primary": "#ffffff",
            "bg_secondary": "#faf5ff",
            "text_color": "#1e1b4b",
            "text_secondary": "#6b6b9d",
            "grid_color": "#e9d5ff",
            "border_color": "#d8b4fe",
            "legend_bg": "#ffffff"
        }

PLOT_THEME = get_plot_theme()

# HEADER
st.markdown('<h1 class="main-header">🧠 Brusselator PINN Solver</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Physics-Informed Neural Network for Chemical Oscillation Systems</p>', unsafe_allow_html=True)

# PINN Hero Section
st.markdown("""
<div style="text-align: center; margin-bottom: 1.5rem;">
    <span class="pinn-badge">✨ Deep Learning Powered</span>
</div>
""", unsafe_allow_html=True)

# Brusselator equations display
st.markdown("""
<div class="equation-box">
    <div style="margin-bottom: 0.5rem; font-size: 0.8rem; color: #9898c8; text-transform: uppercase; letter-spacing: 0.1em;">Brusselator System</div>
    <span style="color: #a855f7;">dx/dt</span> = A + x²y - (B+1)x &nbsp;&nbsp;&nbsp;&nbsp;
    <span style="color: #22d3ee;">dy/dt</span> = Bx - x²y
</div>
""", unsafe_allow_html=True)

# SIDEBAR - PARAMETERS
with st.sidebar:
    st.markdown("## 🧠 PINN Configuration")
    
    # PINN Info Box
    st.markdown("""
    <div class="architecture-box">
        <div style="color: #a855f7; font-weight: 600; margin-bottom: 0.5rem;">Neural Architecture</div>
        <div style="margin-bottom: 0.3rem;">📥 Input: 5 neurons</div>
        <div style="margin-bottom: 0.3rem;">&nbsp;&nbsp;&nbsp;&nbsp;(t, A, B, x₀, y₀)</div>
        <div style="margin-bottom: 0.3rem;">🔄 Hidden: 8 × 128 neurons</div>
        <div style="margin-bottom: 0.3rem;">📤 Output: 2 neurons (x, y)</div>
        <div style="margin-top: 0.5rem; color: #22d3ee;">Activation: Tanh</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
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
    
    st.markdown("---")
    st.markdown("### ⚙️ Simulation")
    st.markdown("**Time Range:** 0 - 20 seconds")
    st.markdown("**PINN Points:** 1,000")
    st.markdown("**Ground Truth:** RK4 @ dt=0.001")

# CONSTANTS AND HELPER FUNCTIONS
T = 20.0  # Fixed time range
GROUND_TRUTH_DT = 0.001  # High accuracy reference

# Color mapping for solvers - PINN is primary
SOLVER_COLORS = {
    "PINN": "#a855f7",
    "Euler": "#fb923c",
    "Improved Euler": "#22d3ee", 
    "RK4": "#34d399",
    "Ground Truth": "#6366f1"
}

SOLVER_STYLES = {
    "PINN": "-",
    "Euler": "--",
    "Improved Euler": "-.",
    "RK4": ":",
    "Ground Truth": "-"
}

DT_OPTIONS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]

@st.cache_resource
def load_pinn_model():
    # Load the PINN model (cached)
    model_path = os.path.join(PARENT_DIR, "PINN", "brusselator_pinn.pth")
    if os.path.exists(model_path):
        return BrusselatorPINNSolver(model_path)
    return None

def run_solver(solver_name, A, B, x0, y0, dt, T):
    # Run a solver and return results with timing info
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
    # Compute ground truth using RK4 at very small dt
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
    # Compute MSE between a solver result and ground truth
    MAX_REASONABLE_MSE = 1e10
    MAX_REASONABLE_VALUE = 1e100
    
    # Check for NaN or Inf in result (numerical overflow)
    if np.any(np.isnan(result['x'])) or np.any(np.isnan(result['y'])) or \
       np.any(np.isinf(result['x'])) or np.any(np.isinf(result['y'])):
        return {
            "mse_x": np.inf,
            "mse_y": np.inf,
            "mse_total": np.inf,
            "rmse_total": np.inf,
            "overflow": True
        }
    
    # Check for extremely large values
    if np.any(np.abs(result['x']) > MAX_REASONABLE_VALUE) or \
       np.any(np.abs(result['y']) > MAX_REASONABLE_VALUE):
        return {
            "mse_x": np.inf,
            "mse_y": np.inf,
            "mse_total": np.inf,
            "rmse_total": np.inf,
            "overflow": True
        }
    
    # Interpolate solver result to ground truth time points
    x_interp = np.interp(ground_truth['t'], result['t'], result['x'])
    y_interp = np.interp(ground_truth['t'], result['t'], result['y'])
    
    # Compute MSE with overflow protection
    with np.errstate(over='ignore', invalid='ignore'):
        mse_x = np.mean((x_interp - ground_truth['x']) ** 2)
        mse_y = np.mean((y_interp - ground_truth['y']) ** 2)
        mse_total = (mse_x + mse_y) / 2
    
    if np.isnan(mse_total) or np.isinf(mse_total) or mse_total > MAX_REASONABLE_MSE:
        return {
            "mse_x": np.inf,
            "mse_y": np.inf,
            "mse_total": np.inf,
            "rmse_total": np.inf,
            "overflow": True
        }
    
    return {
        "mse_x": mse_x,
        "mse_y": mse_y,
        "mse_total": mse_total,
        "rmse_total": np.sqrt(mse_total),
        "overflow": False
    }

def get_accuracy_class(mse):
    if mse < 1e-6:
        return "accuracy-excellent"
    elif mse < 1e-4:
        return "accuracy-good"
    elif mse < 1e-2:
        return "accuracy-moderate"
    else:
        return "accuracy-poor"

def get_accuracy_label(mse):
    if mse < 1e-6:
        return "Excellent"
    elif mse < 1e-4:
        return "Good"
    elif mse < 1e-2:
        return "Moderate"
    else:
        return "Poor"

# MAIN CONTENT - TAB STRUCTURE (PINN-First Layout)
main_tab1, main_tab2, main_tab3 = st.tabs(["🧠 PINN Solver", "⚡ Compare Methods", "📊 Benchmark Suite"])

# TAB 1: PINN SOLVER (Primary Focus)
with main_tab1:
    st.markdown("""
    <div class="info-banner">
        <span style="font-size: 1.1rem;">🎯 <strong>Test the PINN</strong></span>
        <span style="color: var(--text-secondary); margin-left: 1rem;">
            Run the Physics-Informed Neural Network and validate against ground truth
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    # PINN Feature Highlights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card solver-pinn">
            <div class="neural-icon">⚡</div>
            <div style="color: #a855f7; font-weight: 600; font-size: 1.1rem;">Instant Inference</div>
            <div style="color: var(--text-secondary); font-size: 0.9rem;">No iteration required</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card solver-pinn">
            <div class="neural-icon">🎛️</div>
            <div style="color: #a855f7; font-weight: 600; font-size: 1.1rem;">Parameter Flexible</div>
            <div style="color: var(--text-secondary); font-size: 0.9rem;">Handles varying A, B, x₀, y₀</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card solver-pinn">
            <div class="neural-icon">📐</div>
            <div style="color: #a855f7; font-weight: 600; font-size: 1.1rem;">Physics-Informed</div>
            <div style="color: var(--text-secondary); font-size: 0.9rem;">Trained on ODE constraints</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Run PINN button
    if st.button("🚀 Run PINN Solver", type="primary", key="run_pinn", width='stretch'):
        
        # Progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Compute ground truth first
        status_text.text("Computing ground truth (RK4 @ dt=0.001)...")
        ground_truth = compute_ground_truth(A, B, x0, y0, T)
        progress_bar.progress(0.3)
        
        # Run PINN
        status_text.text("Running PINN inference...")
        pinn_result = run_solver("PINN", A, B, x0, y0, None, T)
        progress_bar.progress(0.8)
        
        if pinn_result is not None:
            # Compute accuracy
            pinn_result['accuracy'] = compute_mse(pinn_result, ground_truth)
            progress_bar.progress(1.0)
            status_text.text("✅ Complete!")
            time.sleep(0.3)
            
            # Store results
            st.session_state['pinn_result'] = pinn_result
            st.session_state['pinn_ground_truth'] = ground_truth
            st.session_state['pinn_params'] = {'A': A, 'B': B, 'x0': x0, 'y0': y0}
        else:
            st.error("⚠️ PINN model not found. Please ensure the model file exists.")
        
        progress_bar.empty()
        status_text.empty()
    
    # Display PINN results
    if 'pinn_result' in st.session_state:
        pinn_result = st.session_state['pinn_result']
        ground_truth = st.session_state['pinn_ground_truth']
        params = st.session_state['pinn_params']
        acc = pinn_result['accuracy']
        acc_class = get_accuracy_class(acc['mse_total'])
        acc_label = get_accuracy_label(acc['mse_total'])
        
        # PINN Performance Hero Card
        st.markdown("""
        <div class="pinn-hero-card">
            <div style="position: relative; z-index: 1; text-align: center;">
                <span style="font-size: 2.5rem;">🧠</span>
                <h2 style="color: #f0f0ff; font-family: 'Outfit', sans-serif; margin: 0.5rem 0;">PINN Results</h2>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Use Streamlit columns for metrics
        hero_col1, hero_col2, hero_col3 = st.columns(3)
        
        with hero_col1:
            st.metric(label="⚡ Inference Time", value=f"{pinn_result['total_time']*1000:.3f} ms")
        
        with hero_col2:
            st.metric(label="🎯 MSE vs Ground Truth", value=f"{acc['mse_total']:.2e}")
        
        with hero_col3:
            st.metric(label="📊 Accuracy Rating", value=acc_label)
        
        # Detailed metrics
        st.markdown("### 📊 Detailed Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">MSE (x component)</div>
                <div class="metric-value {acc_class}">{acc['mse_x']:.2e}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">MSE (y component)</div>
                <div class="metric-value {acc_class}">{acc['mse_y']:.2e}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">RMSE (Total)</div>
                <div class="metric-value {acc_class}">{acc['rmse_total']:.2e}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Prediction Points</div>
                <div class="metric-value">1,000</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Solution Plots
        st.markdown("---")
        st.markdown("### 📈 PINN Solution vs Ground Truth")
        
        plot_tab1, plot_tab2 = st.tabs(["🔬 Time Series", "🌀 Phase Portrait"])
        
        with plot_tab1:
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
            
            # Plot ground truth (background)
            axes[0].plot(ground_truth['t'], ground_truth['x'], 
                        color=SOLVER_COLORS['Ground Truth'], 
                        linewidth=3, label='Ground Truth', alpha=0.5)
            axes[1].plot(ground_truth['t'], ground_truth['y'], 
                        color=SOLVER_COLORS['Ground Truth'], 
                        linewidth=3, label='Ground Truth', alpha=0.5)
            
            # Plot PINN (foreground, emphasized)
            axes[0].plot(pinn_result['t'], pinn_result['x'], 
                        color=SOLVER_COLORS['PINN'], 
                        linewidth=2.5, label='PINN', linestyle='--')
            axes[1].plot(pinn_result['t'], pinn_result['y'], 
                        color=SOLVER_COLORS['PINN'], 
                        linewidth=2.5, label='PINN', linestyle='--')
            
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
        
        with plot_tab2:
            theme = get_plot_theme()
            fig, ax = plt.subplots(figsize=(8, 8))
            fig.patch.set_facecolor(theme["bg_primary"])
            ax.set_facecolor(theme["bg_secondary"])
            ax.tick_params(colors=theme["text_secondary"])
            for spine in ax.spines.values():
                spine.set_color(theme["border_color"])
            ax.xaxis.label.set_color(theme["text_color"])
            ax.yaxis.label.set_color(theme["text_color"])
            ax.title.set_color(theme["text_color"])
            ax.grid(True, alpha=0.3, color=theme["grid_color"])
            
            # Ground truth
            ax.plot(ground_truth['x'], ground_truth['y'], 
                   color=SOLVER_COLORS['Ground Truth'], 
                   linewidth=3, label='Ground Truth', alpha=0.5)
            
            # PINN
            ax.plot(pinn_result['x'], pinn_result['y'], 
                   color=SOLVER_COLORS['PINN'], 
                   linewidth=2.5, label='PINN', linestyle='--')
            ax.plot(pinn_result['x'][0], pinn_result['y'][0], 'o', 
                   color=SOLVER_COLORS['PINN'], markersize=12, label='Start')
            
            ax.set_xlabel('x', fontsize=12)
            ax.set_ylabel('y', fontsize=12)
            ax.set_title('Phase Portrait (x vs y)', fontsize=14, fontweight='bold')
            ax.legend(facecolor=theme["legend_bg"], edgecolor=theme["border_color"], labelcolor=theme["text_color"])
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # Parameter Summary
        st.markdown("---")
        st.markdown("### 📌 Test Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            **Model Parameters:**
            - A = {params['A']}
            - B = {params['B']}
            - x₀ = {params['x0']}
            - y₀ = {params['y0']}
            """)
        
        with col2:
            st.markdown(f"""
            **Simulation Settings:**
            - Time Range: 0 - {T} seconds
            - PINN Points: 1,000
            - Ground Truth: RK4 @ dt={GROUND_TRUTH_DT}
            """)

# TAB 2: COMPARE METHODS (PINN vs Numerical)
with main_tab2:
    st.markdown("""
    <div class="info-banner">
        <span style="font-size: 1.1rem;">⚡ <strong>Compare PINN vs Numerical Methods</strong></span>
        <span style="color: var(--text-secondary); margin-left: 1rem;">
            See how the neural network stacks up against traditional solvers
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Comparison configuration
    st.markdown("### 🔧 Select Comparison Methods")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        comparison_methods = st.multiselect(
            "Numerical methods to compare against PINN:",
            options=["Euler", "Improved Euler", "RK4"],
            default=["RK4"],
            help="Select which numerical methods to compare with PINN"
        )
    
    with col2:
        comparison_dt = st.select_slider(
            "Time step (dt) for numerical methods:",
            options=DT_OPTIONS,
            value=0.01,
            key="comparison_dt",
            help="Time step used for numerical solvers"
        )
    
    if len(comparison_methods) == 0:
        st.warning("⚠️ Select at least one numerical method to compare")
    else:
        st.markdown("---")
        
        # Method cards
        st.markdown("### 📋 Methods Configuration")
        
        all_methods = ["PINN"] + comparison_methods
        cols = st.columns(len(all_methods))
        
        for i, method in enumerate(all_methods):
            with cols[i]:
                color = SOLVER_COLORS[method]
                if method == "PINN":
                    st.markdown(f"""
                    <div class="metric-card" style="border-left: 4px solid {color}; background: linear-gradient(145deg, #2d2d5a 0%, #1a1a3e 100%);">
                        <div style="color: {color}; font-weight: 700; font-size: 1.1rem;">🧠 PINN</div>
                        <div style="color: var(--text-secondary); font-size: 0.85rem; margin-top: 0.5rem;">Neural Network</div>
                        <div style="color: var(--text-secondary); font-size: 0.85rem;">1,000 points</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    n_steps = int(np.ceil(T / comparison_dt))
                    st.markdown(f"""
                    <div class="metric-card" style="border-left: 4px solid {color};">
                        <div style="color: {color}; font-weight: 700; font-size: 1.1rem;">📊 {method}</div>
                        <div style="color: var(--text-secondary); font-size: 0.85rem; margin-top: 0.5rem;">dt = {comparison_dt}</div>
                        <div style="color: var(--text-secondary); font-size: 0.85rem;">{n_steps:,} steps</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Run comparison
        if st.button("🚀 Run Comparison", type="primary", key="run_comparison", width='stretch'):
            
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Ground truth
            status_text.text("Computing ground truth...")
            ground_truth = compute_ground_truth(A, B, x0, y0, T)
            progress_bar.progress(0.15)
            
            # Run PINN first (primary)
            status_text.text("Running PINN inference...")
            pinn_result = run_solver("PINN", A, B, x0, y0, None, T)
            if pinn_result:
                pinn_result['accuracy'] = compute_mse(pinn_result, ground_truth)
                results.append(pinn_result)
            progress_bar.progress(0.3)
            
            # Run numerical methods
            total = len(comparison_methods)
            for i, method in enumerate(comparison_methods):
                status_text.text(f"Running {method}...")
                result = run_solver(method, A, B, x0, y0, comparison_dt, T)
                if result:
                    result['accuracy'] = compute_mse(result, ground_truth)
                    results.append(result)
                progress_bar.progress(0.3 + 0.7 * (i + 1) / total)
            
            status_text.text("✅ Complete!")
            time.sleep(0.3)
            progress_bar.empty()
            status_text.empty()
            
            st.session_state['comparison_results'] = results
            st.session_state['comparison_ground_truth'] = ground_truth
            st.session_state['comparison_params'] = {'A': A, 'B': B, 'x0': x0, 'y0': y0}
            st.session_state['comparison_dt_value'] = comparison_dt
        
        # Display comparison results
        if 'comparison_results' in st.session_state:
            results = st.session_state['comparison_results']
            ground_truth = st.session_state['comparison_ground_truth']
            params = st.session_state['comparison_params']
            comp_dt = st.session_state['comparison_dt_value']
            
            # Find PINN result for highlighting
            pinn_result = next((r for r in results if r['name'] == 'PINN'), None)
            
            # Performance comparison
            st.markdown("## ⚡ Performance Comparison")
            
            cols = st.columns(len(results))
            
            for i, result in enumerate(results):
                with cols[i]:
                    color = SOLVER_COLORS[result['name']]
                    acc = result['accuracy']
                    acc_class = get_accuracy_class(acc['mse_total'])
                    acc_label = get_accuracy_label(acc['mse_total'])
                    
                    is_pinn = result['name'] == 'PINN'
                    card_bg = "linear-gradient(145deg, #2d2d5a 0%, #1a1a3e 100%)" if is_pinn else "linear-gradient(145deg, var(--bg-card) 0%, var(--bg-card-end) 100%)"
                    
                    st.markdown(f"""
                    <div class="metric-card" style="border-left: 4px solid {color}; background: {card_bg};">
                        <div style="color: {color}; font-weight: 700; font-size: 1.2rem; margin-bottom: 1rem;">
                            {'🧠' if is_pinn else '📊'} {result['name']}
                        </div>
                        <div style="margin-bottom: 0.8rem;">
                            <div class="metric-label">Compute Time</div>
                            <div class="metric-value">{result['total_time']*1000:.3f} ms</div>
                        </div>
                        <div style="margin-bottom: 0.8rem;">
                            <div class="metric-label">MSE</div>
                            <div class="metric-value {acc_class}">{acc['mse_total']:.2e}</div>
                        </div>
                        <div>
                            <div class="metric-label">Rating</div>
                            <div class="{acc_class}" style="font-weight: 600;">{acc_label}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Comparison plots
            st.markdown("---")
            st.markdown("## 📈 Solution Comparison")
            
            comp_tab1, comp_tab2, comp_tab3 = st.tabs(["📊 All Methods", "🔍 Individual", "🌀 Phase Portrait"])
            
            with comp_tab1:
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
                
                # Ground truth (background)
                axes[0].plot(ground_truth['t'], ground_truth['x'], 
                            color=SOLVER_COLORS['Ground Truth'], 
                            linewidth=3, label='Ground Truth', alpha=0.4)
                axes[1].plot(ground_truth['t'], ground_truth['y'], 
                            color=SOLVER_COLORS['Ground Truth'], 
                            linewidth=3, label='Ground Truth', alpha=0.4)
                
                # All methods (PINN emphasized)
                for result in results:
                    lw = 3 if result['name'] == 'PINN' else 2
                    alpha = 1.0 if result['name'] == 'PINN' else 0.8
                    axes[0].plot(result['t'], result['x'], 
                                color=SOLVER_COLORS[result['name']], 
                                linestyle=SOLVER_STYLES[result['name']],
                                linewidth=lw, label=result['name'], alpha=alpha)
                    axes[1].plot(result['t'], result['y'], 
                                color=SOLVER_COLORS[result['name']], 
                                linestyle=SOLVER_STYLES[result['name']],
                                linewidth=lw, label=result['name'], alpha=alpha)
                
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
            
            with comp_tab2:
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
                    
                    color = SOLVER_COLORS[result['name']]
                    
                    # x comparison
                    axes[idx, 0].plot(ground_truth['t'], ground_truth['x'], 
                                     color=SOLVER_COLORS['Ground Truth'], 
                                     linewidth=2, label='Ground Truth', alpha=0.7)
                    axes[idx, 0].plot(result['t'], result['x'], 
                                     color=color, linewidth=2, label=result['name'], linestyle='--')
                    axes[idx, 0].set_xlabel('Time (t)')
                    axes[idx, 0].set_ylabel('x(t)')
                    axes[idx, 0].set_title(f"{result['name']} - x(t) | MSE: {result['accuracy']['mse_x']:.2e}", 
                                           fontsize=11, fontweight='bold')
                    axes[idx, 0].legend(facecolor=theme["legend_bg"], edgecolor=theme["border_color"], labelcolor=theme["text_color"])
                    
                    # y comparison
                    axes[idx, 1].plot(ground_truth['t'], ground_truth['y'], 
                                     color=SOLVER_COLORS['Ground Truth'], 
                                     linewidth=2, label='Ground Truth', alpha=0.7)
                    axes[idx, 1].plot(result['t'], result['y'], 
                                     color=color, linewidth=2, label=result['name'], linestyle='--')
                    axes[idx, 1].set_xlabel('Time (t)')
                    axes[idx, 1].set_ylabel('y(t)')
                    axes[idx, 1].set_title(f"{result['name']} - y(t) | MSE: {result['accuracy']['mse_y']:.2e}", 
                                           fontsize=11, fontweight='bold')
                    axes[idx, 1].legend(facecolor=theme["legend_bg"], edgecolor=theme["border_color"], labelcolor=theme["text_color"])
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            with comp_tab3:
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
                
                # Ground truth
                ax.plot(ground_truth['x'], ground_truth['y'], 
                       color=SOLVER_COLORS['Ground Truth'], 
                       linewidth=3, label='Ground Truth', alpha=0.4)
                
                # All methods
                for result in results:
                    lw = 3 if result['name'] == 'PINN' else 2
                    ax.plot(result['x'], result['y'], 
                           color=SOLVER_COLORS[result['name']], 
                           linestyle=SOLVER_STYLES[result['name']],
                           linewidth=lw, label=result['name'])
                    ax.plot(result['x'][0], result['y'][0], 'o', 
                           color=SOLVER_COLORS[result['name']], markersize=10)
                
                ax.set_xlabel('x', fontsize=12)
                ax.set_ylabel('y', fontsize=12)
                ax.set_title('Phase Portrait Comparison', fontsize=14, fontweight='bold')
                ax.legend(facecolor=theme["legend_bg"], edgecolor=theme["border_color"], labelcolor=theme["text_color"])
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            # Comparison table
            st.markdown("---")
            st.markdown("## 📋 Detailed Comparison")
            
            comparison_data = []
            for result in results:
                row = {
                    "Solver": f"{'🧠 ' if result['name'] == 'PINN' else '📊 '}{result['name']}",
                    "Type": "Neural Network" if result['name'] == 'PINN' else "Numerical",
                    "dt / Points": "1,000 pts" if result['name'] == 'PINN' else str(comp_dt),
                    "Steps": "-" if result['n_steps'] is None else f"{result['n_steps']:,}",
                    "Time (ms)": f"{result['total_time']*1000:.4f}",
                    "MSE (total)": f"{result['accuracy']['mse_total']:.2e}",
                    "MSE (x)": f"{result['accuracy']['mse_x']:.2e}",
                    "MSE (y)": f"{result['accuracy']['mse_y']:.2e}",
                    "Rating": get_accuracy_label(result['accuracy']['mse_total'])
                }
                comparison_data.append(row)
            
            df = pd.DataFrame(comparison_data)
            st.dataframe(df, hide_index=True, width='stretch')

# TAB 3: BENCHMARK SUITE
with main_tab3:
    st.markdown("""
    <div class="info-banner">
        <span style="font-size: 1.1rem;">📊 <strong>PINN Benchmark Suite</strong></span>
        <span style="color: var(--text-secondary); margin-left: 1rem;">
            Statistically evaluate PINN performance across many parameter combinations
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Benchmark configuration
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ⚙️ Benchmark Settings")
        n_param_sets = st.number_input(
            "Number of Parameter Sets",
            min_value=10,
            max_value=10000,
            value=100,
            step=10,
            help="Number of random parameter sets to test (10-10000)"
        )
        
        benchmark_dt = st.select_slider(
            "Time Step (dt) for Numerical Methods",
            options=DT_OPTIONS,
            value=0.01,
            key="benchmark_dt",
            help="Time step used for numerical comparison"
        )
        
        include_numerical = st.checkbox("Include numerical methods for comparison", value=True)
    
    with col2:
        st.markdown("### 📊 Parameter Ranges")
        st.markdown("""
        Random parameters will be sampled from:
        - **A**: 0.5 - 2.5
        - **B**: 1.0 - 6.0
        - **x₀**: 0.0 - 3.0
        - **y₀**: 0.0 - 3.0
        
        **Ground Truth**: RK4 @ dt=0.001
        """)
    
    st.markdown("---")
    
    # Run benchmark
    if st.button("🚀 Run Benchmark", type="primary", key="run_benchmark", width='stretch'):
        
        # Generate random parameter sets
        np.random.seed(42)
        param_sets = []
        for _ in range(n_param_sets):
            param_sets.append({
                'A': np.random.uniform(0.5, 2.5),
                'B': np.random.uniform(1.0, 6.0),
                'x0': np.random.uniform(0.0, 3.0),
                'y0': np.random.uniform(0.0, 3.0)
            })
        
        # Solvers to benchmark
        if include_numerical:
            all_solvers = ["PINN", "Euler", "Improved Euler", "RK4"]
        else:
            all_solvers = ["PINN"]
        
        benchmark_results = {solver: {'times': [], 'mses': []} for solver in all_solvers}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, params in enumerate(param_sets):
            status_text.text(f"Processing parameter set {i+1}/{n_param_sets}...")
            
            # Ground truth
            gt = compute_ground_truth(params['A'], params['B'], params['x0'], params['y0'], T)
            
            # Run each solver
            for solver_name in all_solvers:
                dt = benchmark_dt if solver_name != "PINN" else None
                result = run_solver(solver_name, params['A'], params['B'], params['x0'], params['y0'], dt, T)
                
                if result is not None:
                    accuracy = compute_mse(result, gt)
                    if not accuracy.get('overflow', False) and not np.isinf(accuracy['mse_total']):
                        benchmark_results[solver_name]['times'].append(result['total_time'])
                        benchmark_results[solver_name]['mses'].append(accuracy['mse_total'])
            
            progress_bar.progress((i + 1) / n_param_sets)
        
        status_text.text("✅ Benchmark complete!")
        time.sleep(0.3)
        progress_bar.empty()
        status_text.empty()
        
        st.session_state['benchmark_results'] = benchmark_results
        st.session_state['benchmark_n_sets'] = n_param_sets
        st.session_state['benchmark_dt_value'] = benchmark_dt
        st.session_state['benchmark_param_sets'] = param_sets
        st.session_state['benchmark_solvers'] = all_solvers
    
    # Display benchmark results
    if 'benchmark_results' in st.session_state:
        benchmark_results = st.session_state['benchmark_results']
        n_sets = st.session_state['benchmark_n_sets']
        b_dt = st.session_state['benchmark_dt_value']
        param_sets = st.session_state['benchmark_param_sets']
        all_solvers = st.session_state['benchmark_solvers']
        
        # PINN Hero Stats
        pinn_times = benchmark_results['PINN']['times']
        pinn_mses = benchmark_results['PINN']['mses']
        
        if pinn_times and pinn_mses:
            pinn_avg_time = np.mean(pinn_times) * 1000
            pinn_avg_mse = np.mean(pinn_mses)
            pinn_acc_label = get_accuracy_label(pinn_avg_mse)
            pinn_acc_class = get_accuracy_class(pinn_avg_mse)
            
            st.markdown(f"""
            <div class="pinn-hero-card">
                <div style="position: relative; z-index: 1; text-align: center;">
                    <span style="font-size: 2.5rem;">🧠</span>
                    <h2 style="color: #f0f0ff; font-family: 'Outfit', sans-serif; margin: 0.5rem 0;">PINN Benchmark Results</h2>
                    <div style="color: #9898c8;">Tested across {n_sets} random parameter combinations</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Use Streamlit columns for benchmark metrics
            bench_col1, bench_col2, bench_col3, bench_col4 = st.columns(4)
            
            with bench_col1:
                st.metric(label="⚡ Avg Inference", value=f"{pinn_avg_time:.3f} ms")
            
            with bench_col2:
                st.metric(label="🎯 Avg MSE", value=f"{pinn_avg_mse:.2e}")
            
            with bench_col3:
                st.metric(label="📊 Accuracy Rating", value=pinn_acc_label)
            
            with bench_col4:
                st.metric(label="✅ Valid Tests", value=f"{len(pinn_mses)}/{n_sets}")
        
        # Comparison cards (if numerical methods included)
        if len(all_solvers) > 1:
            st.markdown("### 📊 Method Comparison Summary")
            
            cols = st.columns(len(all_solvers))
            
            for i, solver_name in enumerate(all_solvers):
                with cols[i]:
                    times = benchmark_results[solver_name]['times']
                    mses = benchmark_results[solver_name]['mses']
                    
                    if times and mses:
                        avg_time = np.mean(times) * 1000
                        std_time = np.std(times) * 1000
                        avg_mse = np.mean(mses)
                        std_mse = np.std(mses)
                        
                        color = SOLVER_COLORS[solver_name]
                        acc_label = get_accuracy_label(avg_mse)
                        acc_class = get_accuracy_class(avg_mse)
                        
                        is_pinn = solver_name == 'PINN'
                        card_bg = "linear-gradient(145deg, #2d2d5a 0%, #1a1a3e 100%)" if is_pinn else "linear-gradient(145deg, var(--bg-card) 0%, var(--bg-card-end) 100%)"
                        
                        st.markdown(f"""
<div style="background: {card_bg}; border-radius: 16px; padding: 1.4rem; border-left: 4px solid {color}; box-shadow: 0 8px 32px var(--shadow-color);">
    <div style="font-size: 0.75rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.8rem; font-weight: 600;">
        {'🧠' if is_pinn else '📊'} {solver_name}
    </div>
    <div style="font-size: 1.6rem; font-weight: 700; color: var(--text-primary); margin-bottom: 0.2rem;">
        {avg_time:.3f} ms
    </div>
    <div style="font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 1rem;">
        ⏱️ Avg Time | Std: {std_time:.3f} ms
    </div>
    <div style="border-top: 1px solid var(--border-color); padding-top: 1rem;">
        <div style="font-size: 0.7rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.3rem;">🎯 Avg MSE</div>
        <div class="{acc_class}" style="font-size: 1.4rem; font-weight: 600;">{avg_mse:.2e}</div>
        <div style="font-size: 0.75rem; color: var(--text-secondary);">Std: {std_mse:.2e}</div>
    </div>
    <div style="margin-top: 0.8rem; padding: 0.5rem; background: {'rgba(168, 85, 247, 0.2)' if is_pinn else 'rgba(100, 100, 150, 0.2)'}; border-radius: 6px; text-align: center;">
        <span class="{acc_class}" style="font-weight: 600; font-size: 0.85rem;">Rating: {acc_label}</span>
    </div>
</div>
""", unsafe_allow_html=True)
        
        # Summary table
        st.markdown("---")
        st.markdown("### 📋 Summary Table")
        
        summary_data = []
        for solver_name in all_solvers:
            times = benchmark_results[solver_name]['times']
            mses = benchmark_results[solver_name]['mses']
            
            if times and mses:
                summary_data.append({
                    "Solver": f"{'🧠 ' if solver_name == 'PINN' else '📊 '}{solver_name}",
                    "Type": "Neural Network" if solver_name == 'PINN' else "Numerical",
                    "dt": "N/A" if solver_name == 'PINN' else str(b_dt),
                    "Avg Time (ms)": f"{np.mean(times)*1000:.4f}",
                    "Std Time (ms)": f"{np.std(times)*1000:.4f}",
                    "Avg MSE": f"{np.mean(mses):.2e}",
                    "Std MSE": f"{np.std(mses):.2e}",
                    "Min MSE": f"{np.min(mses):.2e}",
                    "Max MSE": f"{np.max(mses):.2e}",
                    "Rating": get_accuracy_label(np.mean(mses))
                })
        
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, hide_index=True, width='stretch')
        
        # Visualizations
        st.markdown("---")
        st.markdown("### 📊 Performance Visualizations")
        
        viz_tab1, viz_tab2, viz_tab3 = st.tabs(["📊 MSE Distribution", "⏱️ Time Distribution", "📈 Speed vs Accuracy"])
        
        with viz_tab1:
            theme = get_plot_theme()
            fig, ax = plt.subplots(figsize=(12, 6))
            fig.patch.set_facecolor(theme["bg_primary"])
            ax.set_facecolor(theme["bg_secondary"])
            ax.tick_params(colors=theme["text_secondary"])
            for spine in ax.spines.values():
                spine.set_color(theme["border_color"])
            ax.xaxis.label.set_color(theme["text_color"])
            ax.yaxis.label.set_color(theme["text_color"])
            ax.title.set_color(theme["text_color"])
            
            mse_data = [benchmark_results[s]['mses'] for s in all_solvers]
            bp = ax.boxplot(mse_data, tick_labels=all_solvers, patch_artist=True)
            
            for patch, solver in zip(bp['boxes'], all_solvers):
                patch.set_facecolor(SOLVER_COLORS[solver])
                patch.set_alpha(0.7)
            
            for flier in bp['fliers']:
                flier.set(marker='o', markerfacecolor='white', markeredgecolor='white', alpha=0.7)
            for whisker in bp['whiskers']:
                whisker.set(color=theme["text_secondary"], linewidth=1.5)
            for cap in bp['caps']:
                cap.set(color=theme["text_secondary"], linewidth=1.5)
            for median in bp['medians']:
                median.set(color='#fbbf24', linewidth=2)
            
            ax.set_ylabel('MSE', fontsize=12)
            ax.set_title('MSE Distribution by Solver', fontsize=14, fontweight='bold')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3, color=theme["grid_color"])
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with viz_tab2:
            theme = get_plot_theme()
            fig, ax = plt.subplots(figsize=(12, 6))
            fig.patch.set_facecolor(theme["bg_primary"])
            ax.set_facecolor(theme["bg_secondary"])
            ax.tick_params(colors=theme["text_secondary"])
            for spine in ax.spines.values():
                spine.set_color(theme["border_color"])
            ax.xaxis.label.set_color(theme["text_color"])
            ax.yaxis.label.set_color(theme["text_color"])
            ax.title.set_color(theme["text_color"])
            
            avg_times = [np.mean(benchmark_results[s]['times'])*1000 for s in all_solvers]
            std_times = [np.std(benchmark_results[s]['times'])*1000 for s in all_solvers]
            colors = [SOLVER_COLORS[s] for s in all_solvers]
            
            bars = ax.bar(all_solvers, avg_times, yerr=std_times, capsize=5, color=colors, alpha=0.8,
                         error_kw={'ecolor': 'white', 'capthick': 2, 'elinewidth': 1.5})
            
            ax.set_ylabel('Time (ms)', fontsize=12)
            ax.set_title('Average Computation Time by Solver', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, color=theme["grid_color"], axis='y')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with viz_tab3:
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
            
            for solver in all_solvers:
                avg_time = np.mean(benchmark_results[solver]['times']) * 1000
                avg_mse = np.mean(benchmark_results[solver]['mses'])
                marker_size = 300 if solver == 'PINN' else 200
                ax.scatter(avg_time, avg_mse, s=marker_size, c=SOLVER_COLORS[solver], 
                          label=solver, edgecolors='white', linewidth=2, alpha=0.9)
            
            ax.set_xlabel('Average Time (ms)', fontsize=12)
            ax.set_ylabel('Average MSE', fontsize=12)
            ax.set_title('Speed vs Accuracy Trade-off', fontsize=14, fontweight='bold')
            ax.set_yscale('log')
            ax.legend(facecolor=theme["legend_bg"], edgecolor=theme["border_color"], labelcolor=theme["text_color"])
            ax.grid(True, alpha=0.3, color=theme["grid_color"])
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # Export section
        st.markdown("---")
        st.markdown("## 📥 Export Benchmark Results")
        
        # Create detailed dataframe
        detailed_data = []
        for i, params in enumerate(param_sets):
            row = {
                "Run #": i + 1,
                "A": f"{params['A']:.4f}",
                "B": f"{params['B']:.4f}",
                "x₀": f"{params['x0']:.4f}",
                "y₀": f"{params['y0']:.4f}",
            }
            for solver_name in all_solvers:
                times = benchmark_results[solver_name]['times']
                mses = benchmark_results[solver_name]['mses']
                if i < len(times) and i < len(mses):
                    row[f"{solver_name} Time (ms)"] = f"{times[i]*1000:.4f}"
                    row[f"{solver_name} MSE"] = f"{mses[i]:.2e}"
            detailed_data.append(row)
        
        df_detailed = pd.DataFrame(detailed_data)
        
        with st.expander("📋 View All Individual Run Results", expanded=False):
            st.dataframe(df_detailed, hide_index=True, width='stretch')
        
        # Excel export
        import io
        buffer = io.BytesIO()
        
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df_summary.to_excel(writer, sheet_name='Summary Statistics', index=False)
            df_detailed.to_excel(writer, sheet_name='Individual Runs', index=False)
            
            config_data = pd.DataFrame([{
                "Number of Parameter Sets": n_sets,
                "Time Step (dt)": b_dt,
                "Ground Truth": "RK4 @ dt=0.001",
                "Parameter A Range": "0.5 - 2.5",
                "Parameter B Range": "1.0 - 6.0",
                "Initial x₀ Range": "0.0 - 3.0",
                "Initial y₀ Range": "0.0 - 3.0",
                "Time Range": "0 - 20 seconds"
            }])
            config_data.to_excel(writer, sheet_name='Configuration', index=False)
        
        excel_data = buffer.getvalue()
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.download_button(
                label="📥 Download Benchmark Report (Excel)",
                data=excel_data,
                file_name=f"pinn_benchmark_{n_sets}_sets.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
                width='stretch'
            )
