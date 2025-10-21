import numpy as np
import argparse
import time
import matplotlib.pyplot as plt

# Equation Definition
def brusselator_rhs(x, y, A, B):
    dxdt = A + x * x * y - (B + 1.0) * x
    dydt = B * x - x * x * y
    return dxdt, dydt

# Improved Euler's Method 
def brusselator_euler(A, B, x0, y0, dt, T, record_step_times=False):
    n_steps = int(np.ceil(T / dt)) # Time Steps to Cover [0, T]
    t = np.linspace(0.0, n_steps * dt, n_steps + 1) # Time Array
    # Empty Solution Arrays for x and y
    x = np.empty(n_steps + 1) 
    y = np.empty(n_steps + 1)
    # Set Initial Conditions
    x[0], y[0] = x0, y0

    # Array to Record Step Times
    step_times = np.empty(n_steps) if record_step_times else None

    # Improved Euler Loop
    for k in range(n_steps):
        # Timer to Record Step Time
        t0 = time.perf_counter()
        # Slope at the beginning (k1)
        dx1, dy1 = brusselator_rhs(x[k], y[k], A, B)
        # Euler predictor
        x_pred = x[k] + dt * dx1
        y_pred = y[k] + dt * dy1
        # Slope at the predictor (k2)
        dx2, dy2 = brusselator_rhs(x_pred, y_pred, A, B)
        # Trapezoidal correction
        x[k + 1] = x[k] + 0.5 * dt * (dx1 + dx2)
        y[k + 1] = y[k] + 0.5 * dt * (dy1 + dy2)
        # Record Step Time
        if record_step_times:
            step_times[k] = time.perf_counter() - t0

    if record_step_times:
        return t, x, y, step_times
    return t, x, y, None

def run_demo():
    # Parser to Handle Plot Arguments
    parser = argparse.ArgumentParser(description="Improved Euler (Heun) method for the Brusselator model.")
    parser.add_argument("--A", type=float, default=1.0, help="Parameter A")
    parser.add_argument("--B", type=float, default=3.0, help="Parameter B")
    parser.add_argument("--x0", type=float, default=1.2, help="Initial x")
    parser.add_argument("--y0", type=float, default=2.5, help="Initial y")
    parser.add_argument("--T", type=float, default=20.0, help="Final time")
    parser.add_argument("--dts", type=str, default="0.1,0.05,0.01,0.005",
                        help="Comma separated list of time steps to test")
    parser.add_argument("--record-step-times", action="store_true",
                        help="Record and print each individual step time (may be a lot of output)")
    parser.add_argument("--no-show", action="store_true", help="Skip interactive show (just save figure)")
    parser.add_argument("--outfile", type=str, default="brusselator_euler.png", help="Output plot filename")
    args = parser.parse_args()

    dts = [float(s) for s in args.dts.split(",")]

    print("Timing results:")
    plt.figure(figsize=(8, 5))
    # Compute the Computation Time
    for dt in dts:
        start = time.perf_counter()
        t, x, y, step_times = brusselator_euler(
            args.A, args.B, args.x0, args.y0, dt, args.T,
            record_step_times=args.record_step_times
        )
        # Total Elapsed Time
        total_elapsed = time.perf_counter() - start
        n_steps = len(t) - 1
        per_step = total_elapsed / n_steps
        print(f"dt={dt} steps={n_steps} total={total_elapsed:.6f}s per_step={per_step:.6e}s")

        if args.record_step_times:
            # Print a short summary 
            print(f"  step_times: min={step_times.min():.3e}s max={step_times.max():.3e}s mean={step_times.mean():.3e}s")

        plt.plot(t, x, label=f"x dt={dt}")
        plt.plot(t, y, linestyle="--", label=f"y dt={dt}")

    # Plotting
    plt.xlabel("t")
    plt.ylabel("x(t), y(t)")
    plt.title(f"Brusselator Improved Euler (Heun) integration (A={args.A}, B={args.B})")
    plt.legend(fontsize="small", ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.outfile, dpi=150)
    if not args.no_show:
        plt.show()

if __name__ == "__main__":
    run_demo()