import numpy as np
import argparse

# regular_euler.py
import matplotlib.pyplot as plt

def brusselator_rhs(x, y, A, B):
    dxdt = A + x * x * y - (B + 1.0) * x
    dydt = B * x - x * x * y
    return dxdt, dydt

def brusselator_euler(A, B, x0, y0, dt, T):
    n_steps = int(np.ceil(T / dt))
    t = np.linspace(0.0, n_steps * dt, n_steps + 1)
    x = np.empty(n_steps + 1)
    y = np.empty(n_steps + 1)
    x[0], y[0] = x0, y0
    for k in range(n_steps):
        dxdt, dydt = brusselator_rhs(x[k], y[k], A, B)
        x[k + 1] = x[k] + dt * dxdt
        y[k + 1] = y[k] + dt * dydt
    return t, x, y

def run_demo():
    parser = argparse.ArgumentParser(description="Explicit (regular) Euler method for the Brusselator model.")
    parser.add_argument("--A", type=float, default=1.0, help="Parameter A")
    parser.add_argument("--B", type=float, default=3.0, help="Parameter B")
    parser.add_argument("--x0", type=float, default=1.2, help="Initial x")
    parser.add_argument("--y0", type=float, default=2.5, help="Initial y")
    parser.add_argument("--T", type=float, default=20.0, help="Final time")
    parser.add_argument("--dts", type=str, default="0.1,0.05,0.01,0.005",
                        help="Comma separated list of time steps to test")
    parser.add_argument("--no-show", action="store_true", help="Skip interactive show (just save figure)")
    parser.add_argument("--outfile", type=str, default="brusselator_euler.png", help="Output plot filename")
    args = parser.parse_args()

    dts = [float(s) for s in args.dts.split(",")]

    plt.figure(figsize=(8, 5))
    for dt in dts:
        t, x, y = brusselator_euler(args.A, args.B, args.x0, args.y0, dt, args.T)
        plt.plot(t, x, label=f"x dt={dt}")
        plt.plot(t, y, linestyle="--", label=f"y dt={dt}")

    plt.xlabel("t")
    plt.ylabel("x(t), y(t)")
    plt.title(f"Brusselator Euler integration (A={args.A}, B={args.B})")
    plt.legend(fontsize="small", ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.outfile, dpi=150)
    if not args.no_show:
        plt.show()

if __name__ == "__main__":
    run_demo()