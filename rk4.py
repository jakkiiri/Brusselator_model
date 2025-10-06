import numpy as np
import matplotlib.pyplot as plt

# Define the differential equations for the Brusselator model
def brusselator(t, X, A, B):
    x, y = X
    dxdt = A + x**2 * y - (B + 1) * x
    dydt = B * x - x**2 * y
    return np.array([dxdt, dydt])

# RK4 Method
def rk4(f, t0, tf, X0, h, *args):
    t = np.arange(t0, tf, h)
    X = np.zeros((len(t), len(X0)))
    X[0] = X0
    
    for i in range(1, len(t)):
        k1 = f(t[i-1], X[i-1], *args)
        k2 = f(t[i-1] + 0.5*h, X[i-1] + 0.5*h*k1, *args)
        k3 = f(t[i-1] + 0.5*h, X[i-1] + 0.5*h*k2, *args)
        k4 = f(t[i], X[i-1] + h*k3, *args)
        
        X[i] = X[i-1] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    
    return t, X

# Parameters for the Brusselator model
A = 1.0
B = 3.0
X0 = [1.0, 2.0]  # Initial conditions: x(0) = 1, y(0) = 2
t0 = 0
tf = 100
h = 0.1  # Step size

# Solve using RK4
t, X = rk4(brusselator, t0, tf, X0, h, A, B)

# Extract x and y from the solution
x = X[:, 0]
y = X[:, 1]

# Plot the results
plt.figure(figsize=(10,6))

# Plot x(t) and y(t)
plt.subplot(2, 1, 1)
plt.plot(t, x, label='x(t)')
plt.plot(t, y, label='y(t)')
plt.xlabel('Time t')
plt.ylabel('Concentrations x and y')
plt.title('Concentrations of x and y vs Time')
plt.legend()

# Plot the phase plane
plt.subplot(2, 1, 2)
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Phase Plane: x vs y')

plt.tight_layout()
plt.show()
