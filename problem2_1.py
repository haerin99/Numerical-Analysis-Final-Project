"""
INN Final Project
Problem 2.1: Finite-Difference Reference Solution
EN 553.481/681 Numerical Analysis
"""
import numpy as np
import matplotlib.pyplot as plt

nu = 0.01
dx = 1/64
x = np.arange(0, 1 + dx, dx)

r = 0.5
dt = r * dx**2 / nu

T_final = 0.5
t = np.arange(0, T_final + dt, dt)

# Compute r
r = nu * dt / dx**2

print(f"dt = {dt}")
print(f"r = {r}")

def heat_exact(x, t):
    return np.exp(-nu*np.pi**2*t)*np.sin(np.pi*x) + \
           0.5*np.exp(-9*nu*np.pi**2*t)*np.sin(3*np.pi*x)

# Initial condition
u_fd = np.zeros((len(t), len(x)))
u_fd[0, :] = np.sin(np.pi*x) + 0.5*np.sin(3*np.pi*x)

# Boundary conditions
u_fd[:, 0] = 0
u_fd[:, -1] = 0

# Forward Euler finite-difference scheme
for n in range(len(t) - 1):
    for j in range(1, len(x) - 1):
        u_fd[n+1, j] = u_fd[n, j] + r * (
            u_fd[n, j+1] - 2*u_fd[n, j] + u_fd[n, j-1]
        )

# 2.1(b): L2 error at t = 0.5
u_exact_final = heat_exact(x, T_final)
u_num_final = u_fd[-1, :]

L2_error = np.sqrt(np.sum((u_num_final - u_exact_final)**2) * dx)

print("\nProblem 2.1(b)")
print(f"L2 error at t = 0.5: {L2_error:.6e}")

# 2.1(c): Heatmap
X, T = np.meshgrid(x, t)

plt.figure(figsize=(8, 5))
plt.pcolormesh(X, T, u_fd, shading="auto")
plt.xlabel("x")
plt.ylabel("t")
plt.title("Forward Euler FD Heat Equation Solution")
plt.colorbar(label="u(x,t)")
plt.tight_layout()
plt.savefig("heat_fd_solution_heatmap.png", dpi=300)
plt.show()
