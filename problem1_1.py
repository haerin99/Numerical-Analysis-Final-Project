"""
INN Final Project
Problem 1.1: Classical Numerical Solution
EN 553.481/681 Numerical Analysis
"""
import numpy as np
import matplotlib.pyplot as plt

def f(t, u):
    return -5*u + 5*np.cos(t) - np.sin(t)

def exact_solution(t):
    return np.cos(t) - np.exp(-5*t)

def forward_euler(h):
    t = np.arange(0, 5 + h, h)
    u = np.zeros(len(t))
    u[0] = 0

    for n in range(len(t) - 1):
        u[n+1] = u[n] + h*f(t[n], u[n])

    return t, u

def rk4(h):
    t = np.arange(0, 5 + h, h)
    u = np.zeros(len(t))
    u[0] = 0

    for n in range(len(t) - 1):
        k1 = f(t[n], u[n])
        k2 = f(t[n] + h/2, u[n] + h*k1/2)
        k3 = f(t[n] + h/2, u[n] + h*k2/2)
        k4 = f(t[n] + h, u[n] + h*k3)

        u[n+1] = u[n] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

    return t, u


# 1.1(a) and 1.1(b), h = 0.01

h = 0.01

t_fe, u_fe = forward_euler(h)
t_rk4, u_rk4 = rk4(h)

u_exact = exact_solution(t_fe)

plt.figure(figsize=(8, 5))
plt.plot(t_fe, u_exact, label="Exact solution", linewidth=2)
plt.plot(t_fe, u_fe, "--", label="Forward Euler, h=0.01")
plt.xlabel("t")
plt.ylabel("u(t)")
plt.title("Forward Euler vs Exact Solution")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(t_rk4, exact_solution(t_rk4), label="Exact solution", linewidth=2)
plt.plot(t_rk4, u_rk4, "--", label="RK4, h=0.01")
plt.xlabel("t")
plt.ylabel("u(t)")
plt.title("RK4 vs Exact Solution")
plt.legend()
plt.grid(True)
plt.show()


# 1.1(c): Global errors and convergence order

h_values = [0.01, 0.005, 0.001]

fe_errors = []
rk4_errors = []

for h in h_values:
    t_fe, u_fe = forward_euler(h)
    t_rk4, u_rk4 = rk4(h)

    fe_error = np.max(np.abs(u_fe - exact_solution(t_fe)))
    rk4_error = np.max(np.abs(u_rk4 - exact_solution(t_rk4)))

    fe_errors.append(fe_error)
    rk4_errors.append(rk4_error)

def observed_orders(errors, h_values):
    orders = [np.nan]

    for i in range(1, len(errors)):
        p = np.log(errors[i-1] / errors[i]) / np.log(h_values[i-1] / h_values[i])
        orders.append(p)

    return orders

fe_orders = observed_orders(fe_errors, h_values)
rk4_orders = observed_orders(rk4_errors, h_values)

print("Forward Euler")
print("h\t\tError\t\tObserved Order")
for h, err, p in zip(h_values, fe_errors, fe_orders):
    print(f"{h:<8}\t{err:.6e}\t{p if not np.isnan(p) else '-'}")

print("\nRK4")
print("h\t\tError\t\tObserved Order")
for h, err, p in zip(h_values, rk4_errors, rk4_orders):
    print(f"{h:<8}\t{err:.6e}\t{p if not np.isnan(p) else '-'}")
