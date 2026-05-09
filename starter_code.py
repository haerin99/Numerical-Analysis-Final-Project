"""
Starter Code: PINN Final Project
EN 553.481/681 Numerical Analysis
"""
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PINN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def train_pinn(model, loss_fn, epochs, lr=1e-3, log_every=2000):
    """Train a PINN model.
    Returns: (loss_history, wall_clock_time_seconds)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history = []
    t_start = time.time()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        loss = loss_fn(model)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        if epoch % log_every == 0:
            print(f"  Epoch {epoch}/{epochs}, Loss = {loss.item():.6e}")
    wall_time = time.time() - t_start
    print(f"  Training time: {wall_time:.1f}s")
    return loss_history, wall_time

def plot_loss_curve(loss_history, title="Training Loss"):
    plt.figure(figsize=(6, 4))
    plt.semilogy(loss_history)
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title(title); plt.grid(True, alpha=0.3)
    plt.tight_layout()

def plot_ode_comparison(model, exact_fn, t_range=(0, 5), label="PINN"):
    t = torch.linspace(*t_range, 1000, device=device).unsqueeze(1)
    with torch.no_grad():
        u_pred = model(t).cpu().numpy().flatten()
    t_np = t.cpu().numpy().flatten()
    u_ex = exact_fn(t_np)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(t_np, u_ex, 'k-', lw=2, label='Exact')
    axes[0].plot(t_np, u_pred, 'r--', lw=1.5, label=label)
    axes[0].set_xlabel('t'); axes[0].set_ylabel('u(t)')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f'{label} vs Exact')

    err = np.abs(u_pred - u_ex)
    axes[1].plot(t_np, err, 'b-')
    axes[1].set_xlabel('t'); axes[1].set_ylabel('|error|')
    axes[1].set_title(f'Pointwise Error (max = {err.max():.4e})')
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    print(f"  Max absolute error: {err.max():.6e}")
    return err.max()

def plot_heat_comparison(model, exact_fn, label="PINN"):
    """Plot PINN vs exact for heat eq. Returns relative L2 error."""
    Ntest = 100
    x = np.linspace(0, 1, Ntest)
    t = np.linspace(0, 0.5, Ntest)
    X, T = np.meshgrid(x, t)
    xt = np.column_stack([X.ravel(), T.ravel()])
    xt_t = torch.tensor(xt, dtype=torch.float32, device=device)
    with torch.no_grad():
        u_pred = model(xt_t).cpu().numpy().reshape(Ntest, Ntest)
    u_ex = exact_fn(X, T)
    err = np.abs(u_pred - u_ex)
    rel_l2 = np.sqrt(np.sum((u_pred - u_ex)**2)) / np.sqrt(np.sum(u_ex**2))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    c0 = axes[0].pcolormesh(X, T, u_pred, shading='auto', cmap='viridis')
    axes[0].set_xlabel('x'); axes[0].set_ylabel('t')
    axes[0].set_title(f'{label} Prediction'); plt.colorbar(c0, ax=axes[0])
    c1 = axes[1].pcolormesh(X, T, u_ex, shading='auto', cmap='viridis')
    axes[1].set_xlabel('x'); axes[1].set_ylabel('t')
    axes[1].set_title('Exact Solution'); plt.colorbar(c1, ax=axes[1])
    c2 = axes[2].pcolormesh(X, T, err, shading='auto', cmap='hot')
    axes[2].set_xlabel('x'); axes[2].set_ylabel('t')
    axes[2].set_title(f'|Error| (rel L2 = {rel_l2:.4e})'); plt.colorbar(c2, ax=axes[2])
    plt.tight_layout()
    print(f"  Relative L2 error: {rel_l2:.6e}")
    return rel_l2

# =============================================================
# TODO: Implement these four loss functions
# =============================================================

def compute_loss_ode_ad(model):
    """PINN loss for ODE using AUTOGRAD.

    ODE: du/dt = -5u + 5cos(t) - sin(t),  u(0) = 0
    """
    # Sample collocation points uniformly from [0, 5]
    Nr = 500
    t = 5 * torch.rand(Nr, 1, device=device)
    t.requires_grad_(True)

    # du_theta/dt
    u = model(t)
    du_dt = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
        create_graph=True)[0]

    # Residual loss
    residual = du_dt + 5*u - 5*torch.cos(t) + torch.sin(t)
    Lr = torch.mean(residual**2)

    # IC loss
    t0 = torch.zeros(1, 1, device=device)
    u0_pred = model(t0)
    Lic = torch.mean(u0_pred**2)

    # Total loss
    loss = Lr + 50*Lic

    return loss


def compute_loss_ode_fdm(model, epsilon=1e-3):
    """PINN loss for ODE using FINITE DIFFERENCES.

    Same ODE as above. Instead of autograd, approximate du/dt
    using the central difference formula:

        du/dt(t) ≈ (u(t + epsilon) - u(t - epsilon)) / (2 * epsilon)
    """
    # Sample collocation points from [0, 5]
    Nr = 500
    t = 5 * torch.rand(Nr, 1, device=device)

    # Central difference formula
    u = model(t)
    du_dt = (model(t + epsilon) - model(t - epsilon)) / (2 * epsilon)

    # Residual loss
    residual = du_dt + 5*u - 5*torch.cos(t) + torch.sin(t)
    Lr = torch.mean(residual**2)

    # IC loss
    t0 = torch.zeros(1, 1, device=device)
    u0_pred = model(t0)
    Lic = torch.mean(u0_pred**2)

    # Total loss
    loss = Lr + 50*Lic

    return loss


def compute_loss_heat_ad(model):
    """PINN loss for heat equation using AUTOGRAD.

    PDE: u_t = 0.01 * u_xx  on (0,1) x (0, 0.5]
    IC:  u(x, 0) = sin(pi*x) + 0.5*sin(3*pi*x)
    BC:  u(0, t) = u(1, t) = 0
    """
    nu = 0.01

    # Residual loss
    Nr = 10000

    x = torch.rand(Nr, 1, device=device)
    t = 0.5 * torch.rand(Nr, 1, device=device)

    x.requires_grad_(True)
    t.requires_grad_(True)

    xt = torch.cat([x, t], dim=1)
    u = model(xt)

    # u_t
    u_t = torch.autograd.grad(
        u,
        t,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    # u_x
    u_x = torch.autograd.grad(
        u,
        x,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    # u_xx
    u_xx = torch.autograd.grad(
        u_x,
        x,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True
    )[0]

    # Residual
    residual = u_t - nu * u_xx
    Lr = torch.mean(residual**2)

    # Initial condition loss
    Nic = 200

    x_ic = torch.rand(Nic, 1, device=device)
    t_ic = torch.zeros(Nic, 1, device=device)

    xt_ic = torch.cat([x_ic, t_ic], dim=1)

    u_ic_pred = model(xt_ic)
    u_ic_true = torch.sin(torch.pi * x_ic) + 0.5 * torch.sin(3 * torch.pi * x_ic)

    Lic = torch.mean((u_ic_pred - u_ic_true)**2)

    # Boundary condition loss
    Nbc = 200

    t_bc = 0.5 * torch.rand(Nbc, 1, device=device)

    x_left = torch.zeros(Nbc, 1, device=device)
    x_right = torch.ones(Nbc, 1, device=device)

    xt_left = torch.cat([x_left, t_bc], dim=1)
    xt_right = torch.cat([x_right, t_bc], dim=1)

    u_left = model(xt_left)
    u_right = model(xt_right)

    Lbc = torch.mean(u_left**2) + torch.mean(u_right**2)

    # Total loss
    loss = Lr + 20*Lic + 20*Lbc

    return loss


def compute_loss_heat_fdm(model, epsilon=1e-3):
    """PINN loss for heat equation using FINITE DIFFERENCES.

    Same PDE, IC, BC as above. Approximate derivatives:

        u_t(x,t)  ≈ (u(x, t+eps) - u(x, t-eps)) / (2*eps)
        u_xx(x,t) ≈ (u(x+eps, t) - 2*u(x,t) + u(x-eps, t)) / eps^2
    """
    nu = 0.01
    eps = epsilon

    # Residual loss
    Nr = 10000

    # Sample interior points
    x = eps + (1 - 2*eps) * torch.rand(Nr, 1, device=device)
    t = eps + (0.5 - 2*eps) * torch.rand(Nr, 1, device=device)

    xt = torch.cat([x, t], dim=1)

    # Points for u_t central difference
    xt_t_plus = torch.cat([x, t + eps], dim=1)
    xt_t_minus = torch.cat([x, t - eps], dim=1)

    # Points for u_xx central difference
    xt_x_plus = torch.cat([x + eps, t], dim=1)
    xt_x_minus = torch.cat([x - eps, t], dim=1)

    # Finite-difference approximations
    u = model(xt)
    u_t_plus = model(xt_t_plus)
    u_t_minus = model(xt_t_minus)
    u_x_plus = model(xt_x_plus)
    u_x_minus = model(xt_x_minus)

    u_t = (u_t_plus - u_t_minus) / (2 * eps)

    u_xx = (u_x_plus - 2*u + u_x_minus) / (eps**2)

    # PDE residual: u_t - nu*u_xx = 0
    residual = u_t - nu*u_xx

    Lr = torch.mean(residual**2)

    # Initial condition loss
    Nic = 200

    x_ic = torch.rand(Nic, 1, device=device)
    t_ic = torch.zeros(Nic, 1, device=device)

    xt_ic = torch.cat([x_ic, t_ic], dim=1)

    u_ic_pred = model(xt_ic)
    u_ic_true = torch.sin(torch.pi * x_ic) + 0.5 * torch.sin(3 * torch.pi * x_ic)

    Lic = torch.mean((u_ic_pred - u_ic_true)**2)

    # Boundary condition loss
    Nbc = 200

    t_bc = 0.5 * torch.rand(Nbc, 1, device=device)

    x_left = torch.zeros(Nbc, 1, device=device)
    x_right = torch.ones(Nbc, 1, device=device)

    xt_left = torch.cat([x_left, t_bc], dim=1)
    xt_right = torch.cat([x_right, t_bc], dim=1)

    u_left = model(xt_left)
    u_right = model(xt_right)

    Lbc = torch.mean(u_left**2) + torch.mean(u_right**2)
    
    # Total loss
    loss = Lr + 20*Lic + 20*Lbc

    return loss

def compute_loss_ode_ad_Nr(model, Nr=500):
    """
    Accepts Nr
    """
    t = 5 * torch.rand(Nr, 1, device=device)
    t.requires_grad_(True)

    u = model(t)

    du_dt = torch.autograd.grad(
        u, t,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    residual = du_dt + 5*u - 5*torch.cos(t) + torch.sin(t)
    Lr = torch.mean(residual**2)

    t0 = torch.zeros(1, 1, device=device)
    Lic = torch.mean(model(t0)**2)

    return Lr + 50*Lic


def compute_loss_ode_fdm_Nr(model, Nr=500, epsilon=1e-3):
    """
    Accepts Nr
    """
    t = 5 * torch.rand(Nr, 1, device=device)

    u = model(t)
    u_plus = model(t + epsilon)
    u_minus = model(t - epsilon)

    du_dt = (u_plus - u_minus) / (2*epsilon)

    residual = du_dt + 5*u - 5*torch.cos(t) + torch.sin(t)
    Lr = torch.mean(residual**2)

    t0 = torch.zeros(1, 1, device=device)
    Lic = torch.mean(model(t0)**2)

    return Lr + 50*Lic

if __name__ == "__main__":
    ode_exact = lambda t: np.cos(t) - np.exp(-5*t)
    nu = 0.01
    heat_exact = lambda X, T: np.exp(-nu*np.pi**2*T)*np.sin(np.pi*X) + \
                          0.5*np.exp(-9*nu*np.pi**2*T)*np.sin(3*np.pi*X)
    
    """
    # --- Problem 1.2: ODE with AD ---
    print("=" * 50)
    print("Problem 1.2: ODE PINN (Autograd)")
    print("=" * 50)
   ## TODO: Experiments for Problem 1.2: train the AD-PINN for the ODE, plot loss curve and results
    # ODE PINN model
    ode_ad_model = PINN(input_dim=1, hidden_dim=32, num_layers=3, output_dim=1).to(device)

    # Train for 10,000 epochs
    ode_ad_loss, ode_ad_time = train_pinn(
        model=ode_ad_model,
        loss_fn=compute_loss_ode_ad,
        epochs=10000,
        lr=1e-3,
        log_every=2000)

    # Plot loss curve
    plot_loss_curve(ode_ad_loss, title="ODE AD-PINN Training Loss")
    plt.show()

    # Plot PINN solution vs exact solution
    ode_ad_error = plot_ode_comparison(model=ode_ad_model, exact_fn=ode_exact, t_range=(0, 5), label="AD-PINN")
    plt.show()
    
    
    # --- Problem 1.3: ODE with FDM ---
    print("\n" + "=" * 50)
    print("Problem 1.3: ODE PINN (FDM)")
    print("=" * 50)
    ## TODO: Experiments for Problem 1.3: train the FDM-PINN for the ODE, plot loss curve and results

    # Create ODE FDM-PINN model
    ode_fdm_model = PINN(input_dim=1, hidden_dim=32, num_layers=3, output_dim=1).to(device)

    # Train for 10,000 epochs
    ode_fdm_loss, ode_fdm_time = train_pinn(
        model=ode_fdm_model,
        loss_fn=lambda model: compute_loss_ode_fdm(model, epsilon=1e-3),
        epochs=10000,
        lr=1e-3,
        log_every=2000
    )

    # Plot loss curve
    plot_loss_curve(ode_fdm_loss, title="ODE FDM-PINN Training Loss")
    plt.show()

    # Plot FDM-PINN solution vs exact solution
    ode_fdm_error = plot_ode_comparison(model=ode_fdm_model, exact_fn=ode_exact, t_range=(0, 5), label="FDM-PINN")
    plt.show()


    # --- Problem 1.4 ---
    print("\n" + "=" * 50)
    print("Problem 1.4: ODE PINN Comparison")
    print("=" * 50)

    # 1.4(b)
    epsilon_values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

    fdm_epsilon_errors = []
    fdm_epsilon_losses = []
    fdm_epsilon_times = []

    for eps in epsilon_values:
        print(f"\nTraining FDM-PINN with epsilon = {eps}")

        model_eps = PINN(
            input_dim=1,
            hidden_dim=32,
            num_layers=3,
            output_dim=1
        ).to(device)

        loss_eps, time_eps = train_pinn(
            model=model_eps,
            loss_fn=lambda model, eps=eps: compute_loss_ode_fdm(model, epsilon=eps),
            epochs=10000,
            lr=1e-3,
            log_every=2000
        )

        error_eps = plot_ode_comparison(
            model=model_eps,
            exact_fn=ode_exact,
            t_range=(0, 5),
            label=f"FDM-PINN eps={eps}"
        )
        plt.show()

        fdm_epsilon_errors.append(error_eps)
        fdm_epsilon_losses.append(loss_eps[-1])
        fdm_epsilon_times.append(time_eps)

    # Plot max error vs epsilon
    plt.figure(figsize=(7, 5))
    plt.loglog(epsilon_values, fdm_epsilon_errors, marker="o")
    plt.xlabel("epsilon")
    plt.ylabel("Max absolute error")
    plt.title("Problem 1.4(b): FDM-PINN Max Error vs Epsilon")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.show()


    # --- Problem 2.2: Heat with AD ---
    print("\n" + "=" * 50)
    print("Problem 2.2: Heat PINN (Autograd)")
    print("=" * 50)
    ## TODO: Experiments for Problem 2.2: train the AD-PINN for the heat equation, plot loss curve and results
    heat_ad_model = PINN(
        input_dim=2,
        hidden_dim=64,
        num_layers=4,
        output_dim=1
    ).to(device)

    heat_ad_loss, heat_ad_time = train_pinn(
        model=heat_ad_model,
        loss_fn=compute_loss_heat_ad,
        epochs=20000,
        lr=1e-3,
        log_every=2000
    )

    plot_loss_curve(
        heat_ad_loss,
        title="Heat AD-PINN Training Loss"
    )
    plt.show()

    heat_ad_error = plot_heat_comparison(
        model=heat_ad_model,
        exact_fn=heat_exact,
        label="Heat AD-PINN"
    )
    plt.show()


    # --- Problem 2.3: Heat with FDM ---
    print("\n" + "=" * 50)
    print("Problem 2.3: Heat PINN (FDM)")
    print("=" * 50)
   ## TODO: Experiments for Problem 2.3: train the FDM-PINN for the heat equation, plot loss curve and results
    heat_fdm_model = PINN(
        input_dim=2,
        hidden_dim=64,
        num_layers=4,
        output_dim=1
    ).to(device)

    heat_fdm_loss, heat_fdm_time = train_pinn(
        model=heat_fdm_model,
        loss_fn=lambda model: compute_loss_heat_fdm(model, epsilon=1e-3),
        epochs=20000,
        lr=1e-3,
        log_every=2000
    )

    plot_loss_curve(
        heat_fdm_loss,
        title="Heat FDM-PINN Training Loss"
    )
    plt.show()

    heat_fdm_error = plot_heat_comparison(
        model=heat_fdm_model,
        exact_fn=heat_exact,
        label="Heat FDM-PINN"
    )
    plt.show()
    """

    # --- Problem 2.4 ---
    print("\n" + "=" * 50)
    print("Problem 2.4(b): Heat FDM-PINN epsilon study")
    print("=" * 50)

    # 2.4(b)
    epsilon_values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

    heat_fdm_eps_errors = []
    heat_fdm_eps_losses = []
    heat_fdm_eps_times = []

    for eps in epsilon_values:
        print(f"\nTraining Heat FDM-PINN with epsilon = {eps}")

        model_eps = PINN(
            input_dim=2,
            hidden_dim=64,
            num_layers=4,
            output_dim=1
        ).to(device)

        loss_eps, time_eps = train_pinn(
            model=model_eps,
            loss_fn=lambda model, eps=eps: compute_loss_heat_fdm(model, epsilon=eps),
            epochs=20000,
            lr=1e-3,
            log_every=2000
        )

        rel_l2_eps = plot_heat_comparison(
            model=model_eps,
            exact_fn=heat_exact,
            label=f"Heat FDM-PINN eps={eps}"
        )

        plt.show()

        heat_fdm_eps_errors.append(rel_l2_eps)
        heat_fdm_eps_losses.append(loss_eps[-1])
        heat_fdm_eps_times.append(time_eps)

    # Plot relative L2 error vs epsilon
    plt.figure(figsize=(7, 5))
    plt.loglog(epsilon_values, heat_fdm_eps_errors, marker="o")
    plt.xlabel("epsilon")
    plt.ylabel("Relative L2 error")
    plt.title("Heat FDM-PINN Error vs Epsilon")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 2.4(c)
    nu = 0.01
    dx = 1/64
    x = np.arange(0, 1 + dx, dx)

    r_unstable = 0.6
    dt_unstable = r_unstable * dx**2 / nu

    T_final = 0.5
    t_unstable = np.arange(0, T_final + dt_unstable, dt_unstable)

    print(f"dx = {dx}")
    print(f"dt = {dt_unstable}")
    print(f"r = {r_unstable}")

    # Initial condition
    u_unstable = np.zeros((len(t_unstable), len(x)))
    u_unstable[0, :] = np.sin(np.pi*x) + 0.5*np.sin(3*np.pi*x)

    # Boundary conditions
    u_unstable[:, 0] = 0
    u_unstable[:, -1] = 0

    # Forward Euler
    for n in range(len(t_unstable) - 1):
        for j in range(1, len(x) - 1):
            u_unstable[n+1, j] = u_unstable[n, j] + r_unstable * (
                u_unstable[n, j+1] - 2*u_unstable[n, j] + u_unstable[n, j-1]
            )

    # Plot 
    X_unstable, T_unstable = np.meshgrid(x, t_unstable)

    plt.figure(figsize=(8, 5))
    plt.pcolormesh(X_unstable, T_unstable, u_unstable, shading="auto")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title("Problem 2.4(c): Unstable Forward Euler FD Heat Solution, r = 0.6")
    plt.colorbar(label="u(x,t)")
    plt.tight_layout()
    plt.show()
    
    
    # --- Problem 3 ---

    # 3(b)
    Nr_values = [100, 500, 2000, 10000]

    ode_ad_Nr_errors = []
    ode_fdm_Nr_errors = []

    for Nr in Nr_values:
        print(f"\nTraining ODE AD-PINN with Nr = {Nr}")

        model_ad = PINN(
            input_dim=1,
            hidden_dim=32,
            num_layers=3,
            output_dim=1
        ).to(device)

        loss_ad, time_ad = train_pinn(
            model=model_ad,
            loss_fn=lambda model, Nr=Nr: compute_loss_ode_ad_Nr(model, Nr=Nr),
            epochs=10000,
            lr=1e-3,
            log_every=2000
        )

        err_ad = plot_ode_comparison(
            model=model_ad,
            exact_fn=ode_exact,
            t_range=(0, 5),
            label=f"AD-PINN Nr={Nr}"
        )
        plt.show()

        ode_ad_Nr_errors.append(err_ad)

        print(f"\nTraining ODE FDM-PINN with Nr = {Nr}")

        model_fdm = PINN(
            input_dim=1,
            hidden_dim=32,
            num_layers=3,
            output_dim=1
        ).to(device)

        loss_fdm, time_fdm = train_pinn(
            model=model_fdm,
            loss_fn=lambda model, Nr=Nr: compute_loss_ode_fdm_Nr(model, Nr=Nr, epsilon=1e-3),
            epochs=10000,
            lr=1e-3,
            log_every=2000
        )

        err_fdm = plot_ode_comparison(
            model=model_fdm,
            exact_fn=ode_exact,
            t_range=(0, 5),
            label=f"FDM-PINN Nr={Nr}"
        )
        plt.show()

        ode_fdm_Nr_errors.append(err_fdm)

    # Plot final error vs Nr
    plt.figure(figsize=(7, 5))
    plt.loglog(Nr_values, ode_ad_Nr_errors, marker="o", label="AD-PINN")
    plt.loglog(Nr_values, ode_fdm_Nr_errors, marker="o", label="FDM-PINN")
    plt.xlabel("Number of collocation points, Nr")
    plt.ylabel("Max absolute error")
    plt.title("ODE Error vs Collocation Points")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 3(c)
    


    # --- Summary ---
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"{'Method':<25} {'Problem':<10} {'Error':<15} {'Time (s)':<10}")
    print("-" * 60)
    ## TODO: Print a summary table comparing the 4 methods (ODE-AD, ODE-FDM, Heat-AD, Heat-FDM) in terms of max error and training time.


    print("\nDone! All plots saved.")
