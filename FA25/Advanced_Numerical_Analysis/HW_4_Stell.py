import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.integrate as integrate

# Q 3
# Use BTCS to solve vicous burgers equation
# u_t + u*u_x = nu*u_xx
# assume u(x,0) = sin(pi*x), x in [0,1]
# BCS: u(0,t) = u(1,t) = 0, t>0


# part a) lag nonlinear term
# part b) linearize about previous time step
# part c) Newton's method


# part a) lag nonlinear term
def BTCS_with_lag_term(dx, dt, x, t, nu = 1, u_0 = None, sampled_t_values = None):
    if u_0 is None:
        u_0 = np.sin(np.pi * x)  # initial condition
    if sampled_t_values is None:
        sampled_t_values = [0.1, 0.2, 0.3, 0.4, 0.5]  # times to sample solution

    r = nu * dt / dx**2
    u = u_0.copy()
    N = len(x)
    A = np.zeros((N-2, N-2))

    sampled_u = [u.copy()] 
    sampled_t_values_full = [0] + sampled_t_values  if 0 not in sampled_t_values else sampled_t_values

    for i in range(N-2):
        if i > 0:
            A[i, i-1] = -r
        A[i, i] = 1 + 2*r
        if i < N-3:
            A[i, i+1] = -r
    
    for n in range(1, len(t)):
        b = np.zeros(N-2)
        for i in range(1, N-1):
            b[i-1] = u[i] - dt * u[i] * (u[i+1] - u[i-1]) / (2*dx) + r * (u[i-1] - 2*u[i] + u[i+1])
        u_inner = np.linalg.solve(A, b)
        u[1:N-1] = u_inner
        u[0] = 0
        u[-1] = 0
    
        for t_sample in sampled_t_values:
            if abs(t[n] - t_sample) < dt/2:
                sampled_u.append(u.copy())
                break

    return sampled_u, sampled_t_values_full

# part b) linearize about previous time step

def BTCS_with_linearization(dx, dt, x, t, nu = 1, u_0 = None, sampled_t_values = None):
    if u_0 is None:
        u_0 = np.sin(np.pi * x)  # initial condition
    if sampled_t_values is None:
        sampled_t_values = [0.1, 0.2, 0.3, 0.4, 0.5]  # times to sample solution

    r = nu * dt / dx**2
    u = u_0.copy()
    N = len(x)

    sampled_u = [u.copy()] 
    sampled_t_values_full = [0] + sampled_t_values  if 0 not in sampled_t_values else sampled_t_values

    for n in range(1, len(t)):
        A = np.zeros((N-2, N-2))
        b = np.zeros(N-2)
        for i in range(1, N-1):
            if i > 1:
                A[i-1, i-2] = - (dt * u[i-1]) / (2*dx) - r
            A[i-1, i-1] = 1 + (dt * (u[i+1] - u[i-1])) / (2*dx) + 2*r
            if i < N-2:
                A[i-1, i] = (dt * u[i+1]) / (2*dx) - r
            b[i-1] = u[i]
        u_inner = np.linalg.solve(A, b)
        u[1:N-1] = u_inner
        u[0] = 0
        u[-1] = 0
    
        for t_sample in sampled_t_values:
            if abs(t[n] - t_sample) < dt/2:
                sampled_u.append(u.copy())
                break

    return sampled_u, sampled_t_values_full

# part c) Newton's method

def BTCS_with_Newton(dx, dt, x, t, nu = 1, u_0 = None, sampled_t_values = None, tol=1e-6, max_iter=50):
    if u_0 is None:
        u_0 = np.sin(np.pi * x)  # initial condition
    if sampled_t_values is None:
        sampled_t_values = [0.1, 0.2, 0.3, 0.4, 0.5]  # times to sample solution

    r = nu * dt / dx**2
    u = u_0.copy()
    N = len(x)

    sampled_u = [u.copy()] 
    sampled_t_values_full = [0] + sampled_t_values  if 0 not in sampled_t_values else sampled_t_values

    for n in range(1, len(t)):
        u_old = u.copy()
        for iteration in range(max_iter):
            F = np.zeros(N-2)
            J = np.zeros((N-2, N-2))
            for i in range(1, N-1):
                F[i-1] = u[i] - u_old[i] + dt * u[i] * (u[i+1] - u[i-1]) / (2*dx) - r * (u[i-1] - 2*u[i] + u[i+1])
                if i > 1:
                    J[i-1, i-2] = - (dt * u[i]) / (2*dx) - r
                J[i-1, i-1] = 1 + (dt * (u[i+1] - u[i-1])) / (2*dx) + 2*r
                if i < N-2:
                    J[i-1, i] = (dt * u[i]) / (2*dx) - r
            delta_u = np.linalg.solve(J, -F)
            u[1:N-1] += delta_u
            if np.linalg.norm(delta_u, np.inf) < tol:
                break
        u[0] = 0
        u[-1] = 0
    
        for t_sample in sampled_t_values:
            if abs(t[n] - t_sample) < dt/2:
                sampled_u.append(u.copy())
                break

    return sampled_u, sampled_t_values_full

# implementation


def plot_solution_per_method(dx, dt, x, t, nu=1, method=None, sampled_t_values=None):
    if method == 'lag':
        u_samples, t_values = BTCS_with_lag_term(dx, dt, x, t, nu, sampled_t_values=sampled_t_values)
        method_name = 'BTCS with Lag Nonlinear Term'
    elif method == 'linearization':
        u_samples, t_values = BTCS_with_linearization(dx, dt, x, t, nu, sampled_t_values=sampled_t_values)
        method_name = 'BTCS with Linearization'
    elif method == 'newton':
        u_samples, t_values = BTCS_with_Newton(dx, dt, x, t, nu, sampled_t_values=sampled_t_values)
        method_name = 'BTCS with Newton\'s Method'
    else:
        raise ValueError("Method must be 'lag', 'linearization', or 'newton'.")

    plt.figure(figsize=(10, 6))
    for i, t_sample in enumerate(t_values):
        plt.plot(x, u_samples[i], label=f'Time = {t_sample:.2f}')
    plt.title(method_name)
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.ylim(-1, 1)
    plt.legend()
    plt.grid()
    plt.savefig(f'HW4_{method}_solution.png')
    #plt.show()

def plot_solution_schemes(dx, dt, x, t, nu=1, sampled_t_values=None):
    # plots all three methods on same figure per time step
    u_0 = np.sin(np.pi * x)
    if sampled_t_values is None:
        sampled_t_values = [0.1, 0.2, 0.3, 0.4, 0.5]

    u_lag, t_values = BTCS_with_lag_term(dx, dt, x, t, nu, u_0, sampled_t_values)
    u_linearized, _ = BTCS_with_linearization(dx, dt, x, t, nu, u_0, sampled_t_values)
    u_newton, _ = BTCS_with_Newton(dx, dt, x, t, nu, u_0, sampled_t_values)

    plt.figure(figsize=(15, 10))
    for i, t_sample in enumerate(t_values):
        plt.subplot(3, 2, i+1)
        plt.plot(x, u_lag[i], label='Lag Nonlinear Term', linestyle='--')
        plt.plot(x, u_linearized[i], label='Linearization', linestyle='-.')
        plt.plot(x, u_newton[i], label='Newton\'s Method', linestyle=':')
        plt.title(f'Solution at Time = {t_sample:.2f}')
        plt.xlabel('x')
        plt.ylabel('u(x,t)')
        plt.ylim(-1, 1)
        plt.legend()
        plt.grid()

    plt.savefig('HW4_BTCS_schemes_comparison.png')
    plt.tight_layout()
    #plt.show()

if __name__ == "__main__":
    L = 1.0
    T = 1.0
    dx = 0.01
    dt = 0.001
    x = np.arange(0, L + dx, dx)
    t = np.arange(0, T + dt, dt)
    nu = 0.5

    sampled_t_values = [0.01, 0.05, 0.1, 0.15, 0.2]


    print("Plotting solution schemes...")
    plot_solution_schemes(dx, dt, x, t, nu, sampled_t_values)

    print("Plotting solutions per method...")
    plot_solution_per_method(dx, dt, x, t, nu, method='lag', sampled_t_values=sampled_t_values)
    plot_solution_per_method(dx, dt, x, t, nu, method='linearization', sampled_t_values=sampled_t_values)
    plot_solution_per_method(dx, dt, x, t, nu, method='newton', sampled_t_values=sampled_t_values)
