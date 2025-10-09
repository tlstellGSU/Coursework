import numpy as np
import matplotlib.pyplot as plt

# Question 2
nu = 0.1
exact_solution = lambda x,t: np.sin(2*np.pi*(x))*np.exp(-4*nu*np.pi**2*t)

# this scheme will return only the final numerical vector solution

def FTCS_scheme(dx, dt, x, t=0.1, nu=0.1):
    r = nu*dt/dx**2
    u = np.sin(2*np.pi*x)  # initial condition
    for n in range(1, len(t)):
        u_new = np.zeros_like(u)
        for i in range(1, len(x)-1):
            u_new[i] = u[i] + r*(u[i+1] - 2*u[i] + u[i-1])
        u[0] = 0
        u[-1] = 0
        u = u_new
    return u

def plot_results(x, u_num, u_exact, title, save_path=None):
    plt.plot(x, u_num, 'o-', label='Numerical')
    plt.plot(x, u_exact, 'r--', label='Exact')
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.title(title)
    plt.legend()
    plt.grid()
    if save_path:
        plt.savefig(save_path)
    plt.show()

# part a
# delta x = 0.1, delta t = 0.05

dx_2a = 0.1
dt_2a = 0.05
x_2a = np.arange(0, 1+dx_2a, dx_2a)
t_2a = np.arange(0, 0.1+dt_2a, dt_2a)

u_2a = FTCS_scheme(dx_2a, dt_2a, x_2a, t_2a, nu)
u_exact_2a = exact_solution(x_2a, 0.1)
error_2a = np.linalg.norm(u_2a - u_exact_2a, ord=np.inf)

plot_results(x_2a, u_2a, u_exact_2a, f'FTCS Scheme (dx={dx_2a}, dt={dt_2a})\nError: {error_2a:.4e}', 'FTCS_dx0.1_dt0.05.png')

# part b
# delta x = 0.1/2, delta t = 0.05/4

dx_2b = dx_2a / 2
dt_2b = dt_2a / 4
x_2b = np.arange(0, 1+dx_2b, dx_2b)
t_2b = np.arange(0, 0.1+dt_2b, dt_2b)

u_2b = FTCS_scheme(dx_2b, dt_2b, x_2b, t_2b, nu)
u_exact_2b = exact_solution(x_2b, 0.1)
error_2b = np.linalg.norm(u_2b - u_exact_2b, ord=np.inf)

plot_results(x_2b, u_2b, u_exact_2b, f'FTCS Scheme (dx={dx_2b}, dt={dt_2b})\nError: {error_2b:.4e}', 'FTCS_dx0.05_dt0.0125.png')

# part c
# delta x = 0.1/4, delta t = 0.05/16

dx_2c = dx_2b / 2
dt_2c = dt_2b / 4
x_2c = np.arange(0, 1+dx_2c, dx_2c)
t_2c = np.arange(0, 0.1+dt_2c, dt_2c)

u_2c = FTCS_scheme(dx_2c, dt_2c, x_2c, t_2c, nu)
u_exact_2c = exact_solution(x_2c, 0.1)
error_2c = np.linalg.norm(u_2c - u_exact_2c, ord=np.inf)

plot_results(x_2c, u_2c, u_exact_2c, f'FTCS Scheme (dx={dx_2c}, dt={dt_2c})\nError: {error_2c:.4e}', 'FTCS_dx0.025_dt0.003125.png')

# Question 3

def BTCS_scheme(dx, dt, x, t=0.1, nu=0.1):

    r = nu*dt/dx**2
    u = np.sin(2*np.pi*x)  # initial condition
    N = len(x)
    A = np.zeros((N-2, N-2))
    for i in range(N-2):
        if i > 0:
            A[i, i-1] = -r
        A[i, i] = 1 + 2*r
        if i < N-3:
            A[i, i+1] = -r
    for n in range(1, len(t)):
        b = u[1:-1]
        u_new_inner = np.linalg.solve(A, b)
        u[1:-1] = u_new_inner
        u[0] = 0
        u[-1] = 0
    return u

def CN_scheme(dx, dt, x, t=0.1, nu=0.1):

    r = nu*dt/(2*dx**2)
    u = np.sin(2*np.pi*x)  # initial condition
    N = len(x)
    A = np.zeros((N-2, N-2))
    for i in range(N-2):
        if i > 0:
            A[i, i-1] = -r
        A[i, i] = 1 + 2*r
        if i < N-3:
            A[i, i+1] = -r
    for n in range(1, len(t)):
        b = np.zeros(N-2)
        for i in range(1, N-1):
            b[i-1] = (r)*u[i-1] + (1 - 2*r)*u[i] + (r)*u[i+1]
        u_new_inner = np.linalg.solve(A, b)
        u[1:-1] = u_new_inner
        u[0] = 0
        u[-1] = 0
    return u

# part a
# delta x = 0.1, delta t = 0.05

dx_3a = 0.1
dt_3a = 0.05
x_3a = np.arange(0, 1+dx_3a, dx_3a)
t_3a = np.arange(0, 0.1+dt_3a, dt_3a)

u_3a_BTCS = BTCS_scheme(dx_3a, dt_3a, x_3a, t_3a, nu)
u_exact_3a = exact_solution(x_3a, 0.1)
error_3a = np.linalg.norm(u_3a_BTCS - u_exact_3a, ord=np.inf)

plot_results(x_3a, u_3a_BTCS, u_exact_3a, f'BTCS Scheme (dx={dx_3a}, dt={dt_3a})\nError: {error_3a:.4e}', 'BTCS_dx0.1_dt0.05.png')

u_3a_CN = CN_scheme(dx_3a, dt_3a, x_3a, t_3a, nu)
error_3a_CN = np.linalg.norm(u_3a_CN - u_exact_3a, ord=np.inf)

plot_results(x_3a, u_3a_CN, u_exact_3a, f'Crank-Nicolson Scheme (dx={dx_3a}, dt={dt_3a})\nError: {error_3a_CN:.4e}', 'CN_dx0.1_dt0.05.png')

# part b
# delta x = 0.1/2, delta t = 0.05/2

dx_3b = dx_3a / 2
dt_3b = dt_3a / 2
x_3b = np.arange(0, 1+dx_3b, dx_3b)
t_3b = np.arange(0, 0.1+dt_3b, dt_3b)

u_3b_BTCS = BTCS_scheme(dx_3b, dt_3b, x_3b, t_3b, nu)
u_exact_3b = exact_solution(x_3b, 0.1)
error_3b = np.linalg.norm(u_3b_BTCS - u_exact_3b, ord=np.inf)

plot_results(x_3b, u_3b_BTCS, u_exact_3b, f'BTCS Scheme (dx={dx_3b}, dt={dt_3b})\nError: {error_3b:.4e}', 'BTCS_dx0.05_dt0.025.png')

u_3b_CN = CN_scheme(dx_3b, dt_3b, x_3b, t_3b, nu)
error_3b_CN = np.linalg.norm(u_3b_CN - u_exact_3b, ord=np.inf)

plot_results(x_3b, u_3b_CN, u_exact_3b, f'Crank-Nicolson Scheme (dx={dx_3b}, dt={dt_3b})\nError: {error_3b_CN:.4e}', 'CN_dx0.05_dt0.025.png')

# part c
# delta x = 0.1/4, delta t = 0.05/4

dx_3c = dx_3b / 2
dt_3c = dt_3b / 2
x_3c = np.arange(0, 1+dx_3c, dx_3c)
t_3c = np.arange(0, 0.1+dt_3c, dt_3c)

u_3c_BTCS = BTCS_scheme(dx_3c, dt_3c, x_3c, t_3c, nu)
u_exact_3c = exact_solution(x_3c, 0.1)
error_3c = np.linalg.norm(u_3c_BTCS - u_exact_3c, ord=np.inf)

plot_results(x_3c, u_3c_BTCS, u_exact_3c, f'BTCS Scheme (dx={dx_3c}, dt={dt_3c})\nError: {error_3c:.4e}', 'BTCS_dx0.025_dt0.0125.png')

u_3c_CN = CN_scheme(dx_3c, dt_3c, x_3c, t_3c, nu)
error_3c_CN = np.linalg.norm(u_3c_CN - u_exact_3c, ord=np.inf)

plot_results(x_3c, u_3c_CN, u_exact_3c, f'Crank-Nicolson Scheme (dx={dx_3c}, dt={dt_3c})\nError: {error_3c_CN:.4e}', 'CN_dx0.025_dt0.0125.png')