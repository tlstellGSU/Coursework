import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  

# Question 1

# use FTCS to find the solutions to the viscous Burgers equation
# u_t + u*u_x = nu*u_xx
# x in (0,1), t > 0

# initial condition: u(x,0) = sin(2*pi*x), x in [0,1]
# boundary condition: u(0,t) = u(1,t) = 0, t >= 0

test_nu = [1, 0.1, 0.001, 0.00001]

T = 0.1

dx = 0.01
x = np.arange(0, 1+dx, dx)

sample_t_values_1 = [T/4, T/2, 3*T/4, T]

def FTCS_burgers(dx=dx, x=x, nu=test_nu[0], sample_t_values=sample_t_values_1):

    dt = min(dx, dx**2/(2*nu)) * 0.5
    t=np.arange(0, T+dt, dt)
    r = nu*dt/dx**2
    R = dt/dx
    u = np.sin(2*np.pi*x)  # initial condition
    all_u = [u.copy()]

    sampled_u = [u.copy()]  # store initial condition
    sampled_t_values = [0] + sample_t_values  # include t=0

    for n in range(1, len(t)):
        u_new = np.zeros_like(u)
        for i in range(1, len(x)-1):
            u_new[i] = u[i] - R*u[i]*(u[i]-u[i-1]) + r*(u[i+1]-2*u[i]+u[i-1])
        u[0] = 0
        u[-1] = 0
        u = u_new
        all_u.append(u.copy())

        for t_sample in sample_t_values:
            if abs(t[n] - t_sample) < dt/2:
                sampled_u.append(u.copy())
                break

    return sampled_u, sampled_t_values, all_u, t

def plot_results_q1(nu, save_path=None):

    sampled_u, sampled_t_values, _, _ = FTCS_burgers(dx, x, nu=nu, sample_t_values=sample_t_values_1)
    plt.figure(figsize=(8, 5))
    for i in range(len(sampled_u)):

        t_val = sampled_t_values[i]
        u = sampled_u[i]

        plt.plot(x, u, label=f"t={t_val:.2f}")
        plt.xlabel("x")
        plt.ylabel("u(x,t)")
        plt.title(f"FTCS Scheme for Burgers Equation (nu={nu})")
        plt.legend()
    plt.grid()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def contour_plot_q1(nu, save_path=None):

    _, _, all_u, t = FTCS_burgers(dx, x, nu=nu, sample_t_values=np.linspace(0, T, 100))

    plt.figure(figsize=(8, 5))
    X, Y = np.meshgrid(x, t)
    cp = plt.contourf(X, Y, all_u, levels=50, cmap='viridis')
    plt.colorbar(cp)
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title(f"Contour Plot of u(x,t) for Burgers Equation (nu={nu})")
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def surface_plot_q1(nu, save_path=None):
    _, _, all_u, t = FTCS_burgers(dx, x, nu=nu, sample_t_values=np.linspace(0, T, 100))
    X, Y = np.meshgrid(x, t)

    U = np.array(all_u)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, U, cmap='viridis', edgecolor='none')
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_zlabel("u(x,t)")
    ax.set_title(f"3D Surface Plot of u(x,t) for Burgers Equation (nu={nu})")
    fig.colorbar(surf, shrink=0.5, aspect=10)

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

for nu in test_nu:
    plot_results_q1(nu, save_path=f'Burgers_nu{nu}.png')
    contour_plot_q1(nu, save_path=f'Burgers_Contour_nu{nu}.png')
    surface_plot_q1(nu, save_path=f'Burgers_Surface_nu{nu}.png')

# Question 3
# use CN scheme to approximate the solution to the equation
# u_t = nu*u_xx, x in (0,1), t > 0
# initial condition: u(x,0) = sin(2*pi*x), x in [0,1]
# boundary condition: u(0,t) = u(1,t) = 0, t >= 0

# compare to BTCS and FTCS

sampled_t_values_3 = [0.01, 0.1, 1, 100]

del_t_3 = 0.01
del_x_3 = 0.1
nu_3 = 1/6

x_3 = np.arange(0, 1+del_x_3, del_x_3)
t_3 = np.arange(0, 100+del_t_3, del_t_3)

u_0 = np.sin(2*np.pi*x_3)

def CN_scheme(dx, dt, x, t=t_3, nu=nu_3):

    r = nu * dt / (2 * dx**2)
    u = u_0.copy()
    N = len(x)
    A = np.zeros((N-2, N-2))

    sampled_u = [u.copy()]  # store initial condition
    sampled_t_values = [0] + sampled_t_values_3  # include t=0

    for i in range(N-2):
        if i > 0:
            A[i, i-1] = -r
        A[i, i] = 1 + 2*r
        if i < N-3:
            A[i, i+1] = -r
    
    for n in range(1, len(t)):
        b = np.zeros(N-2)
        for i in range(1, N-1):
            b[i-1] = r*u[i-1] + (1 - 2*r)*u[i] + r*u[i+1]
        u_inner = np.linalg.solve(A, b)
        u[1:N-1] = u_inner
        u[0] = 0
        u[-1] = 0
    
        for t_sample in sampled_t_values_3:
            if abs(t[n] - t_sample) < dt/2:
                sampled_u.append(u.copy())
                break

    return sampled_u, sampled_t_values

def BTCS_scheme(dx, dt, x, t=t_3, nu=nu_3):

    r = nu * dt / dx**2
    u = u_0.copy()
    N = len(x)
    A = np.zeros((N-2, N-2))

    sampled_u = [u.copy()]  # store initial condition
    sampled_t_values = [0] + sampled_t_values_3  # include t=0

    for i in range(N-2):
        if i > 0:
            A[i, i-1] = -r
        A[i, i] = 1 + 2*r
        if i < N-3:
            A[i, i+1] = -r
    
    for n in range(1, len(t)):
        b = u[1:-1]
        u_inner = np.linalg.solve(A, b)
        u[1:N-1] = u_inner
        u[0] = 0
        u[-1] = 0
    
        for t_sample in sampled_t_values_3:
            if abs(t[n] - t_sample) < dt/2:
                sampled_u.append(u.copy())
                break

    return sampled_u, sampled_t_values

def FTCS_scheme(dx, dt, x, t=t_3, nu=nu_3):

    r = nu * dt / dx**2
    u = u_0.copy()
    N = len(x)

    sampled_u = [u.copy()]  # store initial condition
    sampled_t_values = [0] + sampled_t_values_3  # include t=0

    for n in range(1, len(t)):
        u_new = np.zeros_like(u)
        for i in range(1, N-1):
            u_new[i] = u[i] + r*(u[i+1]-2*u[i]+u[i-1])
        u[0] = 0
        u[-1] = 0
        u = u_new

        for t_sample in sampled_t_values_3:
            if abs(t[n] - t_sample) < dt/2:
                sampled_u.append(u.copy())
                break

    return sampled_u, sampled_t_values

def exact_solution(x, t, nu=nu_3):
    return np.exp(-4*np.pi**2*nu*t) * np.sin(2*np.pi*x)

def plot_results_q3(dx=del_x_3, dt=del_t_3, x=x_3, nu=nu_3, save_path_temp=None):

    method_names = ["Crank-Nicolson", "BTCS", "FTCS"]
    
    sampled_u_CN, sampled_t_values_CN = CN_scheme(dx, dt, x, nu=nu)
    sampled_u_BTCS, sampled_t_values_BTCS = BTCS_scheme(dx, dt, x, nu=nu)
    sampled_u_FTCS, sampled_t_values_FTCS = FTCS_scheme(dx, dt, x, nu=nu)

    exact_u = [exact_solution(x, t_val, nu=nu) for t_val in sampled_t_values_CN]

    difference_CN = [np.linalg.norm(u - u_exact, ord=np.inf) for u, u_exact in zip(sampled_u_CN, exact_u)]
    difference_BTCS = [np.linalg.norm(u - u_exact, ord=np.inf) for u, u_exact in zip(sampled_u_BTCS, exact_u)]
    difference_FTCS = [np.linalg.norm(u - u_exact, ord=np.inf) for u, u_exact in zip(sampled_u_FTCS, exact_u)]

    for method_name, sampled_u, sampled_t_values in zip(method_names, [sampled_u_CN, sampled_u_BTCS, sampled_u_FTCS], [sampled_t_values_CN, sampled_t_values_BTCS, sampled_t_values_FTCS]):
        plt.figure(figsize=(8, 5))
        for i in range(len(sampled_u)):

            t_val = sampled_t_values[i]
            u = sampled_u[i]

            plt.plot(x, u, label=f"t={t_val:.2f}")
            plt.xlabel("x")
            plt.ylabel("u(x,t)")
            plt.title(f"{method_name} Scheme for Heat Equation (nu=1/6)")
            plt.legend()
        plt.grid()
        if save_path_temp:
            plt.savefig(save_path_temp + f'{method_name}_Scheme.png')
        else:
            plt.show()

    plt.close('all')

    # plot the exact solution at the same time stamps
    plt.figure(figsize=(8,5))
    for i in range(len(exact_u)):
        t_val = sampled_t_values_CN[i]
        u_exact = exact_u[i]
        plt.plot(x, u_exact, label=f"t={t_val:.2f}")
    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    plt.title("Exact Solution for Heat Equation (nu=1/6)")
    plt.legend()
    plt.grid()
    if save_path_temp:
        plt.savefig(save_path_temp + 'Exact_Solution.png')
    else:
        plt.show()

    plt.close('all')

    # plot the difference between the numerical and exact solutions
    plt.figure(figsize=(8,5))
    plt.semilogy(sampled_t_values_CN, difference_CN, label="Crank-Nicolson")
    plt.semilogy(sampled_t_values_BTCS, difference_BTCS, label="BTCS")
    plt.semilogy(sampled_t_values_FTCS, difference_FTCS, label="FTCS")
    plt.xlabel("Time t")
    plt.ylabel("Max Norm of Error")  
    plt.title("Error Comparison of Numerical Schemes")    
    plt.legend()
    plt.grid()
    if save_path_temp:
        plt.savefig(save_path_temp + 'Error_Comparison.png')
    else:
        plt.show()

plot_results_q3(save_path_temp=f'HW_3_q3_Heat_')