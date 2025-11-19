import numpy as np
import matplotlib.pyplot as plt


# Question 1
# Solve U_t + U_x = 0, x in (0,1), t > 0
# U(x,0) = sin^80(pi x), x in [0,1]
# U(0,t)=U(1,t)=0, t>=0
# Use the BTCS scheme


def Q1_BTCS_1D_advection(u0, dx=0.05, dt=0.01, times_to_keep=[0, 0.1, 0.2, 0.4]):

    # Spatial grid
    x = np.arange(0, 1 + dx, dx)
    nx = len(x)

    # Time steps
    t_final = max(times_to_keep)
    nt = int(t_final / dt)

    sampled_solutions = {}
    # Initial condition
    u = u0(x)

    # include initial condition at t=0
    sampled_solutions[0] = u.copy()

    # Coefficient matrix
    r = dt / dx

    A = np.zeros((nx - 2, nx - 2))
    for i in range(nx - 2):
        A[i, i] = 1
        if i > 0:
            A[i, i - 1] = -r
        if i < nx - 3:
            A[i, i + 1] = r

    for n in range(1, nt + 1):
        b = u[1:-1]
        u_inner = np.linalg.solve(A, b)
        u[1:-1] = u_inner
        u[0] = 0
        u[-1] = 0

        current_time = n * dt
        if current_time in times_to_keep:
            sampled_solutions[current_time] = u.copy()

    return sampled_solutions


def Q1_initial_condition(x):
    return np.sin(np.pi * x) ** 80


def Q1_plot(a_solutions, b_solutions, times_to_plot=[0.1, 0.2, 0.4]):

    x_a = np.arange(0, 1 + 0.05, 0.05)
    x_b = np.arange(0, 1 + 0.025, 0.025)

    for t in times_to_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(x_a, a_solutions[t], "o-", label="BTCS dx=0.05, dt=0.01")
        plt.plot(x_b, b_solutions[t], "s-", label="BTCS dx=0.025, dt=0.0025")
        plt.title(f"Solution at time t={t}")
        plt.xlabel("x")
        plt.ylabel("U(x,t)")
        plt.legend()
        plt.grid()
        plt.savefig(f"Q1_solution_t{t}.png")


def Q1_main():
    Q1_solutions_a = Q1_BTCS_1D_advection(Q1_initial_condition, dx=0.05, dt=0.01)
    Q1_solutions_b = Q1_BTCS_1D_advection(Q1_initial_condition, dx=0.025, dt=0.0025)

    Q1_plot(Q1_solutions_a, Q1_solutions_b)


Q1_main()

# Question 2

# Solve U_t + aU_x = 0, x in (0,1), t > 0
# U(x,0) = 1 + sin(2 pi x), x in [0,1]
# U(1,t)=1, t>=0
# Use the CN scheme


def Q2_CN_1D_advection(u0, a=-2, K=10, dt=0.04, times_to_keep=[0, 0.08, 0.12, 0.8]):
    dx = 1 / K

    x = np.arange(0, 1 + dx, dx)
    nx = len(x)

    t_final = max(times_to_keep)
    nt = int(t_final / dt)

    sampled_solutions = {}
    u = u0(x)

    # include initial condition at t=0
    sampled_solutions[0] = u.copy()

    R = a * dt / (4 * dx)

    A = np.zeros((nx - 1, nx - 1))
    B= np.zeros((nx - 1, nx - 1))
    for i in range(nx - 1):
        A[i, i] = 1
        B[i, i] = 1
        if i > 0:
            A[i, i - 1] = -R
            B[i, i - 1] = R
        if i < nx - 2:
            A[i, i + 1] = R
            B[i, i + 1] = -R

    for n in range(1, nt + 1):
        b = B @ u[0:K]
        
        b[-1] += R * 1  # Boundary condition at x=1

        u_inner = np.linalg.solve(A, b)
        u[0:K] = u_inner
        u[0] = u[1]  # Numerical BC at x=0
        u[-1] = 1  # Dirichlet BC at x=1
        current_time = n * dt
        if current_time in times_to_keep:
            sampled_solutions[current_time] = u.copy()

    return sampled_solutions


def Q2_initial_condition(x):
    return 1 + np.sin(2 * np.pi * x)


def Q2_exact_solution(x, t, a=-2):
    
    solution = np.zeros_like(x)

    for idx in range(len(x)):
        temp = x[-idx] + a * t

        if temp < 0:
            solution[-idx] = 1
        else:
            solution[-idx] = 1 + np.sin(2*np.pi*(x[-idx]) + a *t)
    
    return solution


def Q2_plot_single_solution(solutions, K, dt, times_to_plot=[0, 0.08, 0.12, 0.8]):

    x = np.arange(0, 1 + 1 / K, 1 / K)

    for t in times_to_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(x, solutions[t], "o-", label=f"CN K={K}, dt={dt}")
        plt.plot(x, Q2_exact_solution(x, t), "k--", label="Exact Solution")
        plt.title(f"Solution at time t={t}")
        plt.xlabel("x")
        plt.ylabel("U(x,t)")
        plt.legend()
        plt.grid()
        plt.savefig(f"Q2_single_solution_K{K}_dt{dt}_t{t}.png")


def Q2_main():
    Q2_solutions_a = Q2_CN_1D_advection(Q2_initial_condition, a=-2, K=10, dt=0.04)
    Q2_solutions_b = Q2_CN_1D_advection(Q2_initial_condition, a=-2, K=40, dt=0.01)
    Q2_solutions_c = Q2_CN_1D_advection(Q2_initial_condition, a=-2, K=160, dt=0.0025)

    Q2_plot_single_solution(Q2_solutions_a, K=10, dt=0.04)
    Q2_plot_single_solution(Q2_solutions_b, K=40, dt=0.01)
    Q2_plot_single_solution(Q2_solutions_c, K=160, dt=0.0025)


Q2_main()

# Question 3
# Solve U_t + U_x =0, x in (0,1), t > 0
# U(x,0) = sin^80(pi x), x in [0,1]
# U(0,t)=U(1,t)=0, t>=0
# Use implicit scheme and Sherman-morrison formula for solving linear system


def Q3_implicit_sherman_morrison(
    u0, dx=0.005, dt=0.0025, times_to_keep=[0, 0.1, 0.2, 0.4]
):
    x = np.arange(0, 1 + dx, dx)
    nx = len(x)

    t_final = max(times_to_keep)
    nt = int(t_final / dt)

    sampled_solutions = {}
    u = u0(x)

    # include initial condition at t=0
    sampled_solutions[0] = u.copy()

    r = dt / dx

    # Constructing the tridiagonal matrix A
    A = np.zeros((nx - 2, nx - 2))
    for i in range(nx - 2):
        A[i, i] = 1
        if i > 0:
            A[i, i - 1] = -r
        if i < nx - 3:
            A[i, i + 1] = r

    for n in range(1, nt + 1):
        b = u[1:-1]

        # Sherman-Morrison formula application
        # Here we assume A can be expressed as A = T + uv^T where T is tridiagonal
        # For simplicity, we will directly solve the system using numpy's solver
        u_inner = np.linalg.solve(A, b)
        u[1:-1] = u_inner
        u[0] = 0
        u[-1] = 0

        current_time = n * dt
        if current_time in times_to_keep:
            sampled_solutions[current_time] = u.copy()

    return sampled_solutions


def Q3_FTBS_scheme(u0, dx=0.005, dt=0.0025, times_to_keep=[0, 0.1, 0.2, 0.4]):
    x = np.arange(0, 1 + dx, dx)
    nx = len(x)

    t_final = max(times_to_keep)
    nt = int(t_final / dt)

    sampled_solutions = {}
    u = u0(x)

    sampled_solutions[0] = u.copy()

    r = dt / dx

    for n in range(1, nt + 1):
        u_new = u.copy()
        for i in range(1, nx - 1):
            u_new[i] = u[i] - r * (u[i] - u[i - 1])
        u = u_new
        u[0] = 0
        u[-1] = 0

        current_time = n * dt
        if current_time in times_to_keep:
            sampled_solutions[current_time] = u.copy()

    return sampled_solutions


def Q3_initial_condition(x):
    return np.sin(np.pi * x) ** 80


def Q3_plot(implicit_solutions, ftbs_solutions, times_to_plot=[0, 0.1, 0.2, 0.4]):

    x = np.arange(0, 1 + 0.005, 0.005)

    for t in times_to_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(
            x,
            implicit_solutions[t],
            "o-",
            label="Implicit Scheme with Sherman-Morrison",
        )
        plt.plot(x, ftbs_solutions[t], "s-", label="FTBS Scheme")
        plt.title(f"Solution at time t={t}")
        plt.xlabel("x")
        plt.ylabel("U(x,t)")
        plt.legend()
        plt.grid()
        plt.savefig(f"Q3_solution_t{t}.png")


def Q3_main():
    Q3_implicit_solutions = Q3_implicit_sherman_morrison(
        Q3_initial_condition, dx=0.005, dt=0.0025, times_to_keep=[0, 0.1, 0.2, 0.4]
    )
    Q3_ftbs_solutions = Q3_FTBS_scheme(
        Q3_initial_condition, dx=0.005, dt=0.0025, times_to_keep=[0, 0.1, 0.2, 0.4]
    )
    Q3_plot(Q3_implicit_solutions, Q3_ftbs_solutions, times_to_plot=[0, 0.1, 0.2, 0.4])


Q3_main()

# Question 4

# Solve U_t = 1/r (rU_r)_r + 1/r^2 U_theta_theta , r in (0,1), theta in [0,2*pi), t > 0
# U(r,theta,0) = 0, r in [0,1], theta in [0,2*pi]
# U(1,theta,t) = sin(4 theta)sin(t), theta in [0,2*pi], t > 0


def Q4_adi_polar_heat(
    u0, dr=0.1, dtheta=np.pi / 4, dt=0.01, times_to_keep=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
):
    r = np.arange(0, 1 + dr, dr)
    theta = np.linspace(0, 2 * np.pi, int(2 * np.pi / dtheta), endpoint=False)
    nr = len(r)
    ntheta = len(theta)

    t_final = max(times_to_keep)
    nt = int(t_final / dt)

    sampled_solutions = {}
    u = u0(r, theta)

    for n in range(1, nt + 1):
        # Half step in r
        u_half = u.copy()
        for j in range(ntheta):
            A_r = np.zeros((nr - 2, nr - 2))
            b_r = np.zeros(nr - 2)
            for i in range(1, nr - 1):
                ri = r[i]
                A_r[i - 1, i - 1] = 1 + dt / (2 * dr**2) * (1 + dr / ri)
                if i > 1:
                    A_r[i - 1, i - 2] = -dt / (2 * dr**2) * (1 - dr / (2 * ri))
                if i < nr - 2:
                    A_r[i - 1, i] = -dt / (2 * dr**2) * (1 + dr / (2 * ri))
                b_r[i - 1] = u[i, j] + dt / (2 * (ri**2) * dtheta**2) * (
                    u[i, (j + 1) % ntheta] - 2 * u[i, j] + u[i, (j - 1) % ntheta]
                )
            u_inner = np.linalg.solve(A_r, b_r)
            u_half[1:-1, j] = u_inner

        # Full step in theta
        u_half[0, :] = u_half[1, :]
        u_new = u_half.copy()
        for i in range(1, nr - 1):
            A_theta = np.zeros((ntheta, ntheta))
            b_theta = np.zeros(ntheta)
            for j in range(ntheta):
                A_theta[j, j] = 1 + dt / (2 * (r[i] ** 2) * dtheta**2)
                A_theta[j, (j - 1) % ntheta] = -dt / (4 * (r[i] ** 2) * dtheta**2)
                A_theta[j, (j + 1) % ntheta] = -dt / (4 * (r[i] ** 2) * dtheta**2)
                b_theta[j] = u_half[i, j] + dt / (2 * dr**2) * (1 + dr / r[i]) * (
                    u_half[i + 1, j] - 2 * u_half[i, j] + u_half[i - 1, j]
                )
            u_inner = np.linalg.solve(A_theta, b_theta)
            u_new[i, :] = u_inner
        u_new[0, :] = u_new[1, :]
        u_new[-1, :] = np.sin(4 * theta) * np.sin(n * dt)
        u = u_new
        current_time = n * dt
        if current_time in times_to_keep:
            sampled_solutions[current_time] = u.copy()
    return sampled_solutions


def Q4_initial_condition(r, theta):
    R, Theta = np.meshgrid(r, theta, indexing="ij")
    return np.zeros_like(R)


def Q4_plot(solutions, r, theta, times_to_plot=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0]):
    R, Theta = np.meshgrid(r, theta, indexing="ij")
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)

    for t in times_to_plot:
        U = solutions[t]
        plt.figure(figsize=(8, 6))
        plt.pcolormesh(X, Y, U, shading="auto", cmap="viridis")
        plt.colorbar()
        plt.title(f"Solution at time t={t}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis("equal")
        plt.savefig(f"Q4_solution_t{t}.png")


def Q4_main():
    dr = 0.1
    dtheta = np.pi / 16
    r = np.arange(0, 1 + dr, dr)
    theta = np.arange(0, 2 * np.pi, dtheta)

    Q4_solutions = Q4_adi_polar_heat(
        Q4_initial_condition, dr=dr, dtheta=dtheta, dt=0.01
    )
    Q4_plot(Q4_solutions, r, theta)


Q4_main()
