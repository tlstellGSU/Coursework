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


def Q1_plot(a_solutions, b_solutions, times_to_plot=[0.0, 0.1, 0.2, 0.4]):

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


def Q2_CN_advection(alpha, u0, M, dt, times_to_keep=[0.08,0.12,0.8]):
    x = np.linspace(0,1,M+1)
    dx = 1/M
    nt = int(max(times_to_keep)/dt)

    R = alpha*dt/(4*dx)

    # Build matrices A u^{n+1} = B u^n
    A = np.zeros((M-1,M-1))
    B = np.zeros((M-1,M-1))

    for i in range(M-1):
        A[i,i] = 1
        B[i,i] = 1
        if i>0:
            A[i,i-1] = -R
            B[i,i-1] =  R
        if i<M-2:
            A[i,i+1] =  R
            B[i,i+1] = -R

    # Initial condition
    u = u0(x)
    solutions = {0:u.copy()}

    for n in range(1,nt+1):
        b = B @ u[1:-1]
        # inflow boundary at x=1
        b[0]   += R*u[0]
        b[-1]  += -R*1.0  # inflow uses U(1,t)=1

        u_inner = np.linalg.solve(A,b)
        u[1:-1] = u_inner
        u[-1] = 1.0
        u[0] = u[1]  # outflow boundary

        t = n*dt
        if t in times_to_keep:
            solutions[t] = u.copy()

    return x, solutions

def Q2_initial_condition(x):
    return 1 + np.sin(2 * np.pi * x)


def Q2_exact_solution(x, t, a=-2):
    x_shifted = x - a*t      # for a=-2, this is x + 2t
    sol = np.zeros_like(x)

    for i in range(len(x)):
        xs = x_shifted[i]
        if xs > 1:        # left-moving wave hits inflow boundary
            sol[i] = 1
        else:
            sol[i] = 1 + np.sin(2*np.pi*xs)
    return sol


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
    x_a, Q2_solutions_a = Q2_CN_advection(u0=Q2_initial_condition, alpha=-2, M=10, dt=0.04)
    x_b, Q2_solutions_b = Q2_CN_advection(u0=Q2_initial_condition, alpha=-2, M=40, dt=0.01)
    x_c, Q2_solutions_c = Q2_CN_advection(u0=Q2_initial_condition, alpha=-2, M=160, dt=0.0025)

    Q2_plot_single_solution(Q2_solutions_a, K=10, dt=0.04)
    Q2_plot_single_solution(Q2_solutions_b, K=40, dt=0.01)
    Q2_plot_single_solution(Q2_solutions_c, K=160, dt=0.0025)


Q2_main()

# Question 3
# Solve U_t + U_x =0, x in (0,1), t > 0
# U(x,0) = sin^80(pi x), x in [0,1]
# U(0,t)=U(1,t)=0, t>=0
# Use implicit scheme and Sherman-morrison formula for solving linear system


def Q3_solve_tridiag(a,b,c,r):
    n = len(b)
    cp = np.zeros(n-1)
    rp = np.zeros(n)

    cp[0] = c[0]/b[0]
    rp[0] = r[0]/b[0]
    for i in range(1,n-1):

        denom = b[i] - a[i-1]*cp[i-1]
        cp[i] = c[i]/denom
        rp[i] = (r[i] - a[i-1]*rp[i-1])/denom

    rp[-1] = (r[-1] - a[-2]*rp[-2])/(b[-1] - a[-2]*cp[-2])

    x = np.zeros(n)
    x[-1] = rp[-1]
    for i in range(n-2,-1,-1):
        x[i] = rp[i] - cp[i]*x[i+1]

    return x

def Q3_implicit_sherman_morrison(u0, dx=0.005, dt=0.0025, times_to_keep=[0, 0.1, 0.2, 0.4]):
    x = np.arange(0, 1 + dx, dx)
    nx = len(x)

    t_final = max(times_to_keep)
    nt = int(t_final / dt)

    sampled_solutions = {}
    u = u0(x)

    sampled_solutions[0] = u.copy()

    r = dt / dx

    a = -r * np.ones(nx - 3)
    b = (1 + r) * np.ones(nx - 2)
    c = r * np.ones(nx - 3)

    for n in range(1, nt + 1):
        d = u[1:-1].copy()

        # Sherman-Morrison adjustment
        d[0] += r * 0  # U(0,t)=0
        d[-1] += -r * 0  # U(1,t)=0

        u_inner = Q3_solve_tridiag(a, b, c, d)
        u[1:-1] = u_inner
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
        u0=Q3_initial_condition, dx=0.005, dt=0.0025, times_to_keep=[0, 0.1, 0.2, 0.4]
    )
    Q3_ftbs_solutions = Q3_implicit_sherman_morrison(
        u0=Q3_initial_condition, dx=0.005, dt=0.0025, times_to_keep=[0, 0.1, 0.2, 0.4]
    )
    Q3_plot(Q3_implicit_solutions, Q3_ftbs_solutions, times_to_plot=[0, 0.1, 0.2, 0.4])


Q3_main()

# Question 4

# Solve U_t = 1/r (rU_r)_r + 1/r^2 U_theta_theta , r in (0,1), theta in [0,2*pi), t > 0
# U(r,theta,0) = 0, r in [0,1], theta in [0,2*pi]
# U(1,theta,t) = sin(4 theta)sin(t), theta in [0,2*pi], t > 0


def Q4_adi_polar_heat(
    u0,
    dr=0.1,
    dtheta=np.pi / 4,
    dt=0.01,
    times_to_keep=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
):
    # ------------------------------------------------------------------
    #   Grid
    # ------------------------------------------------------------------
    r = np.arange(0, 1 + dr, dr)
    ang = np.linspace(0, 2*np.pi, int(2*np.pi/dtheta), endpoint=False)

    nr = len(r)
    n_ang = len(ang)

    # Time steps
    t_final = max(times_to_keep)
    nt = int(t_final / dt)

    # Storage
    sampled_solutions = {}

    # Initial condition U(r,ang)
    u = u0(r, ang)

    # ------------------------------------------------------------------
    #  Precompute constant angular second-derivative matrix (periodic)
    #  d^2/dθ^2 operator
    # ------------------------------------------------------------------
    A_ang = np.zeros((n_ang, n_ang))
    for j in range(n_ang):
        A_ang[j, j] = -2
        A_ang[j, (j - 1) % n_ang] = 1
        A_ang[j, (j + 1) % n_ang] = 1
    A_ang = A_ang / (dtheta**2)

    # ------------------------------------------------------------------
    # Main time loop
    # ------------------------------------------------------------------
    for n_step in range(1, nt + 1):
        t = n_step * dt

        # --------------------------
        # Step 1: Radial implicit solve
        # --------------------------
        u_half = u.copy()
        for j_ang in range(n_ang):
            # RHS adds angular term (explicit this half-step)
            rhs = np.zeros(nr)

            for i_r in range(1, nr - 1):
                ri = r[i_r]
                rhs[i_r] = (
                    u[i_r, j_ang]
                    + 0.5*dt*(1/(ri**2)) * (A_ang[j_ang] @ u[i_r, :])
                )

            # Build tridiagonal for radial operator (implicit)
            main = np.zeros(nr - 2)
            lower = np.zeros(nr - 3)
            upper = np.zeros(nr - 3)

            for i in range(1, nr - 1):
                ri = r[i]
                alpha = dt/(2*dr**2)

                main[i-1] = 1 + alpha*(2 + dr/ri)
                if i > 1:
                    lower[i-2] = -alpha*(1 - dr/(2*ri))
                if i < nr - 2:
                    upper[i-1] = -alpha*(1 + dr/(2*ri))

            # Solve tridiagonal (Thomas)
            u_inner = Q3_solve_tridiag(lower, main, upper, rhs[1:-1])
            u_half[1:-1, j_ang] = u_inner

        # Enforce r=0 condition:
        u_half[0, :] = 0

        # --------------------------
        # Step 2: Angular implicit solve
        # --------------------------
        u_new = u_half.copy()
        for i_r in range(1, nr - 1):
            ri = r[i_r]

            # Build angular matrix for this radius
            A = (
                np.eye(n_ang) +
                (dt/(2*ri**2))*A_ang
            )

            # RHS includes radial term from step 1 (explicit)
            rhs = u_half[i_r, :].copy()
            radial = (
                (u_half[i_r+1, :] - 2*u_half[i_r, :] + u_half[i_r-1, :]) / (dr**2)
            )
            rhs += 0.5*dt * radial

            u_new[i_r, :] = np.linalg.solve(A, rhs)

        # r=0 again
        u_new[0, :] = 0

        # boundary at r=1
        u_new[-1, :] = np.sin(4*ang) * np.sin(t)

        u = u_new

        # Save snapshot
        if t in times_to_keep:
            sampled_solutions[t] = u.copy()

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
