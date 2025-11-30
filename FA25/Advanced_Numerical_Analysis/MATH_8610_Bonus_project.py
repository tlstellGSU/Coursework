"""
MATH 8610 - Bonus project

Create a function that solves the general advection equation in 2D and 3D using ADI method
The function should take as input the initial condition, boundary conditions, advection velocities, spatial domain, time domain, and size of grid points in each direction.

Advection equation: u_t + a*u_x + b*u_y (+ c*u_z) = 0

BCs for the ith side: BC_i(t) = alpha_i*u + beta_i*du

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import solve_banded

def thomas_solve(a, b, c, d):
    """
    Solve tridiagonal system with lower diag a[1..n-1], main diag b[0..n-1], upper diag c[0..n-2]
    a, b, c, d are 1D numpy arrays of length n (a[0] unused, c[-1] unused).
    Returns solution x (length n).

    pulled from: https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    """
    n = len(b)
    cp = np.empty(n-1, dtype=float)
    dp = np.empty(n, dtype=float)
    x = np.empty(n, dtype=float)

    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n-1):
        denom = b[i] - a[i]*cp[i-1]
        cp[i] = c[i] / denom
        dp[i] = (d[i] - a[i]*dp[i-1]) / denom
    denom = b[n-1] - a[n-1]*cp[n-2]
    dp[n-1] = (d[n-1] - a[n-1]*dp[n-2]) / denom

    x[n-1] = dp[n-1]
    for i in range(n-2, -1, -1):
        x[i] = dp[i] - cp[i]*x[i+1]
    return x

def make_default_bc():
    # returns (alpha, beta, gfunc) for each side
    zero = lambda t: 0.0
    return {
        'left':   (1.0, 0.0, zero),
        'right':  (1.0, 0.0, zero),
        'bottom': (1.0, 0.0, zero),
        'top':    (1.0, 0.0, zero)
    }

def adi_advection_2d(initial_condition, boundary_conditions, a, b,
                     x_domain, y_domain, t_domain, dx, dy, dt):
    """
    ADI solver for u_t + a u_x + b u_y = 0 with Robin BCs.
    boundary_conditions: dict with keys 'left','right','bottom','top'
    each value is (alpha, beta, gfunc) where gfunc(t) -> scalar (can be array along side).
    Returns: u(nt,nx,ny), xs, ys, times
    """
    if boundary_conditions is None:
        boundary_conditions = make_default_bc()

    x0, x1 = x_domain
    y0, y1 = y_domain
    t0, t1 = t_domain
    nx = int(round((x1 - x0) / dx)) + 1
    ny = int(round((y1 - y0) / dy)) + 1
    nt = int(round((t1 - t0) / dt)) + 1

    # validate initial_condition shape
    if initial_condition.shape != (nx, ny):
        raise ValueError(f"initial_condition.shape {initial_condition.shape} does not match computed (nx,ny)=({nx},{ny})")

    xs = x0 + np.arange(nx) * dx
    ys = y0 + np.arange(ny) * dy
    times = t0 + np.arange(nt) * dt

    u = np.zeros((nt, nx, ny))
    u[0, :, :] = initial_condition.copy()

    # useful scalars
    adx = a * dt / (4.0 * dx)   # used in RHS for cross terms and half coefficient in tri-diag
    bdy = b * dt / (4.0 * dy)
    # overall CFL-like indicator
    cfl = abs(a) * dt / dx + abs(b) * dt / dy
    if cfl > 1.0:
        print(f"Warning: CFL-like number {cfl:.3f} > 1 may cause large errors. Consider reducing dt or increasing dx/dy.")

    # preallocate temporaries
    u_star = np.zeros((nx, ny))

    # Pre-build static template tridiagonal entries (interior pattern)
    lower_template = np.zeros(nx, dtype=float)
    diag_template  = np.ones(nx, dtype=float)
    upper_template = np.zeros(nx, dtype=float)
    for i in range(1, nx-1):
        lower_template[i] = -adx
        diag_template[i]  = 1.0
        upper_template[i] = +adx

    # Y-sweep templates
    lowerY_template = np.zeros(ny, dtype=float)
    diagY_template  = np.ones(ny, dtype=float)
    upperY_template = np.zeros(ny, dtype=float)
    for j in range(1, ny-1):
        lowerY_template[j] = -bdy
        diagY_template[j]  = 1.0
        upperY_template[j] = +bdy

    for n in range(nt - 1):
        t_np1 = times[n+1]

        # ----- X-sweep: solve columns for each interior j -----
        for j in range(1, ny-1):
            lower = lower_template.copy()
            diag  = diag_template.copy()
            upper = upper_template.copy()
            RHS   = np.zeros(nx, dtype=float)

            # interior RHS
            for i in range(1, nx-1):
                RHS[i] = u[n, i, j] - (b * dt) / (4.0 * dy) * (u[n, i, j+1] - u[n, i, j-1])

            # left BC row (i=0): alpha*u0 + beta*(u1 - u0)/dx = gL
            alpha_L, beta_L, gL = boundary_conditions['left']
            gLval = gL(t_np1)
            if beta_L == 0:
                # Dirichlet-like: u0 = g/alpha
                diag[0] = 1.0
                upper[0] = 0.0
                RHS[0] = gLval / alpha_L
            else:
                # (alpha - beta/dx) * u0 + (beta/dx) * u1 = g
                diag[0]  = alpha_L - (beta_L / dx)
                upper[0] = (beta_L / dx)
                RHS[0] = gLval

            # right BC row (i=nx-1): alpha*uN + beta*(uN - uN-1)/dx = gR
            alpha_R, beta_R, gR = boundary_conditions['right']
            gRval = gR(t_np1)
            if beta_R == 0:
                diag[-1] = 1.0
                lower[-1] = 0.0
                RHS[-1] = gRval / alpha_R
            else:
                # (-beta/dx)*u_{N-2} + (alpha + beta/dx)*u_{N-1} = g
                lower[-1] = - (beta_R / dx)
                diag[-1]  = alpha_R + (beta_R / dx)
                RHS[-1] = gRval

            # solve column
            ucol = thomas_solve(lower, diag, upper, RHS)
            u_star[:, j] = ucol

        # Fill y-boundaries in u_star using Robin relations (use already-computed j=1 and j=ny-2)
        alpha_B, beta_B, gB = boundary_conditions['bottom']
        alpha_T, beta_T, gT = boundary_conditions['top']
        for i in range(nx):
            # bottom j=0
            gBval = gB(t_np1)
            if beta_B == 0:
                u_star[i, 0] = gBval / alpha_B
            else:
                # beta*(u1 - u0)/dy + alpha*u0 = g  => u0 = (beta/dy * u1 - g) / (beta/dy + alpha)
                u_star[i, 0] = ((beta_B / dy) * u_star[i, 1] - gBval) / ((beta_B / dy) + alpha_B)
            # top j=ny-1
            gTval = gT(t_np1)
            if beta_T == 0:
                u_star[i, -1] = gTval / alpha_T
            else:
                u_star[i, -1] = ((beta_T / dy) * u_star[i, -2] + gTval) / (alpha_T + (beta_T / dy))

        # ----- Y-sweep: solve rows for each interior i -----
        unp1 = np.zeros((nx, ny))
        for i in range(1, nx-1):
            lower = lowerY_template.copy()
            diag  = diagY_template.copy()
            upper = upperY_template.copy()
            RHS   = np.zeros(ny, dtype=float)

            for j in range(1, ny-1):
                RHS[j] = u_star[i, j] - (a * dt) / (4.0 * dx) * (u_star[i+1, j] - u_star[i-1, j])

            # bottom BC j=0
            gBval = gB(t_np1)
            if beta_B == 0:
                diag[0] = 1.0
                upper[0] = 0.0
                RHS[0] = gBval / alpha_B
            else:
                diag[0]  = alpha_B - (beta_B / dy)
                upper[0] = (beta_B / dy)
                RHS[0] = gBval

            # top BC j=ny-1
            gTval = gT(t_np1)
            if beta_T == 0:
                diag[-1] = 1.0
                lower[-1] = 0.0
                RHS[-1] = gTval / alpha_T
            else:
                lower[-1] = - (beta_T / dy)
                diag[-1]  = alpha_T + (beta_T / dy)
                RHS[-1] = gTval

            urow = thomas_solve(lower, diag, upper, RHS)
            unp1[i, :] = urow

        # Fill x-boundaries in unp1 using Robin relations and already-computed neighbors
        for j in range(ny):
            # left i=0
            gLval = boundary_conditions['left'][2](t_np1)
            alpha_L, beta_L = boundary_conditions['left'][0], boundary_conditions['left'][1]
            if beta_L == 0:
                unp1[0, j] = gLval / alpha_L
            else:
                unp1[0, j] = ((beta_L / dx) * unp1[1, j] - gLval) / ((beta_L / dx) + alpha_L)
            # right i=nx-1
            gRval = boundary_conditions['right'][2](t_np1)
            alpha_R, beta_R = boundary_conditions['right'][0], boundary_conditions['right'][1]
            if beta_R == 0:
                unp1[-1, j] = gRval / alpha_R
            else:
                unp1[-1, j] = ((beta_R / dx) * unp1[-2, j] + gRval) / (alpha_R + (beta_R / dx))

        u[n+1, :, :] = unp1

    return u, xs, ys, times

def Dx_periodic(n, dx):
    e = np.ones(n)
    A = np.zeros((n, n))
    # central skew derivative
    for i in range(n):
        A[i, (i+1) % n] =  1/(2*dx)
        A[i, (i-1) % n] = -1/(2*dx)
    return A

def adi_advection_3d(initial_condition, boundary_conditions,
                               a, b, c,
                               x_domain, y_domain, z_domain, t_domain,
                               dx, dy, dz, dt):
    """
    Optimized 3D ADI solver for u_t + a*u_x + b*u_y + c*u_z = 0
    using skew-symmetric derivatives and tridiagonal solvers.
    
    boundary_conditions: dict with Dirichlet functions for 'left','right','bottom','top','front','back'
    """
    nx = int((x_domain[1] - x_domain[0]) / dx) + 1
    ny = int((y_domain[1] - y_domain[0]) / dy) + 1
    nz = int((z_domain[1] - z_domain[0]) / dz) + 1
    nt = int((t_domain[1] - t_domain[0]) / dt) + 1

    u = np.zeros((nt, nx, ny, nz))
    u[0] = initial_condition

    # Precompute tridiagonal diagonals for x, y, z sweeps
    ax = a * dt / (4*dx)
    bx = 1.0
    ay = b * dt / (4*dy)
    by = 1.0
    az = c * dt / (4*dz)
    bz = 1.0

    xs = x_domain[0] + np.arange(nx) * dx
    ys = y_domain[0] + np.arange(ny) * dy
    zs = z_domain[0] + np.arange(nz) * dz

    # Tridiagonal banded matrices (shape (3, N))
    def make_banded(N, coef):
        ab = np.zeros((3, N))
        ab[0,1:] = coef  # upper
        ab[1,:] = 1.0    # main
        ab[2,:-1] = -coef  # lower
        return ab

    abx = make_banded(nx-2, ax)
    aby = make_banded(ny-2, ay)
    abz = make_banded(nz-2, az)

    # Temporary arrays (reuse each time step)
    u_half = np.zeros((nx, ny, nz))
    u_half2 = np.zeros((nx, ny, nz))
    u_new = np.zeros((nx, ny, nz))

    times = t_domain[0] + dt * np.arange(nt)

    for n in range(nt-1):
        t_now = times[n+1]

        # ---------- X SWEEP ----------
        for j in range(1, ny-1):
            for k in range(1, nz-1):
                B = u[n, 1:-1, j, k] - \
                    (b*dt)/(2*dy)*(u[n, 1:-1, j+1, k] - u[n, 1:-1, j-1, k]) - \
                    (c*dt)/(2*dz)*(u[n, 1:-1, j, k+1] - u[n, 1:-1, j, k-1])
                u_half[1:-1, j, k] = solve_banded((1,1), abx, B)

        # Boundaries (Dirichlet)
        u_half[0,:,:] = boundary_conditions['left'][2](t_now)
        u_half[-1,:,:] = boundary_conditions['right'][2](t_now)
        u_half[:,0,:] = u[n,:,0,:]
        u_half[:,-1,:] = u[n,:,-1,:]
        u_half[:,:,0] = u[n,:,:,0]
        u_half[:,:,-1] = u[n,:,:,-1]

        # ---------- Y SWEEP ----------
        for i in range(1, nx-1):
            for k in range(1, nz-1):
                B = u_half[i,1:-1,k] - \
                    (c*dt)/(2*dz)*(u_half[i,1:-1,k+1] - u_half[i,1:-1,k-1]) - \
                    (a*dt)/(2*dx)*(u_half[i+1,1:-1,k] - u_half[i-1,1:-1,k])
                u_half2[i,1:-1,k] = solve_banded((1,1), aby, B)

        # Boundaries
        u_half2[:,0,:] = boundary_conditions['bottom'][2](t_now)
        u_half2[:,-1,:] = boundary_conditions['top'][2](t_now)
        u_half2[0,:,:] = u_half[0,:,:]
        u_half2[-1,:,:] = u_half[-1,:,:]
        u_half2[:,:,0] = u_half[:,:,0]
        u_half2[:,:,-1] = u_half[:,:,-1]

        # ---------- Z SWEEP ----------
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                B = u_half2[i,j,1:-1] - \
                    (a*dt)/(2*dx)*(u_half2[i+1,j,1:-1] - u_half2[i-1,j,1:-1]) - \
                    (b*dt)/(2*dy)*(u_half2[i,j+1,1:-1] - u_half2[i,j-1,1:-1])
                u_new[i,j,1:-1] = solve_banded((1,1), abz, B)

        # Boundaries
        u_new[0,:,:] = boundary_conditions['left'][2](t_now)
        u_new[-1,:,:] = boundary_conditions['right'][2](t_now)
        u_new[:,0,:] = boundary_conditions['bottom'][2](t_now)
        u_new[:,-1,:] = boundary_conditions['top'][2](t_now)
        u_new[:,:,0] = boundary_conditions['front'][2](t_now)
        u_new[:,:,-1] = boundary_conditions['back'][2](t_now)

        # Save new step
        u[n+1] = u_new

    return u, xs, ys, zs, times

def plot_solution_2d(u, x_domain, y_domain, t_index, dx, dy, nx=None, ny=None):
    """
    Plots the 2D solution at a given time index.

    Parameters:
        u: 3D array of solution values with shape (nt, nx, ny)
        x_domain: tuple (x_min, x_max) representing the spatial domain in the x-direction
        y_domain: tuple (y_min, y_max) representing the spatial domain in the y-direction
        t_index: integer index of the time step to plot
        dx: grid size in the x-direction
        dy: grid size in the y-direction
    """
    if nx is None:
        nx = u.shape[1]
    if ny is None:
        ny = u.shape[2]

    x = np.linspace(x_domain[0], x_domain[1], nx)
    y = np.linspace(y_domain[0], y_domain[1], ny)
    X, Y = np.meshgrid(x, y)

    plt.figure()
    plt.contourf(X, Y, u[t_index, :, :].T, levels=50, cmap='viridis')
    plt.colorbar()
    plt.title(f'Solution at time index {t_index}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def animate_solution_2d(u, x_domain, y_domain, dx, dy, interval=100):
    """
    Efficient animation of a 2D solution using pcolormesh.
    
    u: 3D array of shape (nt, nx, ny)
    """
    nx = u.shape[1]
    ny = u.shape[2]
    nt = u.shape[0]
    x = np.linspace(x_domain[0], x_domain[1], nx)
    y = np.linspace(y_domain[0], y_domain[1], ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    fig, ax = plt.subplots()
    mesh = ax.pcolormesh(X, Y, u[0, :-1, :-1], shading='auto', cmap='viridis')    
    cbar = plt.colorbar(mesh, ax=ax)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    def update(frame):
        mesh.set_array(u[frame, :-1, :-1].ravel())
        ax.set_title(f'Time index {frame}')
        return mesh,

    ani = FuncAnimation(fig, update, frames=nt, interval=interval, blit=True)
    plt.show()

def animate_solution_3d(u, x_domain, y_domain, z_slice_index=0, interval=100):
    """
    Animate a 2D slice of 3D solution over time.
    
    Parameters:
        u: 4D array (nt, nx, ny, nz)
        x_domain, y_domain: spatial domain tuples
        z_slice_index: integer, index of the z-slice to animate
        interval: delay between frames in ms
    """
    nt, nx, ny, nz = u.shape
    x = np.linspace(x_domain[0], x_domain[1], nx)
    y = np.linspace(y_domain[0], y_domain[1], ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Initial surface
    surf = ax.plot_surface(X, Y, u[0, :, :, z_slice_index], cmap='viridis', vmin=u.min(), vmax=u.max())

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u')
    ax.set_zlim(u.min(), u.max())

    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

    def update(frame):
        ax.clear()
        surf = ax.plot_surface(X, Y, u[frame, :, :, z_slice_index], cmap='viridis', vmin=u.min(), vmax=u.max())
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('u')
        ax.set_zlim(u.min(), u.max())
        ax.set_title(f'Time index {frame}')
        return surf,

    ani = FuncAnimation(fig, update, frames=nt, interval=interval, blit=False)
    plt.show()


def main(initial_condition = None, boundary_conditions = None, a = 1.0, b = 1.0, c = 1.0, x_domain = (0, 1), y_domain = (0, 1), z_domain = (0, 1), t_domain = (0, 1), dx = 0.01, dy = 0.01, dz = 0.01, dt = 0.01, dimension = 2):
    """
    Main function to solve the advection equation using the ADI method. Calls the appropriate 2D or 3D solver based on the dimension parameter.
    If initial_condition or boundary_conditions are not provided, default values will be used.
    The default initial condition is a sine wave in all spatial dimensions.
    The default boundary conditions are homogeneous Dirichlet conditions on all sides.
    
    Parameters:
        initial_condition: array of initial values corresponding to the spatial grid
        boundary_conditions: dictionary with keys corresponding to the boundaries and values as functions of time
        a: advection velocity in the x-direction
        b: advection velocity in the y-direction
        c: advection velocity in the z-direction (only for 3D)
        x_domain: tuple (x_min, x_max) representing the spatial domain in the x-direction
        y_domain: tuple (y_min, y_max) representing the spatial domain in the y-direction
        z_domain: tuple (z_min, z_max) representing the spatial domain in the z-direction (only for 3D)
        t_domain: tuple (t_min, t_max) representing the time domain
        dx: grid size in the x-direction
        dy: grid size in the y-direction
        dz: grid size in the z-direction (only for 3D)
        dt: time step size
        dimension: integer (2 or 3) indicating whether to solve in 2D or 3D

    Returns:
        u: array of solution values (either 3D or 4D depending on dimension)
    """
    if dimension == 2:
        if initial_condition is None:
            nx = int((x_domain[1] - x_domain[0]) / dx) + 1
            ny = int((y_domain[1] - y_domain[0]) / dy) + 1
            initial_condition = np.zeros((nx, ny))
            for i in range(nx):
                for j in range(ny):
                    x = x_domain[0] + i*dx
                    y = y_domain[0] + j*dy
                    initial_condition[i, j] = np.sin(np.pi*x) * np.sin(np.pi*y)
        if boundary_conditions is None:
            boundary_conditions = {
                'left': lambda t: 0,
                'right': lambda t: 0,
                'bottom': lambda t: 0,
                'top': lambda t: 0
            }

        u_soln, xs, ys, times = adi_advection_2d(initial_condition, boundary_conditions, a, b, x_domain, y_domain, t_domain, dx, dy, dt)
        #plot_solution_2d(u_soln, x_domain, y_domain, t_index = -1, dx = dx, dy = dy, nx = initial_condition.shape[0], ny = initial_condition.shape[1])
        animate_solution_2d(u_soln, x_domain, y_domain, dx = dx, dy = dy, interval=100)
        return u_soln
    elif dimension == 3:
        if initial_condition is None:
            nx = int((x_domain[1] - x_domain[0]) / dx) + 1
            ny = int((y_domain[1] - y_domain[0]) / dy) + 1
            nz = int((z_domain[1] - z_domain[0]) / dz) + 1
            initial_condition = np.zeros((nx, ny, nz))
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        x = x_domain[0] + i*dx
                        y = y_domain[0] + j*dy
                        z = z_domain[0] + k*dz
                        initial_condition[i, j, k] = np.sin(np.pi*x) * np.sin(np.pi*y) * np.sin(np.pi*z)
        if boundary_conditions is None:
            boundary_conditions = {
                'left': lambda t: 0,
                'right': lambda t: 0,
                'bottom': lambda t: 0,
                'top': lambda t: 0,
                'front': lambda t: 0,
                'back': lambda t: 0
            }
        u_soln, xs, ys, zs, times = adi_advection_3d(initial_condition, boundary_conditions, a, b, c, x_domain, y_domain, z_domain, t_domain, dx, dy, dz, dt)
        animate_solution_3d(u_soln, x_domain, y_domain, z_slice_index = int(len(zs)/2), interval=100)
        return u_soln, xs, ys, zs, times
    
if __name__ == "__main__":

    x_domain = (0, 1)
    y_domain = (0, 1)
    z_domain = (0, 1)
    t_domain = (0, 0.5)

    dx = 0.01
    dy = 0.01
    dz = 0.01
    dt = 0.005

    a = -1.0
    b = 0.0
    c = 1.0

    n_x = int((x_domain[1] - x_domain[0]) / dx) + 1
    n_y = int((y_domain[1] - y_domain[0]) / dy) + 1
    n_z = int((z_domain[1] - z_domain[0]) / dz) + 1

    n_t = int((t_domain[1] - t_domain[0]) / dt) + 1

    # initial condition

    u_0_2D = np.zeros((n_x, n_y))
    for i in range(n_x):
        for j in range(n_y):
            x = x_domain[0] + i*dx
            y = y_domain[0] + j*dy
            u_0_2D[i, j] = np.sin(np.pi*x/0.1) * np.sin(np.pi*y/0.1)

    u_0_3D = np.zeros((n_x, n_y, n_z))
    for i in range(n_x):
        for j in range(n_y):
            for k in range(n_z):
                x = x_domain[0] + i*dx
                y = y_domain[0] + j*dy
                z = z_domain[0] + k*dz
                u_0_3D[i, j, k] = np.sin(np.pi*x) * np.sin(np.pi*y) * np.sin(np.pi*z)
    
    # boundary conditions
    boundary_conditions_2D = {
        # u + u_x = f(t)
        'left':   (0.0, -1.0, lambda t: 0.0),
        'right':  (1.0, 1.0, lambda t: 1.0),
        'bottom': (1.0, 1.0, lambda t: 1.0),
        'top':    (1.0, 1.0, lambda t: 1.0)
    }

    boundary_conditions_3D = {
        'left':   (1.0, 0.0, lambda t: 0.0),
        'right':  (1.0, 0.0, lambda t: 0.0),
        'bottom': (1.0, 0.0, lambda t: 0.0),
        'top':    (1.0, 0.0, lambda t: 0.0),
        'front':  (1.0, 0.0, lambda t: 0.0),
        'back':   (1.0, 0.0, lambda t: 0.0)
    }

    main(initial_condition = u_0_2D, boundary_conditions = boundary_conditions_2D, a = a, b = b, x_domain = x_domain, y_domain = y_domain, t_domain = t_domain, dx = dx, dy = dy, dt = dt, dimension = 2)
    #main(initial_condition = u_0_3D, boundary_conditions = boundary_conditions_3D, a = a, b = b, c = c, x_domain = x_domain, y_domain = y_domain, z_domain = z_domain, t_domain = t_domain, dx = dx, dy = dy, dz = dz, dt = dt, dimension = 3)