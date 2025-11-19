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


def adi_advection_2d(initial_condition, boundary_conditions, a, b, x_domain, y_domain, t_domain, dx, dy, dt):
    """
    Solves the 2D advection equation using the ADI method on a 2D rectangular domain. The boundary conditions are of the form: BC_i(t) = alpha_i*u + beta_i*du

    Parameters:
        initial_condition: 2D array of initial values corresponding to the spatial grid
        boundary_conditions: dictionary with keys 'left', 'right', 'bottom', 'top' and values as functions of time
        a: advection velocity in the x-direction
        b: advection velocity in the y-direction
        x_domain: tuple (x_min, x_max) representing the spatial domain in the x-direction
        y_domain: tuple (y_min, y_max) representing the spatial domain in the y-direction
        t_domain: tuple (t_min, t_max) representing the time domain
        dx: grid size in the x-direction
        dy: grid size in the y-direction
        dt: time step size

    Returns:
        u: 3D array of solution values with shape (nt, nx, ny)
    """
    nx = int((x_domain[1] - x_domain[0]) / dx) + 1
    ny = int((y_domain[1] - y_domain[0]) / dy) + 1
    nt = int((t_domain[1] - t_domain[0]) / dt) + 1

    u = np.zeros((nt, nx, ny))
    u[0, :, :] = initial_condition

    for n in range(0, nt-1):
        # Half step in x-direction
        for j in range(1, ny-1):
            A = np.zeros((nx-2, nx-2))
            B = np.zeros(nx-2)
            for i in range(1, nx-1):
                A[i-1, i-1] = 1 + (a*dt)/(2*dx)
                if i > 1:
                    A[i-1, i-2] = -(a*dt)/(4*dx)
                if i < nx-2:
                    A[i-1, i] = -(a*dt)/(4*dx)
                B[i-1] = u[n, i, j] - (b*dt)/(2*dy) * (u[n, i, j+1] - u[n, i, j-1])
            u_half = np.linalg.solve(A, B)
            for i in range(1, nx-1):
                u[n+0.5, i, j] = u_half[i-1]

        # Half step in y-direction
        for i in range(1, nx-1):
            A = np.zeros((ny-2, ny-2))
            B = np.zeros(ny-2)
            for j in range(1, ny-1):
                A[j-1, j-1] = 1 + (b*dt)/(2*dy)
                if j > 1:
                    A[j-1, j-2] = -(b*dt)/(4*dy)
                if j < ny-2:
                    A[j-1, j] = -(b*dt)/(4*dy)
                B[j-1] = u[n+0.5, i, j]
            u_new = np.linalg.solve(A, B)
            for j in range(1, ny-1):
                u[n+1, i, j] = u_new[j-1]

        # Apply boundary conditions
        for side in boundary_conditions:
            if side == 'left':
                u[n+1, 0, :] = boundary_conditions[side](t_domain[0] + (n+1)*dt)
            elif side == 'right':
                u[n+1, -1, :] = boundary_conditions[side](t_domain[0] + (n+1)*dt)
            elif side == 'bottom':
                u[n+1, :, 0] = boundary_conditions[side](t_domain[0] + (n+1)*dt)
            elif side == 'top':
                u[n+1, :, -1] = boundary_conditions[side](t_domain[0] + (n+1)*dt)

    return u

def adi_advection_3d(initial_condition, boundary_conditions, a, b, c, x_domain, y_domain, z_domain, t_domain, dx, dy, dz, dt):
    """
    Solves the 3D advection equation using the ADI method on a 3D rectangular prism. The boundary conditions are of the form: BC_i(t) = alpha_i*u + beta_i*du

    Parameters:
        initial_condition: 3D array of initial values corresponding to the spatial grid
        boundary_conditions: dictionary with keys 'left', 'right', 'bottom', 'top', 'front', 'back' and values as functions of time
        a: advection velocity in the x-direction
        b: advection velocity in the y-direction
        c: advection velocity in the z-direction
        x_domain: tuple (x_min, x_max) representing the spatial domain in the x-direction
        y_domain: tuple (y_min, y_max) representing the spatial domain in the y-direction
        z_domain: tuple (z_min, z_max) representing the spatial domain in the z-direction
        t_domain: tuple (t_min, t_max) representing the time domain
        dx: grid size in the x-direction
        dy: grid size in the y-direction
        dz: grid size in the z-direction
        dt: time step size
    """

    nx = int((x_domain[1] - x_domain[0]) / dx) + 1
    ny = int((y_domain[1] - y_domain[0]) / dy) + 1
    nz = int((z_domain[1] - z_domain[0]) / dz) + 1
    nt = int((t_domain[1] - t_domain[0]) / dt) + 1

    u = np.zeros((nt, nx, ny, nz))
    u[0, :, :, :] = initial_condition

    for n in range(0, nt-1):
        # Half step in x-direction
        for j in range(1, ny-1):
            for k in range(1, nz-1):
                A = np.zeros((nx-2, nx-2))
                B = np.zeros(nx-2)
                for i in range(1, nx-1):
                    A[i-1, i-1] = 1 + (a*dt)/(2*dx)
                    if i > 1:
                        A[i-1, i-2] = -(a*dt)/(4*dx)
                    if i < nx-2:
                        A[i-1, i] = -(a*dt)/(4*dx)
                    B[i-1] = u[n, i, j, k] - (b*dt)/(2*dy) * (u[n, i, j+1, k] - u[n, i, j-1, k]) - (c*dt)/(2*dz) * (u[n, i, j, k+1] - u[n, i, j, k-1])
                u_half = np.linalg.solve(A, B)
                for i in range(1, nx-1):
                    u[n+0.5, i, j, k] = u_half[i-1]

        # Half step in y-direction
        for i in range(1, nx-1):
            for k in range(1, nz-1):
                A = np.zeros((ny-2, ny-2))
                B = np.zeros(ny-2)
                for j in range(1, ny-1):
                    A[j-1, j-1] = 1 + (b*dt)/(2*dy)
                    if j > 1:
                        A[j-1, j-2] = -(b*dt)/(4*dy)
                    if j < ny-2:
                        A[j-1, j] = -(b*dt)/(4*dy)
                    B[j-1] = u[n+0.5, i, j, k] - (c*dt)/(2*dz) * (u[n+0.5, i, j, k+1] - u[n+0.5, i, j, k-1])
                u_new = np.linalg.solve(A, B)
                for j in range(1, ny-1):
                    u[n+1, i, j, k] = u_new[j-1]
        # Half step in z-direction
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                A = np.zeros((nz-2, nz-2))
                B = np.zeros(nz-2)
                for k in range(1, nz-1):
                    A[k-1, k-1] = 1 + (c*dt)/(2*dz)
                    if k > 1:
                        A[k-1, k-2] = -(c*dt)/(4*dz)
                    if k < nz-2:
                        A[k-1, k] = -(c*dt)/(4*dz)
                    B[k-1] = u[n+1, i, j, k]
                u_new = np.linalg.solve(A, B)
                for k in range(1, nz-1):
                    u[n+1, i, j, k] = u_new[k-1]
        # Apply boundary conditions
        for side in boundary_conditions:
            if side == 'left':
                u[n+1, 0, :, :] = boundary_conditions[side](t_domain[0] + (n+1)*dt)
            elif side == 'right':
                u[n+1, -1, :, :] = boundary_conditions[side](t_domain[0] + (n+1)*dt)
            elif side == 'bottom':
                u[n+1, :, 0, :] = boundary_conditions[side](t_domain[0] + (n+1)*dt)
            elif side == 'top':
                u[n+1, :, -1, :] = boundary_conditions[side](t_domain[0] + (n+1)*dt)
            elif side == 'front':
                u[n+1, :, :, 0] = boundary_conditions[side](t_domain[0] + (n+1)*dt)
            elif side == 'back':
                u[n+1, :, :, -1] = boundary_conditions[side](t_domain[0] + (n+1)*dt)

    return u

def plot_solution_2d(u, x_domain, y_domain, t_index, dx, dy):
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
    nx = u.shape[1]
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
    Animates the 2D solution over time.

    Parameters:
        u: 3D array of solution values with shape (nt, nx, ny)
        x_domain: tuple (x_min, x_max) representing the spatial domain in the x-direction
        y_domain: tuple (y_min, y_max) representing the spatial domain in the y-direction
        dx: grid size in the x-direction
        dy: grid size in the y-direction
        interval: delay between frames in milliseconds
    """

    nx = u.shape[1]
    ny = u.shape[2]
    nt = u.shape[0]
    x = np.linspace(x_domain[0], x_domain[1], nx)
    y = np.linspace(y_domain[0], y_domain[1], ny)
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots()
    cont = ax.contourf(X, Y, u[0, :, :].T, levels=50, cmap='viridis')
    plt.colorbar(cont)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    def update(frame):
        ax.clear()
        cont = ax.contourf(X, Y, u[frame, :, :].T, levels=50, cmap='viridis')
        ax.set_title(f'Solution at time index {frame}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        return cont.collections

    ani = FuncAnimation(fig, update, frames=nt, interval=interval)
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

        u_soln = adi_advection_2d(initial_condition, boundary_conditions, a, b, x_domain, y_domain, t_domain, dx, dy, dt)
        plot_solution_2d(u_soln, x_domain, y_domain, t_index = -1, dx = dx, dy = dy)
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
        return adi_advection_3d(initial_condition, boundary_conditions, a, b, c, x_domain, y_domain, z_domain, t_domain, dx, dy, dz, dt)
    
if __name__ == "__main__":
    main(dimension=2)