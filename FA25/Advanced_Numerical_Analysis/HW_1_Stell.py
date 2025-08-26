# Code for HW 1 for FA25 Advanced Numerical Analysis

# All code is original and written by me, Tommy Stell, for Advanced Numerical Analysis
 
import numpy as np
import matplotlib.pyplot as plt

# Problem 1

# I: U_t = v U_xx, 0 < x < 1, t > 0
#    U(0,t) = 0, U(1,t) = 0, t >= 0
#    U(x,0) = sin(2*pi*x), 0 <= x <= 1

# II: U_t = U_xx, 0 < x < 1, t > 0
#     U(x,0) = cos(pi*x/2), 0 <= x <= 1
#     U_x(0,t) = U(1,t) = 0, t >= 0

# 1.a)

exact_sol_1_I = lambda x, t, v: np.sin(2*np.pi*x)*np.exp(-v*(2*np.pi)**2*t)
exact_sol_1_II = lambda x, t: np.cos(np.pi*x/2)*np.exp(-(np.pi/2)**2*t)

# 1.c) 

v_1c = 1/6
del_x_1c = 1/10
del_t_1c = 0.01

sample_t_1c = [0.01, 0.1, 1, 10]

x_1c = np.arange(0, 1+del_x_1c, del_x_1c)
num_x_1c = len(x_1c)
num_t_1c = int(10/del_t_1c)

u_initial_1c = np.sin(2*np.pi*x_1c)

sols_num = {}
sols_exact = {}

u = u_initial_1c.copy()

for i in range(1, num_t_1c+1):
    u_temp = u.copy()
    for j in range(1, num_x_1c-1):
        u[j] = u_temp[j] + v_1c*del_t_1c/del_x_1c**2 * (u_temp[j+1] - 2*u_temp[j] + u_temp[j-1])

    t_now = i*del_t_1c
    if t_now in sample_t_1c:
        sols_num[t_now] = u.copy()
        sols_exact[t_now] = exact_sol_1_I(x_1c, t_now, v_1c)

for t in sample_t_1c:
    plt.plot(x_1c, sols_num[t], label=f'Numerical t={t}')
    plt.plot(x_1c, sols_exact[t], '--', label=f'Exact t={t}')

plt.title('Problem 1.c) Numerical vs Exact Solutions')
plt.xlabel('x')
plt.ylabel('U(x,t)')
plt.legend()
plt.grid()
plt.savefig('HW1_Problem1c_Solutions.png')

# by the stability theory, we need v*del_t/del_x^2 <= 1/2
stability_ratio_1c = v_1c * del_t_1c / del_x_1c**2
print(f'Stability ratio for Problem 1.c): {stability_ratio_1c}')
print(f'The maximum allowable del_t for stability is: {0.5 * del_x_1c**2 / v_1c}')


# 1.d)

v_1d = 1/6
del_x_1d = 1/10
del_t_1d = 0.01

sample_t_1d = [0.01, 0.1, 1, 10]

x_1d = np.arange(0, 1+del_x_1d, del_x_1d)
num_x_1d = len(x_1d)
num_t_1d = int(10/del_t_1d)

u_initial_1d = np.sin(2*np.pi*x_1d)

u = u_initial_1d.copy()
u_old = u_initial_1d.copy()

sols_num_1d = {}
sols_exact_1d = {}

for i in range(1, num_t_1d+1):
    u_temp = u.copy()
    if i == 1:
        for j in range(1, num_x_1d-1):
            u[j] = 0+ v_1d*del_t_1d/del_x_1d**2 * (u_temp[j+1] - 2*u_temp[j] + u_temp[j-1])
    else:
        for j in range(1, num_x_1d-1):
            u[j] = u_old[j] + 2*v_1d*del_t_1d/del_x_1d**2 * (u_temp[j+1] - 2*u_temp[j] + u_temp[j-1])
    u_old = u_temp.copy()

    t_now = i*del_t_1d
    if t_now in sample_t_1d:
        sols_num_1d[t_now] = u.copy()
        sols_exact_1d[t_now] = exact_sol_1_I(x_1d, t_now, v_1d)

for t in sample_t_1d:
    plt.plot(x_1d, sols_num_1d[t], label=f'Numerical t={t}')
    plt.plot(x_1d, sols_exact_1d[t], '--', label=f'Exact t={t}')

plt.title('Problem 1.d) Numerical vs Exact Solutions')
plt.xlabel('x')
plt.ylabel('U(x,t)')
plt.legend()
plt.grid()
plt.savefig('HW1_Problem1d_Solutions.png')

# 1.e)

# problem 2

# U_t = U_xx, 0 < x < 1, t > 0
# U(x,0) = cos(pi*x/2) 0 <= x <= 1
# U_x(0,t) = sin(2*pi*t), U_x(1,t) = 2*pi, t >= 0

exact_sol_2 = lambda x, t: np.cos(np.pi*x/2)*np.exp(-(np.pi/2)**2*t) + np.sin(2*np.pi*t)*x