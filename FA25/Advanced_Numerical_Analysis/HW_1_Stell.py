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

exact_sol_1_I = lambda x, t, v: np.sin(2 * np.pi * x) * np.exp(
    -v * (2 * np.pi) ** 2 * t
)
exact_sol_1_II = lambda x, t: np.cos(np.pi * x / 2) * np.exp(-((np.pi / 2) ** 2) * t)

# 1.c)

v_1c = 1 / 6
del_x_1c = 1 / 10
del_t_1c = 0.01

sample_t_1c = [0.01, 0.1, 1, 10]

x_1c = np.arange(0, 1 + del_x_1c, del_x_1c)
num_x_1c = len(x_1c)
num_t_1c = int(10 / del_t_1c)

u_initial_1c = np.sin(2 * np.pi * x_1c)

sols_num = {}
sols_exact = {}

u = u_initial_1c.copy()

for i in range(1, num_t_1c + 1):
    u_temp = u.copy()
    for j in range(1, num_x_1c - 1):
        u[j] = u_temp[j] + v_1c * del_t_1c / del_x_1c**2 * (
            u_temp[j + 1] - 2 * u_temp[j] + u_temp[j - 1]
        )

    t_now = i * del_t_1c
    if t_now in sample_t_1c:
        sols_num[t_now] = u.copy()
        sols_exact[t_now] = exact_sol_1_I(x_1c, t_now, v_1c)

plt.figure()
for t in sample_t_1c:
    plt.plot(x_1c, sols_num[t], label=f"Numerical t={t}")
    plt.plot(x_1c, sols_exact[t], "--", label=f"Exact t={t}")

plt.title("Problem 1.c) Numerical vs Exact Solutions")
plt.xlabel("x")
plt.ylabel("U(x,t)")
plt.legend()
plt.grid()
plt.savefig("HW1_Problem1c_Solutions.png")

# by the stability theory, we need v*del_t/del_x^2 <= 1/2
stability_ratio_1c = v_1c * del_t_1c / del_x_1c**2
print(f"Stability ratio for Problem 1.c): {stability_ratio_1c}")
print(f"The maximum allowable del_t for stability is: {0.5 * del_x_1c**2 / v_1c}")


# 1.d)

v_1d = 1 / 6
del_x_1d = 0.1
del_t_1d = 0.01
# del_t_1d = del_t_1d**2/(6*v_1d)

sample_t_1d = [0.01, 0.1, 0.5]

x_1d = np.arange(0, 1 + del_x_1d, del_x_1d)
num_x_1d = len(x_1d)
num_t_1d = int(1 / del_t_1d)

u_initial_1d = np.sin(2 * np.pi * x_1d)

u = u_initial_1d.copy()
u_old = u_initial_1d.copy()

sols_num_1d = {}
sols_exact_1d = {}

for i in range(1, num_t_1d + 1):
    u_temp = u.copy()
    if i == 1:
        for j in range(1, num_x_1d - 1):
            u[j] = u_temp[j] + v_1d * del_t_1d / del_x_1d**2 * (
                u_temp[j + 1] - 2 * u_temp[j] + u_temp[j - 1]
            )
    else:
        for j in range(1, num_x_1d - 1):
            u[j] = u_old[j] + 2 * v_1d * del_t_1d / del_x_1d**2 * (
                u_temp[j + 1] - 2 * u_temp[j] + u_temp[j - 1]
            )
    u_old = u_temp.copy()

    t_now = i * del_t_1d
    if t_now in sample_t_1d:
        sols_num_1d[t_now] = u.copy()
        sols_exact_1d[t_now] = exact_sol_1_I(x_1d, t_now, v_1d)

plt.figure()
for t in sample_t_1d:
    plt.plot(x_1d, sols_num_1d[t], label=f"Numerical t={t}")
    plt.plot(x_1d, sols_exact_1d[t], "--", label=f"Exact t={t}")

plt.title("Problem 1.d) Numerical vs Exact Solutions")
plt.xlabel("x")
plt.ylabel("U(x,t)")
plt.legend()
plt.grid()
plt.savefig("HW1_Problem1d_Solutions.png")

# 1.e)

del_x_1e_a = 0.1
del_t_1e_a = 0.004

del_x_1e_b = 0.05
del_t_1e_b = 0.001

sample_t_1e = [1]

x_1e_a = np.arange(0, 1 + del_x_1e_a, del_x_1e_a)
x_1e_b = np.arange(0, 1 + del_x_1e_b, del_x_1e_b)

num_x_1e_a = len(x_1e_a)
num_x_1e_b = len(x_1e_b)
num_t_1e_a = int(1 / del_t_1e_a)
num_t_1e_b = int(1 / del_t_1e_b)

u_initial_1e_a = np.cos(np.pi * x_1e_a / 2)
u_initial_1e_b = np.cos(np.pi * x_1e_b / 2)

u_a = u_initial_1e_a.copy()
u_b = u_initial_1e_b.copy()

sols_exact_1e = {}
sols_num_1e_a_first = {}
sols_num_1e_b_first = {}
sols_num_1e_a_second = {}
sols_num_1e_b_second = {}

# first order approximation of NBCs
for i in range(1, num_t_1e_a + 1):
    u_temp = u_a.copy()
    for j in range(1, num_x_1e_a - 1):
        u_a[j] = u_temp[j] + del_t_1e_a / del_x_1e_a**2 * (
            u_temp[j + 1] - 2 * u_temp[j] + u_temp[j - 1]
        )
    # Neumann BCs
    u_a[0] = u_a[1]  # U_x(0,t) = 0
    u_a[-1] = 0  # U(1,t) = 0
    t_now = i * del_t_1e_a
    if t_now in sample_t_1e:
        sols_num_1e_a_first[t_now] = u_a.copy()
        sols_exact_1e[t_now] = exact_sol_1_II(x_1e_a, t_now)

# second order approximation of NBCs
u_a = u_initial_1e_a.copy()
for i in range(1, num_t_1e_a + 1):
    u_temp = u_a.copy()
    for j in range(1, num_x_1e_a - 1):
        u_a[j] = u_temp[j] + del_t_1e_a / del_x_1e_a**2 * (
            u_temp[j + 1] - 2 * u_temp[j] + u_temp[j - 1]
        )
    # Neumann BCs
    u_a[0] = u_temp[0] + 2 * del_t_1e_a / del_x_1e_a**2 * (
        u_temp[1] - u_temp[0]
    )  # U_x(0,t) = 0
    u_a[-1] = 0  # U(1,t) = 0
    t_now = i * del_t_1e_a
    if t_now in sample_t_1e:
        sols_num_1e_a_second[t_now] = u_a.copy()

# first order refined
u_b = u_initial_1e_b.copy()
for i in range(1, num_t_1e_b + 1):
    u_temp = u_b.copy()
    for j in range(1, num_x_1e_b - 1):
        u_b[j] = u_temp[j] + del_t_1e_b / del_x_1e_b**2 * (
            u_temp[j + 1] - 2 * u_temp[j] + u_temp[j - 1]
        )
    # Neumann BCs
    u_b[0] = u_b[1]  # U_x(0,t) = 0
    u_b[-1] = 0  # U(1,t) = 0
    t_now = i * del_t_1e_b
    if t_now in sample_t_1e:
        sols_num_1e_b_first[t_now] = u_b.copy()

# second order refined
u_b = u_initial_1e_b.copy()
for i in range(1, num_t_1e_b + 1):
    u_temp = u_b.copy()
    for j in range(1, num_x_1e_b - 1):
        u_b[j] = u_temp[j] + del_t_1e_b / del_x_1e_b**2 * (
            u_temp[j + 1] - 2 * u_temp[j] + u_temp[j - 1]
        )
    # Neumann BCs
    u_b[0] = u_temp[0] + 2 * del_t_1e_b / del_x_1e_b**2 * (
        u_temp[1] - u_temp[0]
    )  # U_x(0,t) = 0
    u_b[-1] = 0  # U(1,t) = 0
    t_now = i * del_t_1e_b
    if t_now in sample_t_1e:
        sols_num_1e_b_second[t_now] = u_b.copy()

plt.figure()
for t in sample_t_1e:
    plt.plot(x_1e_a, sols_num_1e_a_first[t], label=f"Numerical 1st Order Coarse t={t}")
    plt.plot(x_1e_a, sols_num_1e_a_second[t], label=f"Numerical 2nd Order Coarse t={t}")
    plt.plot(x_1e_b, sols_num_1e_b_first[t], label=f"Numerical 1st Order Fine t={t}")
    plt.plot(x_1e_b, sols_num_1e_b_second[t], label=f"Numerical 2nd Order Fine t={t}")
    plt.plot(x_1e_a, sols_exact_1e[t], "--", label=f"Exact t={t}")

plt.title("Problem 1.e) Numerical vs Exact Solutions")
plt.xlabel("x")
plt.ylabel("U(x,t)")
plt.legend()
plt.grid()
plt.savefig("HW1_Problem1e_Solutions.png")


# problem 2

# U_t = U_xx, 0 < x < 1, t > 0
# U(x,0) = cos(pi*x/2) 0 <= x <= 1
# U_x(0,t) = sin(2*pi*t), U_x(1,t) = 2*pi, t >= 0

del_x_2 = 0.01
del_t_2 = 0.00005  

x_2 = np.arange(0, 1 + del_x_2, del_x_2)
num_x_2 = len(x_2)
num_t_2 = int(10 / del_t_2)

u_initial_2 = np.cos(np.pi * x_2) / 2

u = u_initial_2.copy()

sample_t_2 = [0.01, 0.1, 0.5, 1, 5, 10]
sols_num_2 = {}


# second order approximation for the BCs

for i in range(1, num_t_2 + 1):
    u_temp = u.copy()
    for j in range(1, num_x_2 - 1):
        u[j] = u_temp[j] + del_t_2 / del_x_2**2 * (
            u_temp[j + 1] - 2 * u_temp[j] + u_temp[j - 1]
        )

    # Apply boundary conditions
    u[0] = np.sin(2*np.pi*i*del_t_2)

    u[-1] = (
        u_temp[-1]
        + 2 * del_t_2 / del_x_2**2 * (u_temp[-2] - u_temp[-1])
        + 2 * del_t_2 / del_x_2 * 2 * np.pi
    )

    t_now = i * del_t_2
    if t_now in sample_t_2:
        sols_num_2[t_now] = u.copy()

plt.figure()
for t in sample_t_2:
    plt.plot(x_2, sols_num_2[t], label=f"Numerical t={t}")

plt.title("Problem 2: Numerical vs Exact Solutions")
plt.xlabel("x")
plt.ylabel("U(x,t)")
plt.legend()
plt.grid()
plt.savefig("HW1_Problem2_Solutions.png")
