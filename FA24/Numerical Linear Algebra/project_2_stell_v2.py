from scipy.sparse import diags
import numpy as np
import warnings
import pandas as pd
warnings.filterwarnings("ignore")

def inf_matrix_norm(A):
    n = A.shape[0]
    max_sum = 0
    for i in range(n):
        row_sum = np.sum(np.abs(A[i, :]))
        max_sum = max(row_sum, max_sum)
    return max_sum

def final_output(A, Q, T, ev_known, ev_calc):
    print("The orthogonality error is:", inf_matrix_norm(Q @ Q.T - np.identity(Q.shape[0])))
    print("The relative backward error is:", inf_matrix_norm(Q @ T @ Q.T - A) / inf_matrix_norm(A))

    ev_calc.sort
    ev_known.sort
    abs_ev_diff = [abs(ev_calc[i] - ev_known[i]) for i in range(len(ev_calc))]
    print("The average forward eigenvalue error is:", np.mean(abs_ev_diff))

    print("Eigenvalues (calculated):", ev_calc)
    print("Eigenvalues (known):", ev_known)

    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    print(pd.DataFrame(T).head(10))

def tri_diag_matrix_maker(m=10):
    k = [np.ones(m-1), -2 * np.ones(m), np.ones(m-1)]
    offset = [-1, 0, 1]
    A = diags(k, offset).toarray()
    return A

def my_sign_function(x):
    return 1 if x >= 0 else -1

def known_eigenfamily(m=10):
    known_eigenvalues = []
    known_eigenvectors = []
    for k in range(1, m+1):
        temp_val = 2 - 2 * np.cos((np.pi * k) / (m + 1))
        known_eigenvalues.append(temp_val)
        temp_vec = [(2 / (m + 1))**0.5 * np.sin((np.pi * j * k) / (m + 1)) for j in range(1, m+1)]
        known_eigenvectors.append(temp_vec)
    return known_eigenvalues, known_eigenvectors

def wilkinson(T):
    a_m = T[1, 1]
    a_m_1 = T[0, 0]
    b_m_1 = T[1, 0]
    delta = (a_m_1 - a_m) / 2
    mu = a_m - my_sign_function(delta) * b_m_1**2 / (np.abs(delta) + np.sqrt(delta**2 + b_m_1**2))
    return mu

def explicit_shifted_QR(A, tol=1E-12, max_iterations=1000):
    m = A.shape[0]
    Q_tot = np.identity(m)
    T = np.array(A.copy())
    iteration_count = 0
    l = m - 1

    while l > 0 and iteration_count < max_iterations:
        if np.abs(T[l, l-1]) < tol:
            T[l, l-1] = 0
            l -= 1
            continue
        
        T_sub = T[l-1:l+1, l-1:l+1]
        mu = wilkinson(T_sub)

        for j in range(l):
            x = T[j:j+2, j] - mu * (np.identity(2)[:, 0])
            v = x.copy()
            v[0] += my_sign_function(x[0]) * np.linalg.norm(x)
            v /= np.linalg.norm(v)

            # Apply Householder transformation
            T[j:j+2, j:] -= 2 * np.outer(v, v @ T[j:j+2, j:])
            T[:, j:j+2] -= 2 * np.outer((T[:, j:j+2] @ v), v)
            Q_tot[:, j:j+2] -= 2 * np.outer((Q_tot[:, j:j+2] @ v), v)

        T += mu * np.identity(m)
        iteration_count += 1

    eigenvalues = np.diag(T)
    return Q_tot, T, iteration_count, eigenvalues

m = 10
known_eva, known_eve = known_eigenfamily(m=m)
A = tri_diag_matrix_maker(m=m)
Q_calc, T_calc, iteration_count, eigenvalues_calc = explicit_shifted_QR(A)
final_output(A=A, Q=Q_calc, T=T_calc, ev_calc=eigenvalues_calc, ev_known=known_eva)
