from scipy.sparse import diags
import numpy as np
import warnings
import pandas as pd
warnings.filterwarnings("ignore")

def inf_matrix_norm(A):
    # A is a matrix
    n = A.shape[0] # this finds the number of rows
    max_sum = 0
    for i in range (0,n):  # this looks at each individual row to find the sum of their values
        a_i = A[i, :]
        sum = 0             # this line creates a a temporary sum so as to compare to the previous defined maximum sum
        for j in a_i:
            sum += abs(j)
        max_sum = max(sum,max_sum) # this sets the maximum sum as the max value of the previous max vs the new sum
    return max_sum


def final_output(A, Q ,T, ev_known, ev_calc, R):
    
    # othro error
    print("The orthogonality error is: " + str(inf_matrix_norm(Q @ Q.T - np.identity(Q.shape[0]))))

    # rel backward error
    print("The relative backward error is: " + str(inf_matrix_norm(Q @ T @ Q.T - A)/inf_matrix_norm(A)))

    ev_calc.sort()
    ev_known.sort()

    abs_ev_diff = []

    for i in range(0,len(ev_calc)):
        abs_ev_diff.append(abs(ev_calc[i] - ev_known[i]))


    # forward eigenvalues
    print("The average forward eigenvalue error is: " + str(np.mean(abs_ev_diff)))

    print("It took " + str(iteration_count) + " iterations")

    print(ev_calc)
    print(ev_known)

    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    print(pd.DataFrame(T).head(10))

    
def tri_diag_matrix_maker(m = 10):
    
    k = [np.ones(m-1),-2*np.ones(m),np.ones(m-1)]
    offset = [-1,0,1]
    A = diags(k,offset).toarray()

    return np.zeros(A.shape[0]) - A


def my_sign_function(x):
    if x >= 0:
        return 1
    else:
        return -1

def known_eigenfamily(m=10):
    
    known_eigenvalues = []
    known_eigenvectors = []
    
    for k in range(1,m+1):
        temp_val = 2 - 2 * np.cos((np.pi*k)/(m+1))
        known_eigenvalues.append(temp_val)
        
        temp_vec = []
        
        for j in range(1,m+1):
            temp = (2/(m+1))**0.5 * np.sin((np.pi*j*k)/(m+1))
            temp_vec.append(temp)
        known_eigenvectors.append(temp_vec)

    return known_eigenvalues, known_eigenvectors

def wilkinson(T: np.ndarray):
    T = np.array(T)
    a_m = T[1,1]
    a_m_1 = T[0,0]
    b_m_1 = T[1, 0]

    delta = (a_m_1 - a_m)/2
    

    mu = a_m - my_sign_function(delta) * b_m_1**2 / (np.abs(delta) + np.sqrt(delta**2 + b_m_1**2))

    return mu

def explicit_shifted_QR(A, tol = 1E-16, max_iterations=1000):
    m = A.shape[0]
    
    l = m-1
    Q_tot = np.identity(m)
    R = A.copy()
    W = np.zeros((2,m-1))
    iteration_count = 0
    T = np.array(A.copy())


    while l > 0 and iteration_count < max_iterations:
        if np.abs(T[l,l-1]) < tol:
            T[l,l-1] = 0
            T[l-1,l] = 0
            l -= 1
            
            continue
        
        T_sub = T[l-1:l+1, l-1:l+1]
        
        mu = wilkinson(T_sub)

        T -= mu * np.identity(m)

        for j in range (0,l):
            x = T[j:j+2, j]
            v = x.copy()
            #v_temp = my_sign_function(x[0])* np.linalg.norm(x)
            e1 = np.zeros((x.shape[0]), dtype=int)
            e1[0] = 1
            #v = v_temp * e1 + x
            v[0] +=  my_sign_function(x[0])*np.linalg.norm(x)
            v /= np.linalg.norm(v)

            W[:,j] = v

            end = min(m, j+2) + 1
            
            T[j:j+2, j:end] -= 2 * np.outer(v, v @ T[j:j+2, j:end])
            T[j+1, j] = 0

            Q_tot[:, j:j+2] -= 2 * np.outer(Q_tot[:, j:j+2] @ v, v)

        for j in range (0,l):
            v = W[:,j]
            #print(v)

            start = max(1,j-1)
            #print(T[start:j+2, j:j+2])
            

            T[start:j+2, j:j+2] -= 2 * np.outer(T[start:j+2, j:j+2] @ v, v)

            if j -1 > 0:
                T[j-1, j+1] = 0
            


        T += mu * np.identity(m)

        iteration_count += 1
        #print(f"Iteration {iteration_count}, l ={l}")

    eigenvalues = list(np.absolute(np.diag(T)))


    return Q_tot, T, iteration_count, eigenvalues, R


m = 10

known_eva, known_eve = known_eigenfamily(m=m)

A = tri_diag_matrix_maker(m=m)

Q_calc, T_calc, iteration_count, eigenvalues_calc, R = explicit_shifted_QR(A)
final_output(A=A, Q=Q_calc, T=T_calc, ev_calc= eigenvalues_calc, ev_known= known_eva, R=R)


