from scipy.sparse import diags
import numpy as np
import warnings
import pandas as pd
import time
warnings.filterwarnings("ignore")

start_time = time.time()

# please note that this is version 5. I did make a version 6 but realized that what I wanted to change wasn't actually broken so I came back to this one


# This is the same function as in project 1 but it worked fine then so I hope its okay now
def inf_matrix_norm(A):
    # A is a matrix
    n = A.shape[0] # this finds the number of rows
    max_sum = 0
    for i in range (0,n):  # this looks at each individual row to find the sum of their values
        a_i = A[i, :]
        sum = 0            # this line creates a a temporary sum so as to compare to the previous defined maximum sum
        for j in a_i:
            sum += abs(j)
        max_sum = max(sum,max_sum) # this sets the maximum sum as the max value of the previous max vs the new sum
    return max_sum


# this is to just clean up the final output at the end
def final_output(A, Q ,T, ev_known, ev_calc, iteration_count):
    
    # ortho error
    print("The orthogonality error is: " + str(inf_matrix_norm(Q @ Q.T - np.identity(Q.shape[0]))))

    # rel backward error
    print("The relative backward error is: " + str(inf_matrix_norm(Q @ T @ Q.T - A)/inf_matrix_norm(A)))

    # this is to put the eigenvalues in the same order
    ev_calc.sort()
    ev_known.sort()

    abs_ev_diff = []

    # to calculate the mean error in the difference of eigenvalues
    for i in range(0,len(ev_calc)):
        abs_ev_diff.append(abs(ev_calc[i] - ev_known[i]))


    # forward eigenvalues
    print("The average forward eigenvalue error is: " + str(np.mean(abs_ev_diff)))

    # iteration count
    print("It took " + str(iteration_count) + " iterations")

    #print("Calculated eigenvalues: " + str(ev_calc))
    #print("Known eigenvalues: " + str(ev_known))

    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    #print(pd.DataFrame(T).head(10))
    #print(pd.DataFrame(A).head(10))

    print("It took " + str(time.time() - start_time) + " seconds")


# this is to create a general tri-diagonal matrix of the form:

# A = [ a  c  0  0  ...  0  0  0
#       b  a  c  0  ...  0  0  0
#       0  b  a  c  ...  0  0  0
#       .  .  .  .  ...  .  .  .
#       0  0  0  0  ...  b  a  c
#       0  0  0  0  ...  0  b  a ]  

# where A is m x m and c = b

def tri_diag_matrix_maker(m = 10, a = 2, b = -1, c = -1):
    
    k = [b* np.ones(m-1),a*np.ones(m),c *np.ones(m-1)]
    offset = [-1,0,1]
    A = diags(k,offset).toarray()

    return A 

# my own sign function which sets sign(0) = 1, why not f(0)= -1? I don't like negative numbers... I always make a silly sign error
def my_sign_function(x):
    if x >= 0:
        return 1
    else:
        return -1

# returns the known eigenvalues of the tri diagonal form of the same form as the tri_diag_matrix_maker
def known_eigenfamily(m=10, a = 2, b = -1, c = -1):
    
    known_eigenvalues = []
    known_eigenvectors = []
    
    # I googled tri-diagonal matrices and saw this was a Toeplitz matrix so I just read up more on them and saw that the eigenvalues match this form
    # Here is the source I used to create this formula: https://onlinelibrary.wiley.com/doi/10.1002/nla.1811
    for k in range(1,m+1):
        temp_val = a - 2 * np.sqrt(b*c)*np.cos((np.pi*k)/(m+1))
        known_eigenvalues.append(temp_val)
        
        temp_vec = []
        
        for j in range(1,m+1):
            temp = (2/(m+1))**0.5 * np.sin((np.pi*j*k)/(m+1))
            temp_vec.append(temp)
        known_eigenvectors.append(temp_vec)

    return known_eigenvalues, known_eigenvectors

# my function to calculate the wilkinson shift
def wilkinson(T: np.ndarray):
    T = np.array(T)
    a_m = T[1,1]
    a_m_1 = T[0,0]
    b_m_1 = T[1, 0]

    delta = (a_m_1 - a_m)/2
    
    mu = a_m - my_sign_function(delta) * b_m_1**2 / (np.abs(delta) + np.sqrt(delta**2 + b_m_1**2))

    return mu

# here is the main function to calculate the shifted QR iterations for the eigenvalues 
def explicit_shifted_QR(A, 
                        tol = 1E-16, 
                        max_iterations=10000 # this was a sanity check since I normally avoid "while" statements like the plague
                        ):
    

    # this intialized all the variables in the algorithm
    m = A.shape[0]
    l = m-1
    Q_tot = np.identity(m)
    W = np.zeros((2,l))
    iteration_count = 0

    T = np.array(A.copy())


    while l > -1 and iteration_count < max_iterations:

        if np.abs(T[l,l-1]) < tol:

            T[l,l-1] = 0
            T[l-1,l] = 0
            l -= 1
            
            continue
        
        T_sub = T[l-1:l+1, l-1:l+1]

        mu = wilkinson(T_sub)

        # shifting the T matrix with the wilkinson shift
        T -= mu * np.identity(m)

        for j in range (0,l):
            
            # this section is to calculate the householder vectors
            x = T[j:j+2, j]
            v = x.copy()
            e1 = np.zeros((x.shape[0]), dtype=int)
            e1[0] = 1
            v[0] +=  my_sign_function(x[0])*np.linalg.norm(x)
            v /= np.linalg.norm(v)

            # to store the householder vectors
            W[:,j] = v

            end = min(m, j+2) + 1
            
            # everything adjusting the T matrix has the slice ending in "j+2" to account for the slice actually ending on the time before it stated end
            T[j:j+2, j:end] -= 2 * np.outer(v, v @ T[j:j+2, j:end])

            # first hard set to 0
            T[j+1,j] = 0

            # updating the Q matrix with the efficient householder transformation
            Q_tot[:, j:j+2] -= 2 * np.outer(Q_tot[:, j:j+2] @ v, v)
            
        for j in range (0,l):    
            
            # recalling the correct householder vector
            v = W[:,j]

            # I spent 8 hours trying to debug this and it turns out to be the zero in the start variable
            # it all came down to miscounting the start of the lists in the algorithm vs python
            # I felt vindicated when I found this 
            # and also slightly stupid for not catching it earlier  
            start = max(0, j-1)

            T[start:j + 2, j:j+2] -= 2 * np.outer(T[start:j + 2, j:j+2] @ v, v)

            # the next hard zero set
            if j  > 0:
                T[j-1,j+1] = 0
        
        # unshifting the T matrix
        T += mu * np.identity(m)

        # counting up the iterations
        iteration_count += 1

    # returning the eigenvalues purely the diagonals of T
    eigenvalues = list(np.diag(T))

    return Q_tot, T, iteration_count, eigenvalues

# these next two lines are so I could play with the algorithm. I know it only works on symmetric matrices but I found it beneficial to check for my own sanity
m = 10
tri_diag_row = [-1, 2, -1]

# initializing the nown eigenvalues
known_eva, known_eve = known_eigenfamily(m=m, a = tri_diag_row[1], b= tri_diag_row[0], c= tri_diag_row[2])

# making the symmetric tridiagonal matrix
A = tri_diag_matrix_maker(m=m, a = tri_diag_row[1], b= tri_diag_row[0], c= tri_diag_row[2])


Q_calc, T_calc, iteration_count, eigenvalues_calc = explicit_shifted_QR(A)

final_output(A=A, Q=Q_calc, T=T_calc, ev_calc= eigenvalues_calc, ev_known= known_eva, iteration_count= iteration_count)

    