# Project 2 - Numerical Linear ALgebra - Stell
from scipy.sparse import diags
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def back_subsitution(b, R):
    # b is a column vector, R is a matrix
    n = R.shape[1]
    x = np.zeros((n)) # this initializes x as a zero vector
    #x = x.reshape(-1,1)
    R = np.array(R)
    
    for i in range(n-1,-1,-1):
        x[i] = b[i]
        for j in range(i+1,n):
            x[i] -= R[i,j]*x[j]
        x[i] /= R[i,i]
    return x

# this is the implemntaton of least squares using QR factorization combining the previosuly defined functions
def least_sq_QR(A,b):
    # A is a matrix, b is a column vector
    Q,R, W = householder(A) # this uses the Stell-made Householder function
    v_b = Q.T @ b
    
    x = back_subsitution(v_b, R)
    return x

# this computes and return the infinite matrix norm of a vector A
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

# this computes the householder transformaton and returns the matrices Q and R
def householder(A: np.ndarray):
    # A is a matrix initialized as a numpy array
    m,n = A.shape
    R = A.copy() # this initializes R as A
    Q = np.identity(m) # this intitailzes Q as the identity matrix
    W = np.zeros((m,n))
   
   # print(A.shape)
    for k in range(0,n): # this sums over the n columns
        x = R[k:, k] # this creates the x vector as the stated in the householder algorithm
        #print(x.shape)
       
        v_k_temp = np.sign(x[0])*np.linalg.norm(x) # I created a temp value to represent the first portion of the v_k vector
        e1 = np.zeros((x.shape[0]), dtype = int) # this initializes the unit vector as a kx1 vector of zeroes
        e1[0] = 1 # this sets the first value of the unit vector to 1
        #print(e1.shape)
        v_k = v_k_temp * e1 + x 
        v_k /= np.linalg.norm(v_k) # this creates and normalizes the v_k vector
        
        #print(v_k.shape)
        #print(R[k:,k:].shape)
        #R_temp = v_k.T @ R[k:,k:]
        #print(R_temp.shape)
        #print(v_k.shape)
        
        R[k:,k:] -= 2 * np.outer(v_k, v_k @ R[k:,k:]) # this updates the R matrix 
        #Q[:,k:] -= 2 * (Q[:,k:] @ np.outer(v_k,v_k))
        Q[:, k:] -= 2 * np.outer(Q[:, k:] @ v_k, v_k) # this updates the Q matrix
        #R[k:,k:] = R[k:,k:] - 2 *v_k @ (v_k.T * R[k:, k:])
       # R[k:,k:] = R[k:,k:] - 2 *v_k @ np.dot(v_k, R[k:,k:])

        # this is where I create the matrix W
        W[k:, k] = v_k
            
        

    #return Q[:n].T, np.triu(R[:n])
    return Q, R, W
    # Q and R are defined by the Householder transformation algorithm


def tri_diag_matrix_maker(m = 10):
    
    k = [np.ones(m-1),-2*np.ones(m),np.ones(m-1)]
    offset = [-1,0,1]
    A = diags(k,offset).toarray()

    return A

def known_eigenfamily(m=10):
    
    known_eigenvalues = []
    known_eigenvectors = []
    
    for k in range(1,m+1):
        temp_val = 2 - 2 * np.cos((np.pi()*k)/(m+1))
        known_eigenvalues.append(temp_val)
        
        temp_vec = []
        
        for j in range(1,m+1):
            temp = (2/(m+1))**0.5 * np.sin((np.pi()*j*k)/(m+1))
            temp_vec.append(temp)
        known_eigenvectors = np.array(known_eigenvectors.append(temp_vec))

    return known_eigenvalues, known_eigenvectors

def explicit_shifted_QR(m = 10, tol = 1E-16):
    l = m-1
    Q = np.identity(m)
    W = np.zeros(2,m-1)
    while l > 0:
        