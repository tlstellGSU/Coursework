import numpy as np
from matplotlib import pyplot as plt
import warnings
import time
warnings.filterwarnings("ignore")

# note: I left in the print functions to check the shape/type of different variables at different points
# This is because I wanted to show some of my work in creating and then checking the code 

# this just allows me to time how long it took
t_0 = time.time()

# this function creates a vandermonde matrix for some given grid of x points
def V_matrix(x_list,N):
    # x_list is an array of x grid points, N is the degree 
    return(np.array([[element**i for i in range(N)] for element in x_list]))

# this creates a column vector for sin(x) values across a given grid of x points
def sin_matrix(x_list):
    # x_list is the array of x grid points to evaluate at
    return(np.array([np.sin(i) for i in x_list]).reshape(-1,1))

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

# this allows me to easily create a grid of points for a given range from a to b and N grid points
def x_list_gen(a, b, N):
    # a is the inital point, b is the final point, and N is the number of points
    h = (b-a)/N   # I know you said to use h = 2/5 but I wanted a function that could allow me to play with the approximation more...
    temp = [a + h*j for j in range(N)]
    return temp, h

# this is a function to compute the back subsitution of a matrix R and a vector b to solve the expression: Rx=b
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

# this creates a list of f(x) given that f(x) is a polynomial with coefficients given by coeffs
# and evaluated at the x values given by x_list
def polynomial_function_y(coeffs,x_list):
    # coeffs is the array of coefficients as determined by the least squares algorithm
    # x_list is the grid points to be examined
    y = []

    # this is just for evaluteding the polynomial at a single point
    if len( x_list) ==1:
        temp = 0
        for i in range(len(coeffs)):
            temp += coeffs[i]*x_list[0]**i
        return float(temp)
    
    # this takes and generates a y value given the coefficients
    for element in x_list:
        temp = 0
        for i in range(len(coeffs)):
            temp += coeffs[i]*element**i
        y.append(temp)
    return y

# this is my implementation of horners method
def horners_method(x0, coeffs):
    # x0 is the x value to evaluate at, coeffs is the coefficent matrix determined by the least squares algorithm
    temp = 0
    for element in reversed(coeffs):
        temp = temp * x0 + element
    return temp

# creating the x grid
x_list, h = x_list_gen(-4,4,20)

# this is just to show that the correct h value is used
print("the h value is = " + str(h) + "\n")

# intitalizing A as a vandermonde matrix of degree 8
A = V_matrix(x_list,8)

# creating the Q and R matrices using the householder algorithm 
Q,R, W = householder(A)  # this is using my function
q,r = np.linalg.qr(A)  # this is using the numpy defined function just to check my work

# this checks and computes the computed backward error
print("The computed backward error = " + str(inf_matrix_norm(Q @ R - A)/inf_matrix_norm(A)) + str("\n"))

# this checks and computes the orthogonality error
print("The orthogonality error = " + str(inf_matrix_norm(Q.T @ Q - np.identity(Q.shape[0])))+ str("\n"))

# this checks and computes the error between my QR factorization and the built in QR factorization
print("The error between my function for QR vs the built-in version = " + str(inf_matrix_norm(Q @ R - q @ r))+ str("\n"))

# this initializes a vector of sin(x) values evaluated at the x grid
b = sin_matrix(x_list)

# defining the coeffs of the polynomial as described earlier
coeffs = [float(x) for x in least_sq_QR(A,b)]

# Here is where I print W
print(W)
print("\n")

#print(coeffs)

# finding f(pi) for the given polynomial
p_pi = horners_method(np.pi, coeffs)

# this checks the error between using horners method vs my polynomial evaluation
print("the difference between horners method and the polynomial function = " + str(abs(p_pi - polynomial_function_y(coeffs,[np.pi])))+ str("\n"))

# this finds the absolute error in evaluting |p(pi) - sin(pi)| where sin(pi)=0
print("The absolute error in p(pi) = " + str(abs(p_pi))+ str("\n"))

# this creates a series of f(x) values for the polynomial
p = polynomial_function_y(coeffs,x_list)

# this allows me to check how long the program ran for
print("This took " + str( time.time()- t_0) + " seconds"+ str("\n"))

# this creates a plot of the polynomial and the sin(x)
plt.plot( x_list, sin_matrix(x_list), label = "sin(x)")
plt.plot( x_list, p, label = "polynomial")
plt.legend()
plt.show()