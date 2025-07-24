import numpy as np
import scipy.sparse as sp
import random

state = random.seed(137)

# the following are stolen directly from you

l = 305
m = l**2
sigma = 6.001

v=np.ones(l**2)
A1=sp.spdiags([-v,2*v,-v], [-1,0,1],l,l)
I_l=sp.eye(l)
A = sp.kron(I_l,A1,format='csc') + sp.kron(A1,I_l)

# Return a function to solve (A-sigma I) Y = B
def make_mul_inv(A, sigma):
    F = sp.linalg.splu(A - sigma * sp.eye(m))
    def mul_invA(B):
        Y = F.solve(B)
        R = A @ Y - sigma * Y - B
        # Use the residual to do on step of iterative refinement.
        return Y - F.solve(R)
    return mul_invA

mul_invA = make_mul_inv(A, sigma)
B = np.random.randn(l**2,2)

X = mul_invA(B)
R = A @ X - sigma * X - B
print(np.linalg.norm(R, ord=np.inf))

# this ends stuff provided by you


def known_eigenvalues(l=305):
    
    all_eigenvalues = []
    
    for i in range(0,l):
        for j in range(0,l):
            temp = 4*(np.sin(i*np.pi/(2*(l+1)))**2 + np.sin(j*np.pi/(2*(l+1)))**2)
            all_eigenvalues.append(temp)
    
    all_eigenvalues.sort()

    return all_eigenvalues

def lanczos_algorithm(A, n =100, reortho = True):
    m = A.shape[0]
    T = np.zeros((n,n))
    Q = np.zeros((m,n))
    b = np.random.rand(m)
    Q[:,0] = b/np.linalg.norm(b)
    beta = 0
    q_0 = np.zeros((m,1))
    for j in range(0,n):
        v = mul_invA(Q[:,j])
        alpha = Q[:,j].T @ v
        T[j,j] = alpha
        v -= beta*q_0 - alpha* Q[:,j]
        q_0 = Q[:,j]
        beta = np.linalg.norm(v)
        if j+1 < n:
            T[j,j+1] = beta
            T[j+1,j] = beta
            Q[:,j+1] = v/beta
            if reortho:
                yes = 1
            Q[:,j+1] -= Q[:,0:j] @ (Q[:,0:j].T @ Q[:,j+1])
            Q[:,j+1] /= np.linalg.norm(Q[:,j+1], ord =2)
        
    return Q,T

Q,T = lanczos_algorithm(A=A, n=100, reortho=False)

def final_output(Q=Q,T=T, sigma = sigma, n = 100):
    theta_vec, V = np.linalg.eig(T)
    lambda_vec = [sigma + 1/element for element in theta_vec]
    Y = Q@V
    print("The orthogonality error is: " + str(np.linalg.norm(Y.T @ Y - np.identity(Y.size), ord = 2)))
    kappa = np.linalg.norm(Y) * np.linalg.norm(make_mul_inv(Y,sigma = 0))
    print("The condition number of Y is: " + str(kappa))

    residues = []

    for k in range(0,n):
        temp = A@ Y[:,k] - lambda_vec[k]*Y[:,k]
        residues.append(np.linalg.norm(temp))

    count_small_eigval = 0
    for element in residues:
        if element <= 10E-12:
            count_small_eigval += 1
    
    print("The number of small eigenvalues in the resiudes is: " + str(count_small_eigval))


final_output()






