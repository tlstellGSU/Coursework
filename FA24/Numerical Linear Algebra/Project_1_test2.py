import numpy as np

def householder_transformation(A):
    """
    Perform QR decomposition using Householder transformations without explicitly computing Householder matrices.
    
    Parameters:
    A (numpy.ndarray): The matrix to be decomposed.
    
    Returns:
    tuple: (Q, R) where Q is the orthogonal matrix and R is the upper triangular matrix.
    """
    A = A.astype(float)
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy()
    
    for i in range(min(m, n)):
        # Extract the subvector from the current column
        x = R[i:, i]
        # Create the vector for Householder reflection
        norm_x = np.linalg.norm(x)
        sign = np.sign(x[0])
        u1 = x[0] + sign * norm_x
        u = x / u1
        u[0] = 1
        u = u.reshape(-1, 1)
        
        # Compute the Householder reflection without explicitly creating H
        # Update R
        R[i:, i:] -= 2 * u @ (u.T @ R[i:, i:])
        
        # Update Q
        Q[:, i:] -= 2 * (Q[:, i:] @ u) @ u.T
    
    return Q, R

def V_matrix(x,N):
    return(np.array([[element**i for i in range(N)] for element in x]))

x_list = [-4+j*2/5 for j in range(0,20)]

Q,R = householder_transformation(V_matrix(x_list,8))

q,r = np.linalg.qr(V_matrix(x_list,8))

check = q @ r - Q @ R
print(check)