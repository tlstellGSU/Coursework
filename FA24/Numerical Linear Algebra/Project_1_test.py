import numpy as np

def qr(A):
    m, n = A.shape
    Q = np.eye(m)
    for i in range(n - (m == n)):
        H = np.eye(m)
        H[i:, i:] = make_householder(A[i:, i])
        Q = np.dot(Q, H)
        A = np.dot(H, A)
    return Q, A

def make_householder(a):
    v = a / (a[0] + np.copysign(np.linalg.norm(a), a[0]))
    v[0] = 1
    H = np.eye(a.shape[0])
    H -= (2 / np.dot(v, v)) * np.dot(v[:, None], v[None, :])
    return H

def V_matrix(x,N):
    return(np.array([[element**i for i in range(N)] for element in x]))

x_list = [-4+j*2/5 for j in range(0,20)]

Q,R = qr(V_matrix(x_list,8))

q,r = np.linalg.qr(V_matrix(x_list,8))

check = q @ r - Q @ R
print(check)