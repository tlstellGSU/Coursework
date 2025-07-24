import numpy as np
import scipy.sparse as sp
import time


start_time = time.time()
print("\n")

print("This takes a moment to run")
print("The main time consumer is making the list of known eigenvalues")
print("\n")
# fun fact, I use 137 as my random seed every time. I even have it tattooed on my arm (not because I use it as a random seed, for other reasons)
# I did pick a random seed because I wanted to have the random/arbitrary vector "b" but I still wanted control over it so I could check the code
state = np.random.seed(137)

# the following are stolen directly from you and no changes were made

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
B = np.random.randn(l**2,1)

X = mul_invA(B)
R = A @ X - sigma * X - B
#print(np.linalg.norm(R, ord=np.inf))

# this ends stuff provided by you

# Here, I made a function which directly computed the known eigenvalues of the matrix A
def known_eigenvalues(l=l):
    
    all_eigenvalues = []
    
    for i in range(0,l):
        for j in range(0,l):
            temp = 4*(np.sin(i*np.pi/(2*(l+1)))**2 + np.sin(j*np.pi/(2*(l+1)))**2)
            all_eigenvalues.append(temp)
    
    all_eigenvalues.sort()

    return all_eigenvalues

all_ev = known_eigenvalues()

time_1 = time.time()
print("making the known eigenvalues took " + str(time_1 - start_time)[:7] + " seconds")

# Here, I started to make the Lanczos algorithm 

def lanczos_algorithm(A, n =100, reortho = True, b = B):
    m = len(b)
    T = np.zeros((n,n))
    Q = np.zeros((m,n))
    b = b.reshape(-1) # this was included because it kept giving me errors about how python was initializing the vector b so I just had to reshape it to force it to be a column vector
    Q[:,0] = b/np.linalg.norm(b)
    beta = 0
    q_0 = np.zeros((m))
    for j in range(0,n):
        v = mul_invA( Q[:,j])
        alpha = np.dot(Q[:,j], v)
        T[j,j] = alpha
        v -= beta*q_0 + alpha* Q[:,j]
        q_0 = Q[:,j]
        beta = np.linalg.norm(v)
        if j + 1 < n: # the j < n had to be adjusted to match the indexing in python
            T[j,j+1] = beta
            T[j+1,j] = beta
            Q[:,j+1] = v/beta
            if reortho: # I made it so I didn't have to comment out code, I could just call the same function again with reortho turned on or off
                for i in range(j + 1):  
                    Q[:, j + 1] -= np.dot(Q[:,i],Q[:,j+1])*Q[:,i] # The first time I wrote this, it crashed my computer. The next 15 times I ran it, it did the same thing :(
                Q[:, j + 1] /= np.linalg.norm(Q[:, j + 1], ord=2) # One issue I ran into was that it was not defaulting to the two norm for some reason so I wound up just forcing it to default. Turns out that only happens for matrices and not vectors
    return Q,T


n = 100

#known_ev_values = known_eigenvalues(l=l)

# This is where I made the final outputs be easy to call for both with and without reorthogonalization

def final_output(reortho, sigma = sigma, n = n):
    print("\n")

    if reortho:
        print("This is using reorthogonalization")
    else:
        print("This is not using reorthogonalization")
    print("\n")
    Q,T = lanczos_algorithm(A=A, n=n, reortho=reortho)
    theta_vec, V = np.linalg.eigh(T)
    lambda_vec = [sigma + 1/theta for theta in theta_vec]
    Y = Q@V
    # I made a temp holder for the matrix Y.T @ Y because I had issues with the identity matrix size
    Y_T_Y = Y.T @ Y
    Y_temp = np.linalg.norm(Y_T_Y - np.identity(Y_T_Y.shape[0]), ord = 2)
    print("The orthogonality error is: " + str(Y_temp))
    print("\n")

    kappa = np.linalg.cond(Y)
    print("The condition number of Y is: " + str(kappa))
    print("\n")

    # Here is where I started trying to find the residues left from the computed eigenvalues
    # And this next part counts how many small (under 10E-12) residues of eigenvalues there are

    residues = []
    tight_ev = []
    count_small_eigval = 0
        
    for k in range(0,n):
        temp = np.linalg.norm(A@ Y[:,k] - lambda_vec[k]*Y[:,k])
     
        residues.append(temp)

        if temp <= 10E-15:
            count_small_eigval += 1
            tight_ev.append(lambda_vec[k])

    residues_mean = sum(residues)/len(residues)
        
    
    print("The number of small eigenvalue resiudes is: " + str(count_small_eigval))
    print("\n")

    print("The average residue is: " + str(residues_mean))
    print("\n")
    lambda_vec.sort()
    #print(lambda_vec)
    #print("\n")
    repeated_ev = list(set([element for element in tight_ev if tight_ev.count(element) >= 2]))
    #print(repeated_ev)
    print("There are " + str(len(repeated_ev)) + " eigenvalues with multiplicity >= 2")
    
    print("\n")



    return tight_ev



reortho_lambda = final_output(reortho=True)

time_2 = time.time()

print("Running the Lanczos algorithm for the reorthogonalization case took " + str(time_2 - time_1)[:7] + " seconds")
print("\n")

no_reortho_lambda = final_output(reortho=False)

print("Running the Lanczos algorithm for the non-reorthogonalization case took " + str(time.time() - time_2)[:7] + " seconds")
print("\n")

#reortho_lambda.sort()
#no_reortho_lambda.sort()

#print(reortho_lambda)
#print("\n")
#print(no_reortho_lambda)
#print("\n")

reortho_lambda_unqiue = list(set(reortho_lambda))
non_reortho_lambda_unique = list(set(no_reortho_lambda))
actual_ev_unqiue = list(set(all_ev))
reortho_count = 0
non_reortho_count = 0


for actual in actual_ev_unqiue:
    for element in reortho_lambda_unqiue:
        if abs(element - actual) <= 10E-15:
            reortho_count += 1
    for element in non_reortho_lambda_unique:
        if abs(element - actual) <= 10E-15:
            non_reortho_count += 1


print("The number of exactly converged eigenvalues in the reorthogonalization case are: ")
print(reortho_count)
print("\n")

print("The number of exactly converged eigenvalues in the non-reorthogonalization case are: ")
print(non_reortho_count)
print("\n")


# The run time was of interest to me since it kept crashing my computer but it wound up working eventually

print("All together, it took " + str(time.time() - start_time)[:7] + " seconds")

print("\n")
