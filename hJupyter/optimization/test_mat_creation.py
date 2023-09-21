import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, linalg
import time as tm



# Create matrix by stacking matrices

# Function to create sparse matrix with random values
def create_sparse_matrix(n, m, density):
    # Create random matrix
    A = np.random.rand(n, m)
    # Create sparse matrix
    A = csr_matrix(A)
    # Create mask
    mask = np.random.rand(n, m) < density
    # Apply mask
    A = A.multiply(mask)
    return csc_matrix(A)


n= 100
m = 150
# Create two sparse matrices
A = create_sparse_matrix(n, m, 0.1)
B = create_sparse_matrix(n, m, 0.1)

print("A: ", A)

tic = tm.time()
# Stack matrices
C = np.vstack((A, B))
toc = tm.time()

print("Time to stack matrices: ", toc - tic)

# Get indices of non-zero elements for A and B
i1, j1 = A.nonzero()
i2, j2 = B.nonzero()

print("i1: ", len(i1))
print("j1: ", len(j1))

print("i2: ", len(i2))
print("j2: ", len(j2))

# Merge indices
i = np.hstack((i1, i2 + A.shape[0]))
j = np.hstack((j1, j2))


Avalues = [ A[i,j] for i, j in zip(*A.nonzero())]
Bvalues = [ B[i,j] for i, j in zip(*B.nonzero())]
# Get values at indices i, j for A and B
values = np.hstack((Avalues, Bvalues))

print("i: ", len(i))
print("j: ", len(j))

print("i: ", i[:5])
print("j: ", j[:5])
print("values: ", values[:5])


tic = tm.time()
# Create sparse matrix
C2 = coo_matrix((values, (i, j)), shape=(A.shape[0] + B.shape[0], A.shape[1]))
toc = tm.time()

print("Time to create sparse matrix: ", toc - tic)
