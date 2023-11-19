
from scipy import sparse


def spdot(A, B):
    return sparse.spmatrix.dot(A, B)
