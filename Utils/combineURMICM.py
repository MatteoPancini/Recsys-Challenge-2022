import numpy as np
from scipy import sparse as sp

def combine(ICM: sp.csr_matrix, URM : sp.csr_matrix):
    return sp.hstack((URM.T, ICM), format='csr')