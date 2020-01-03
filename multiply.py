#!/bin/python
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from numba import jit
MAXN = 1000

def gen_mat(n):
    mat = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(random.randint(0, MAXN))
        mat.append(row)
    
    return mat

@jit('int64[:,:](int64[:,:],int64[:,:])')
def fast_mult(a, b):
    mat = []
    for i in range(len(a)):
        mat.append([])
        for j in range(len(b[0])):
            mat[i].append(0)
            # Row i of a
            # Column j of b
            # ans[i][j] = sum(aik * bkj)
            for k in range(len(b)):
                mat[i][j]+=a[i][k] * b[k][j]
    return mat


def mult(a, b):
    mat = []
    for i in range(len(a)):
        mat.append([])
        for j in range(len(b[0])):
            mat[i].append(0)
            # Row i of a
            # Column j of b
            # ans[i][j] = sum(aik * bkj)
            for k in range(len(b)):
                mat[i][j]+=a[i][k] * b[k][j]
    return mat

times_custom = []
times_numba_custom = []
times_numpy = []
tests = 500
domain = range(0, tests, 50)
for n in domain:
    print(n)
    mat_1 = gen_mat(n)
    mat_2 = gen_mat(n)
    
    # Custom mat-mult
    start = time.time()
    mat_ans = mult(mat_1, mat_2)
    end = time.time()
    times_custom.append(end-start)

    # Custom mat-mult with numba
    start = time.time()
    mat_ans = fast_mult(mat_1, mat_2)
    end = time.time()
    times_numba_custom.append(end-start)

    np_1 = np.array(mat_1)
    np_2 = np.array(mat_2)

    # Numpy mat-mult
    start = time.time()
    mat_ans = np.matmul(np_1, np_2)
    end = time.time()
    times_numpy.append(end-start)

plt.plot(domain, times_custom, label='custom')
plt.plot(domain, times_numba_custom, label='numba custom')
plt.plot(domain, times_numpy, label='numpy')
plt.legend(loc="upper left")
plt.loglog()
plt.show()