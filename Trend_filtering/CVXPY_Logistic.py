import numpy as np
import time
import cvxpy as cp
import scipy.sparse as sparse

def CVXPY_Logistic(r, A, delta):
    ################################################################
    # Use the CVXPY with Mosek solver to solve the problem
    #
    # minimize \sum_{i=1}^N log(1 + exp(a_i^T x))
    # s.t. \|D^(r) x\|_1 <= delta
    #
    # A = [a_1^T; a_2^T; ... ; a_N^T] \in R^{ N x n }
    # D^(r) is the r-order discrete difference operator
    #
    #
    # Author: Haoyue Wang
    # Email: haoyuew@mit.edu
    # Date: 2019. 11. 24
    # Reference:
    ################################################################

    #### Construct the diff matrix H
    N = A.shape[0]
    n = A.shape[1]
    e = np.ones((n,))
    if r == 1:
        H = sparse.diags([e, -e], [0, 1], shape=(n - 1, n))
    else:
        if r == 2:
            H = sparse.diags([e, -2 * e, e], [0, 1, 2], shape=(n - 2, n))

    ## Start solving
    start = time.time()
    u = cp.Variable(n)
    constraints = [cp.norm(H * u, 1) <= delta]

    obj = cp.Minimize(cp.sum(cp.logistic(A * u)))
    prob = cp.Problem(obj, constraints)

    # cvxpy_optval = prob.solve(solver = 'SCS')
    # cvxpy_optval = prob.solve(solver = 'ECOS')
    # cvxpy_optval = prob.solve(solver = 'ECOS_BB')
    cvxpy_optval = prob.solve()
    end = time.time()
    time_cvxpy = end - start

    return time_cvxpy, cvxpy_optval