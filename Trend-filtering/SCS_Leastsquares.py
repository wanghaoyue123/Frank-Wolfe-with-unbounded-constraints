import numpy as np
import time
import cvxpy as cp
import scipy.sparse as sparse

def SCS_Leastsquares(r, A, b, delta, eps):
    ################################################################
    # Use CVXPY with SCS solver to solve the problem
    #
    # minimize \|Ax - b\|_2^2
    # s.t. \|D^(r) x\|_1 <= delta
    #
    # A is an (N times n) real matrix
    # b is a vector with length N
    # D^(r) is the r-order discrete difference operator
    #
    #
    ################################################################

    ## Construct the sparse matrix H = D^(r)
    N = A.shape[0]
    n = A.shape[1]

    e = np.ones((n,))
    if r == 1:
        H = sparse.diags([e, -e], [0, 1], shape=(n - 1, n))
    else:
        if r == 2:
            H = sparse.diags([e, -2 * e, e], [0, 1, 2], shape=(n - 2, n))

    ## Solve with cvxpy
    u = cp.Variable(n)
    b_ = np.reshape(b, (N,))

    constraints = [cp.norm(H * u, 1) <= delta]
    obj = cp.Minimize(cp.norm(b_ - A * u))
    prob = cp.Problem(obj, constraints)

    start = time.time()
#     cvxpy_optval = prob.solve(solver = 'GUROBI')
    cvxpy_optval = prob.solve(solver = 'SCS', eps = eps)
    # cvxpy_optval = prob.solve(solver = 'CPLEX')
    # cvxpy_optval = prob.solve(solver = 'ECOS')
#     cvxpy_optval = prob.solve(solver='MOSEK')
    # cvxpy_optval = prob.solve(solver = 'ECOS_BB')
    # cvxpy_optval = prob.solve(solver = 'CVXOPT')
    end = time.time()
    time_cvxpy = end - start


    return time_cvxpy, cvxpy_optval




