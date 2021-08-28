import numpy as np
import scipy
import matplotlib.pyplot as plt
import time
import cvxpy as cp
import mosek
from numpy import genfromtxt
import scipy.sparse as sparse
import random






def solve_A_new(A, L, beta, tau, tol=1e-13):
    
    rhs = tau**2 + L*A/beta - 0.5
    u = 1
    lhs = A/u + 0.5*A**2/u**2 - (L/beta)*u
    
    if lhs>rhs:
        while True:
            u *= 2
            lhs = A/u + 0.5*A**2/u**2 - (L/beta)*u
            if lhs < rhs:
                u_lb = u/2
                u_ub = u
                break
    else:
        while True:
            u /= 2
            lhs = A/u + 0.5*A**2/u**2 - (L/beta)*u
            if lhs > rhs:
                u_lb = u
                u_ub = u*2
                break
                
    while u_ub - u_lb>1e-13:
        
        u = 0.5*(u_lb + u_ub)
        lhs = A/u + 0.5*A**2/u**2 - (L/beta)*u
        if lhs < rhs:
            u_ub = u
        else:
            u_lb = u
            
    
    return u+A






def FW_subproblem(r, delta, g, c, x0, R_new, u_start, eta_new, iter_max_inner):
    
    u = u_start
    grad = g + c*(u-x0)
    obj = g.dot(u) + (c/2)*np.linalg.norm(u-x0)**2
    
    for i in range(iter_max_inner):
        grad = g + c*(u-x0)
        obj = g.dot(u) + (c/2)*np.linalg.norm(u-x0)**2
        s = linear_oracle(r, delta, x0, R_new, grad)
        FW_gap = -grad.dot(s-u)
        if FW_gap < eta_new:
            print("FW iters=", i+1)
            break
#         al_star = 2/(i+3) 
        al_star = -(g + c*(u-x0)).dot(s-u)/(c*np.linalg.norm(s-u)**2)
        al = min(max(al_star,0), 1)
        u = u + al*(s-u)

        
    return u



def linear_oracle(r, delta, x0, R_new, grad):
    
    n = x0.shape[0]
    
    e = np.ones((n,))
    if r == 1:
        H = sparse.diags([e, -e], [0, 1], shape=(n - 1, n))
    else:
        if r == 2:
            H = sparse.diags([e, -2 * e, e], [0, 1, 2], shape=(n - 2, n))
            
            
    ## Solve with cvxpy
    start = time.time()
    uu = cp.Variable(n)
    constraints = [cp.norm(H * uu, 1) <= delta, cp.norm(uu-x0, 2) <= R_new]
    obj = cp.Minimize(grad.T@uu)
    prob = cp.Problem(obj, constraints)
    optval = prob.solve(solver='MOSEK', verbose=False)

    return uu.value








def acc_projection_free(mat_A, b, r, relative_sigma, delta, cvxpy_optval, iter_max = 1000, iter_max_inner = 10000):
    
    N, n = mat_A.shape
    x0 = np.zeros(n)
    x = np.zeros(n)
    y = np.zeros(n)
    # _x = np.zeros(n)
    tau = np.sqrt(3) #
    R = 1 #
    A = 0
    bar_eta = 0
    beta = 1 #
    p = 218
    B = p + 1

    L = np.linalg.norm(mat_A, 2)**2


    g = np.zeros(n)

    st = time.time()
    time_hist = np.zeros(iter_max)
    gap_hist = np.zeros(iter_max)

    for k in range(iter_max):

        print("--------------------")
        feas = np.linalg.norm(np.diff(x),1)
        print("feas=", feas)
        obj = 0.5*np.linalg.norm(mat_A@x - b)**2
        gap = (obj-cvxpy_optval)/cvxpy_optval
        print("gap = ", (obj-cvxpy_optval)/cvxpy_optval)
        time_hist[k] = time.time() - st
        print("time_hist[k] = ", time_hist[k])
        if gap < 1e-2:
            time_hist = time_hist[0:k]
            gap_hist = gap_hist[0:k]
            break

        if k == 0:
            A_new = beta/(2*L)
            _x = x
            R_new = R
        else:
            A_new = solve_A_new(A, L, beta, tau)
            _x = (A/A_new)*y + (1-A/A_new)*x
            R_new = (2*tau)/(tau-1)*np.linalg.norm(x0 - _x) + (tau+1)/(tau-1)*np.sqrt(2*bar_eta/beta)

        beta_new = k+2
        eta_new = R_new**2/B
        c = beta_new
        g = g + (A_new-A)* (mat_A.T@(mat_A.dot(_x)-b))
        u_start = x0
        x_new = FW_subproblem(r, delta, g/A_new, c/A_new, x0, R_new, u_start, eta_new/A_new, iter_max_inner)
        bar_eta_new = bar_eta + eta_new
        y_new = (A/A_new)*y + (1-A/A_new)*x_new

        x, y = x_new, y_new
        A, beta, eta, bar_eta, R = A_new, beta_new, eta_new, bar_eta_new, R_new
    
    return time_hist, gap_hist
    

    
    

