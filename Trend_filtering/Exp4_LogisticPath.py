import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from Data_generation import generate_data_logistic


from CVXPY_Logistic import  CVXPY_Logistic
from Original_CGG_Logistic import  Original_CGG_Logistic
from Original_AwaystepCGG_Logistic import Original_AwaystepCGG_Logistic
from Transformed_CGG_Logistic import Transformed_CGG_Logistic
from Transformed_AwaystepCGG_Logistic import  Transformed_AwaystepCGG_Logistic


import numpy as np
import time

def logistic_linesearch(Ax, Ad, stepmax):
    # f(x) = \sum_{i=1}^N log(1+exp(a_i^T x) )
    # min_t f(x+ td)  for t\in [0,1]
    # right_grad = d.T @ logistic_loss_grad(A, x+d)
    right_grad = np.sum( Ad / (1 + np.exp(-Ax - stepmax * Ad)) )
    if right_grad < 0:
        t =  stepmax
    else:
        lb = 0
        ub = 1
        k = 0
        while 1:
            k = k+1
            mid = (lb+ub)/2
            mid_grad = np.sum( Ad / (1 + np.exp(-Ax - mid * Ad)) )
            # mid_grad = d.T @ logistic_loss_grad(A, x + mid*d)
            if np.abs(mid_grad) < 1e-5:
                t = mid
                break
            if mid_grad>0:
                ub = mid
            else:
                lb = mid
    return t


def CGG_Logistic_warmstart(r, A, delta, step_size, x00, cvxpy_optval, tol, k_plus, itermax = 2000):
    ################################################################
    # Use the CGG method to compute the following problem, with a warmstart initial point.
    #
    # minimize \sum_{i=1}^N log(1 + exp(a_i^T x))
    # s.t. \|D^(r) x\|_1 <= delta
    #
    # A = [a_1^T; a_2^T; ... ; a_N^T] \in R^{ N x n }
    # D^(r) is the r-order discrete difference operator
    #
    # Inputs:
    #
    # step_size: It can take two values: 'simple' or 'linesearch'.
    #           'simple' denotes the simple step size rule 2/(k+2) at iteration k for the Frank-Wolfe step,
    #           'linesearch' denotes line search for the Frank-Wolfe step.
    #
    # x00: The initial point for warm start.
    #
    # cvxpy_optval: The (accurate) optimal value by cvxpy.
    #
    # tol: The target tolerance in relative gap.
    #
    # k_plus: Used for warm start for the step size 2/(k_plus + k)
    #
    # itermax: The maximum number of iterations.
    #
    #
    #
    #
    # Author: Haoyue Wang
    # Email: haoyuew@mit.edu
    # Date: 2019. 11. 24
    # Reference:
    ################################################################
    N = A.shape[0]
    n = A.shape[1]
    L_or = 0.25 * np.linalg.norm(A, 2) ** 2

    ## Construct an orthogonal basis F for the subspace ker(D^(r))
    if r == 1:
        F = (1 / np.sqrt(n)) * np.ones((n, 1))
    else:
        gg = np.ones((n, 1))
        E = np.ones((n, r))
        for i in range(r):
            E[:, i] = np.reshape(gg, (n,))
            gg = np.cumsum(gg, axis=0)
        FF = np.linalg.qr(E, mode='reduced')
        F = FF[0]

    Tmp = F
    for i in range(r):
        Tmp = np.cumsum(Tmp, axis=0)
    BT = Tmp
    B = BT.T
    B1 = B[:, 0:n - r]
    B2 = B[:, n - r:n]
    D = np.linalg.inv(B2) @ B1


    # The main algorithm
    time_vec = np.zeros((itermax, 1))
    obj_val = np.zeros((itermax, 1))
    start = time.time()

    x = x00
    Ax = A @ x
    AF = A @ F
    AUr = A
    for i in range(r):
        AUr = np.cumsum(AUr, axis=1)
    AUrID = AUr[:, 0:n - r] - AUr[:, n - r:n] @ D

    Ur = np.eye(n)
    for i in range(r):
        Ur = np.cumsum(Ur, axis=1)
    UrID = Ur[:, 0:n - r] - Ur[:, n - r:n] @ D

    flag = 0
    for k in range(itermax):

        obj_val[k] = np.sum(np.log(1 + np.exp(Ax) ) )
        gap = (obj_val[k]- cvxpy_optval)/cvxpy_optval

        if gap < tol and flag == 0:
            time_CGG = time_vec[k-1,0]
            x_CGG = x
            stop_k = k
            flag = 1
            break

        # Take the gradient step in unbounded direction
        g = np.sum(A.T / np.reshape(1 + np.exp(-Ax), (N,)), axis=1)
        grad = np.reshape(g, (n, 1))

        Ft_grad = F.T @ grad
        Pt_grad = F @ Ft_grad
        APt_grad = AF @ Ft_grad
        y = x - (1 / L_or) * Pt_grad
        Ay = Ax - (1 / L_or) * APt_grad

        Ft_y = F.T @ y
        Pt_y = F @ Ft_y
        APt_y= AF @ Ft_y

        ## Compute the FW step
        tilde_c = grad
        for i in range(r):
            tilde_c = np.cumsum(tilde_c,axis=0)
        cc = tilde_c[0:n-r] - D.T @ tilde_c[n-r:n]
        cc = np.reshape(cc,(n-r,))

        FW_index = np.argmax(np.abs(cc))
        sgn = -np.sign(cc[FW_index])
        x_FW = UrID[:,FW_index] * sgn * delta
        x_FW = np.reshape(x_FW, (n,1))
        Ax_FW= AUrID[:,FW_index] * sgn * delta
        Ax_FW = np.reshape(Ax_FW, (N, 1))
        #############

        d = x_FW - (y - Pt_y)
        Ad= Ax_FW - Ay + APt_y
        if step_size == 'linesearch':
            step = logistic_linesearch(Ay,Ad,1)
        else:
            if step_size == 'simple':
                step = 2 / (k + 2 + k_plus)

        x = y + step * d
        Ax = Ay + step * Ad

        # Record the time
        time_vec[k] = time.time()- start


    if flag == 0:
        rel_gap = (np.min(obj_val) - cvxpy_optval)/(1+cvxpy_optval)
        Fail_information = ['Failed to reach %.1e for delta = %.2f, relative_gap = %.3e' %(tol, delta, rel_gap)]
        # print('Failed to reach %.1e for delta = %.2f, relative_gap = ', (np.min(obj_val) - cvxpy_optval)/(1+cvxpy_optval))
        x_return = x
        stop_k = itermax
        time_CGG = time_vec[itermax-1]
    else:
        Fail_information = ['delta = %.2f: Success!' %(delta)]
        x_return = x_CGG

    ###########################################3
    # Output:
    #
    # x_return: The solution by CGG
    #
    # time_CGG: The time used by CGG to reach the tol.
    #
    # stop_k: The iterations used by CGG to reach the tol.
    #
    # Fail_information: Record whether CGG reach the tol within maximum iterations or not.
    return [x_return, time_CGG, stop_k, Fail_information]



def CGG_Logistic_Path(r, N, n, sigma, tol, delta_interval, steplength, itermax = 2000):
    ###########################################
    # Compute the solution along a path of delta
    #
    # Input:
    #
    # delta_interval: the minimum and maximum value of delta.
    #
    # steplength: the gap between consecutive values of deltas.
    #
    #
    #
    ###########################################
    delta_min = delta_interval[0]
    delta_max = delta_interval[1]
    num_nodes = int((delta_max - delta_min)/steplength)
    x = np.zeros((n, 1))
    k_plus = 0
    X = np.random.randn(N,n)
    A = generate_data_logistic(X, r, N, n, sigma)

    tm_collector_CGG = np.zeros((num_nodes,))
    tm_collector_cvxpy = np.zeros((num_nodes,))
    delta_collector = np.zeros((num_nodes,))
    k_stop_collector = np.zeros((num_nodes,))
    Fail_information_collector = [0 for i in range(num_nodes)]
    for j in range(num_nodes):
        delta = 0.1 + j * 0.01
        tm_cvxpy, cvxpy_optval = CVXPY_Logistic(r, A, delta)
        [x, tm_CGG, stop_k, Fail_information] = CGG_Logistic_warmstart(r=r, A=A, delta=delta, step_size='simple', x00=x, cvxpy_optval=cvxpy_optval, tol=tol, k_plus=k_plus, itermax = itermax)
        k_plus = min(0.05 * stop_k, 50)

        delta_collector[j] = delta
        tm_collector_CGG[j] = tm_CGG
        tm_collector_cvxpy[j] = tm_cvxpy
        Fail_information_collector[j] = Fail_information
        k_stop_collector[j] = stop_k

    report_table = [delta_collector, tm_collector_CGG, tm_collector_cvxpy, k_stop_collector, Fail_information_collector]
    cvxpy_total_time = np.sum(tm_collector_cvxpy)
    CGG_total_time = np.sum(tm_collector_CGG)

    ###########################################3
    # Output:
    #
    # CGG_total_time: The total time used by CGG to compute the path.
    #
    # cvxpy_total_time: The total time used by cvxpy to compute the path.
    #
    # report_table: A table containing information whether the CGG reach the tol within the maximum iterations or not.
    return CGG_total_time, cvxpy_total_time, report_table





N1 = 200
n1 = 2000
sigma11 = 1 / 10000
sigma12 = 1 / 4000

N2 = 3000
n2 = 300
sigma21 = 1 / 1500
sigma22 = 1 / 600

N3 = 1000
n3 = 1000
sigma31 = 1 / 5000
sigma32 = 1 / 2000

r = 2
itermax = 3000
tol = 1e-4


Data_table = np.zeros((6,2))


# CGG_total_time, cvxpy_total_time, report_table = CGG_Logistic_Path(r=1, N=N1, n=n1, sigma=sigma11, tol=tol, delta_interval=[0.1,2], steplength=0.01, itermax = itermax)
# Data_table[0,:] = [CGG_total_time, cvxpy_total_time]
# CGG_total_time, cvxpy_total_time, report_table = CGG_Logistic_Path(r=1, N=N1, n=n1, sigma=sigma12, tol=tol, delta_interval=[0.1,2], steplength=0.01, itermax = itermax)
# Data_table[1,:] = [CGG_total_time, cvxpy_total_time]
# CGG_total_time, cvxpy_total_time, report_table = CGG_Logistic_Path(r=1, N=N2, n=n2, sigma=sigma21, tol=tol, delta_interval=[0.1,2], steplength=0.01, itermax = itermax)
# Data_table[2,:] = [CGG_total_time, cvxpy_total_time]
# CGG_total_time, cvxpy_total_time, report_table = CGG_Logistic_Path(r=1, N=N2, n=n2, sigma=sigma22, tol=tol, delta_interval=[0.1,2], steplength=0.01, itermax = itermax)
# Data_table[3,:] = [CGG_total_time, cvxpy_total_time]
# CGG_total_time, cvxpy_total_time, report_table = CGG_Logistic_Path(r=1, N=N3, n=n3, sigma=sigma31, tol=tol, delta_interval=[0.1,2], steplength=0.01, itermax = itermax)
# Data_table[4,:] = [CGG_total_time, cvxpy_total_time]
# CGG_total_time, cvxpy_total_time, report_table = CGG_Logistic_Path(r=1, N=N3, n=n3, sigma=sigma32, tol=tol, delta_interval=[0.1,2], steplength=0.01, itermax = itermax)
# Data_table[5,:] = [CGG_total_time, cvxpy_total_time]


CGG_total_time, cvxpy_total_time, report_table = CGG_Logistic_Path(r=2, N=100, n=200, sigma=1/600, tol=tol, delta_interval=[0.1,2], steplength=0.01, itermax = itermax)
# Data_table[0,:] = [CGG_total_time, cvxpy_total_time]
# CGG_total_time, cvxpy_total_time, report_table = CGG_Logistic_Path(r=2, N=N1, n=n1, sigma=sigma12, tol=tol, delta_interval=[0.1,2], steplength=0.01, itermax = itermax)
# Data_table[1,:] = [CGG_total_time, cvxpy_total_time]
# CGG_total_time, cvxpy_total_time, report_table = CGG_Logistic_Path(r=2, N=N2, n=n2, sigma=sigma21, tol=tol, delta_interval=[0.1,2], steplength=0.01, itermax = itermax)
# Data_table[2,:] = [CGG_total_time, cvxpy_total_time]
# CGG_total_time, cvxpy_total_time, report_table = CGG_Logistic_Path(r=2, N=N2, n=n2, sigma=sigma22, tol=tol, delta_interval=[0.1,2], steplength=0.01, itermax = itermax)
# Data_table[3,:] = [CGG_total_time, cvxpy_total_time]
# CGG_total_time, cvxpy_total_time, report_table = CGG_Logistic_Path(r=2, N=N3, n=n3, sigma=sigma31, tol=tol, delta_interval=[0.1,2], steplength=0.01, itermax = itermax)
# Data_table[4,:] = [CGG_total_time, cvxpy_total_time]
# CGG_total_time, cvxpy_total_time, report_table = CGG_Logistic_Path(r=2, N=N3, n=n3, sigma=sigma32, tol=tol, delta_interval=[0.1,2], steplength=0.01, itermax = itermax)
# Data_table[5,:] = [CGG_total_time, cvxpy_total_time]


Data_frame = pd.DataFrame(Data_table )
Data_frame.columns = ('CGG_simple', 'CVXPY')
Data_frame.index = (
# 'r=%d, (N,n)=(%d,%d), sigma=%.3e' %(1,N1,n1,sigma11),
# 'r=%d, (N,n)=(%d,%d), sigma=%.3e' %(1,N1,n1,sigma12),
# 'r=%d, (N,n)=(%d,%d), sigma=%.3e' %(1,N2,n2,sigma21),
# 'r=%d, (N,n)=(%d,%d), sigma=%.3e' %(1,N2,n2,sigma22),
# 'r=%d, (N,n)=(%d,%d), sigma=%.3e' %(1,N3,n3,sigma31),
# 'r=%d, (N,n)=(%d,%d), sigma=%.3e' %(1,N3,n3,sigma32),
'r=%d, (N,n)=(%d,%d), sigma=%.3e' %(2,N1,n1,sigma11),
'r=%d, (N,n)=(%d,%d), sigma=%.3e' %(2,N1,n1,sigma12),
'r=%d, (N,n)=(%d,%d), sigma=%.3e' %(2,N2,n2,sigma21),
'r=%d, (N,n)=(%d,%d), sigma=%.3e' %(2,N2,n2,sigma22),
'r=%d, (N,n)=(%d,%d), sigma=%.3e' %(2,N3,n3,sigma31),
'r=%d, (N,n)=(%d,%d), sigma=%.3e' %(2,N3,n3,sigma32),
)

Table_latex = Data_frame.to_latex(float_format='%.3f')

fo = open('TexFile/Exp4_Logistic_Path.tex', 'a')
fo.write('\n')
fo.write('\\begin{table}[H]\n')
fo.write('\\centering\n')
fo.write('\\label{Table_logistic_path}\n')
fo.write(Table_latex)
# fo.write('\\caption{r = %d, N = %d, n = %d}\n' %(1,N1,n1) )
fo.write('\\caption{Logistic path}\n')
fo.write('\\end{table}\n')
fo.write('\n')
fo.write('\n')
