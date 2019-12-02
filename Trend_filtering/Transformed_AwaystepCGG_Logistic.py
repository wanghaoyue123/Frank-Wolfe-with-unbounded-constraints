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
            if np.abs(mid_grad) < 1e-4:
                t = mid
                break
            if mid_grad>0:
                ub = mid
            else:
                lb = mid
    return t


def Transformed_AwaystepCGG_Logistic(r, A, delta, itermax = 2000):
    ################################################################
    # Original problem:
    #
    # minimize \sum_{i=1}^N log(1 + exp(a_i^T x))
    # s.t. \|D^(r) x\|_1 <= delta
    #
    # where A = [a_1^T; a_2^T; ... ; a_N^T] \in R^{ N x n }
    # D^(r) is the r-order discrete difference operator
    #
    # Transformed problem:
    #
    # minimize \sum_{i=1}^N log(1 + exp(a_i^T U^r x))
    # s.t. |x_1| + ... + |x_{n-r}| <= delta
    #
    # where U = [1, 1, ... 1, 1
    #               1, ... 1, 1
    #                  ... 1, 1
    #                      ....
    #                         1]
    #
    # Parameters:
    #
    # itermax: The maximum number of iterations.
    #
    #
    # Author: Haoyue Wang
    # Email: haoyuew@mit.edu
    # Date: 2019. 11. 24
    # Reference:
    ################################################################
    N = A.shape[0]
    n = A.shape[1]

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

    ## Set the initial point x00
    u_initial = np.zeros((n - r, 1))
    u_initial[0] = delta
    x00 = np.zeros((n, 1))
    x00[0:n - r] = u_initial
    x00[n - r:n] = -D @ u_initial

    ## The main algorithm
    time_vec = np.zeros((itermax, 1))
    obj_val = np.zeros((itermax, 1))
    start = time.time()
    x = x00
    AUr = A
    for i in range(r):
        AUr = np.cumsum(AUr, axis=1)
    L_tr = 0.25 * np.linalg.norm(AUr, 2) ** 2
    AUrx = AUr @ x

    ## The table of active verteices, 1: active; 0: nonactive
    ver_set = np.zeros((2, n - r))
    al = np.zeros((2, n - r))
    ver_set[0, 0] = 1
    al[0, 0] = 1

    for k in range(itermax):
        obj_val[k] = np.sum(np.log(1 + np.exp(AUrx)))

        # Take the gradient step in unbounded direction
        g = np.sum(AUr.T / np.reshape(1 + np.exp(-AUrx), (N,)), axis=1)
        grad = np.reshape(g, (n, 1))
        x[n - r:n] = x[n - r:n] - (1 / L_tr) * grad[n - r:n]
        y = x
        Pt_y = np.zeros((n, 1))
        Pt_y[n - r:n] = y[n - r:n]
        AUry = AUr @ y
        g = np.sum(AUr.T / np.reshape(1 + np.exp(-AUry), (N,)), axis=1)
        grad = np.reshape(g, (n, 1))

        ####### Computing away step and FW step
        tilde_c = grad[0:n - r]
        cc = np.reshape(tilde_c, (n - r,))

        ## away step
        cc_cc = np.zeros((2, n - r))
        cc_cc[0] = cc
        cc_cc[1] = -cc
        active_cc = ver_set * cc_cc
        flat_index = np.flatnonzero(active_cc)
        flat_ac_cc = active_cc.ravel()[flat_index]
        pivot_iin = np.argmax(flat_ac_cc)
        flat_pivot_index = flat_index[pivot_iin]

        if flat_pivot_index > n - r - 1:
            u_away_index = flat_pivot_index - (n - r)
            u_away_setindex = (1, u_away_index)
            x_away = np.zeros((n, 1))
            x_away[u_away_index] = - delta

        else:
            u_away_index = flat_pivot_index
            u_away_setindex = (0, u_away_index)
            x_away = np.zeros((n, 1))
            x_away[u_away_index] = delta

        ## FW step
        x_FW = np.zeros((n, 1))
        x_FW_index = np.argmax(np.abs(cc))
        x_FW[x_FW_index] = -np.sign(cc[x_FW_index]) * delta

        ## take the step
        d_away = (y - Pt_y) - x_away
        d_FW = x_FW - (y - Pt_y)
        if grad.T @ d_FW < grad.T @ d_away:
            flag = 1
            d = d_FW
            step_max = 1
        else:
            flag = 0
            d = d_away
            alpha = al.ravel()[flat_pivot_index]
            if alpha < 1 - 1e-10:
                step_max = alpha / (1 - alpha)
            else:
                step_max = 1e10
        AUrd = AUr @ d
        step = logistic_linesearch(AUry,AUrd,step_max)
        x = y + step * d
        AUrx = AUr @ x

        ### Update active vertex and weights
        if flag == 1:  # FW step
            if step == 1:
                ver_set = np.zeros((2, n - r))
                al = np.zeros((2, n - r))
                if x_FW[x_FW_index] > 0:
                    ver_set[0, x_FW_index] = 1
                    al[0, x_FW_index] = 1
                else:
                    ver_set[1, x_FW_index] = 1
                    al[1, x_FW_index] = 1
            else:
                if x_FW[x_FW_index] > 0:
                    ver_set[0, x_FW_index] = 1
                    al = (1 - step) * al
                    al[0, x_FW_index] = al[0, x_FW_index] + step
                else:
                    ver_set[1, x_FW_index] = 1
                    al = (1 - step) * al
                    al[1, x_FW_index] = al[1, x_FW_index] + step
        else:  # away step
            if np.abs(step - step_max) < 1e-9:
                ver_set[u_away_setindex] = 0
                al[u_away_setindex] = 0
                al = (1 + step) * al
            else:
                al = (1 + step) * al
                al[u_away_setindex] = al[u_away_setindex] - step

        # Record the time
        time_vec[k] = time.time()- start

    return time_vec, obj_val

