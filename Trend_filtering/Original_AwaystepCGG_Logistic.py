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


def Original_AwaystepCGG_Logistic(r, A, delta, itermax = 2000):
    ################################################################
    # Use the Awaystep-CGG method for the problem
    #
    # minimize \sum_{i=1}^N log(1 + exp(a_i^T x))
    # s.t. \|D^(r) x\|_1 <= delta
    #
    # A = [a_1^T; a_2^T; ... ; a_N^T] \in R^{ N x n }
    # D^(r) is the r-order discrete difference operator
    #
    # Parameters:
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

    ## Set the initial point x00
    inv_lin = range(n - 1, -1, -1)
    u_initial = np.zeros((n - r, 1))
    u_initial[0] = delta
    y_initial = np.zeros((n, 1))
    y_initial[0:n - r] = u_initial
    y_initial[n - r:n] = -D @ u_initial
    tmp = y_initial[inv_lin]
    for i in range(r):
        tmp = np.cumsum(tmp, axis=0)
    x00 = tmp[inv_lin]

    # The main algorithm
    time_vec = np.zeros((itermax, 1))
    obj_val = np.zeros((itermax, 1))
    start = time.time()
    x = x00
    Ax = A @ x
    AF = A @ F

    ## The table of active verteices, 1: active
    ver_set = np.zeros((2, n - r))
    al = np.zeros((2, n - r))
    ver_set[0, 0] = 1
    al[0, 0] = 1

    for k in range(itermax):

        obj_val[k] = np.sum(np.log(1 + np.exp(Ax)))

        # Take the gradient step in unbounded direction
        g = np.sum(A.T / np.reshape(1 + np.exp(-Ax), (N,)), axis=1)
        grad = np.reshape(g, (n, 1))
        Ft_grad = F.T @ grad
        Pt_grad = F @ Ft_grad
        APt_grad = AF @ Ft_grad
        y = x - (1 / L_or) * Pt_grad
        Ay = Ax - (1 / L_or) * APt_grad
        Pt_y = F.T @ y
        Pt_y = F @ Pt_y

        ####### Computing away step and FW step
        c = grad
        tilde_c = c
        for i in range(r):
            tilde_c = np.cumsum(tilde_c, axis=0)
        cc = tilde_c[0:n - r] - D.T @ tilde_c[n - r:n]
        cc = np.reshape(cc, (n - r,))

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
            u_away = np.zeros((n - r, 1))
            u_away[u_away_index] = - delta
        else:
            u_away_index = flat_pivot_index
            u_away_setindex = (0, u_away_index)
            u_away = np.zeros((n - r, 1))
            u_away[u_away_index] = delta
        y_away = np.zeros((n, 1))
        y_away[0:n - r] = u_away
        y_away[n - r:n] = -D @ u_away
        tmp = y_away[inv_lin]
        for i in range(r):
            tmp = np.cumsum(tmp, axis=0)
        x_away = tmp[inv_lin]

        ## FW step
        u_FW = np.zeros((n - r, 1))
        u_FW_index = np.argmax(np.abs(cc))
        u_FW[u_FW_index] = -np.sign(cc[u_FW_index]) * delta
        y_FW = np.zeros((n, 1))
        y_FW[0:n - r] = u_FW
        y_FW[n - r:n] = -D @ u_FW
        tmp = y_FW[inv_lin]
        for i in range(r):
            tmp = np.cumsum(tmp, axis=0)
        x_FW = tmp[inv_lin]

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
            if alpha < 1 - 1e-8:
                step_max = alpha / (1 - alpha)
            else:
                step_max = 1e8

        Ad = A @ d
        step = logistic_linesearch(Ay, Ad, step_max)
        x = y + step * d
        Ax = Ay + step * Ad

        ### Update active vertex and weights
        if flag == 1:  # FW step
            if step == 1:
                ver_set = np.zeros((2, n - r))
                al = np.zeros((2, n - r))
                if u_FW[u_FW_index] > 0:
                    ver_set[0, u_FW_index] = 1
                    al[0, u_FW_index] = 1
                else:
                    ver_set[1, u_FW_index] = 1
                    al[1, u_FW_index] = 1
            else:
                if u_FW[u_FW_index] > 0:
                    ver_set[0, u_FW_index] = 1
                    al = (1 - step) * al
                    al[0, u_FW_index] = al[0, u_FW_index] + step
                else:
                    ver_set[1, u_FW_index] = 1
                    al = (1 - step) * al
                    al[1, u_FW_index] = al[1, u_FW_index] + step
        else:  # away step
            if np.abs(step - step_max) < 1e-9:
                ver_set[u_away_setindex] = 0
                al[u_away_setindex] = 0
                al = (1 + step) * al
            else:
                al = (1 + step) * al
                al[u_away_setindex] = al[u_away_setindex] - step
        time_vec[k] = time.time() - start

    return time_vec, obj_val

