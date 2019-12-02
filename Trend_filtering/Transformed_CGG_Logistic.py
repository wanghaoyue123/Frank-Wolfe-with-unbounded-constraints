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


def Transformed_CGG_Logistic(r, A, delta, step_size, itermax = 2000):
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
    # step_size: It can take two values: 'simple' or 'linesearch'.
    #           'simple' denotes the simple step size rule 2/(k+2) at iteration k for the Frank-Wolfe step,
    #           'linesearch' denotes line search for the Frank-Wolfe step.
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

    # The main algorithm
    time_vec = np.zeros((itermax, 1))
    obj_val = np.zeros((itermax, 1))
    start = time.time()
    x = x00
    AUr = A
    for i in range(r):
        AUr = np.cumsum(AUr, axis=1)
    L_tr = 0.25 * np.linalg.norm(AUr, 2) ** 2

    AUrx = AUr @ x

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

        ## Compute the FW step
        x_FW_index = np.argmax(np.abs(grad[0:n - 1]))
        x_FW = np.zeros((n,1))
        x_FW[x_FW_index] = -np.sign(grad[x_FW_index]) * delta

        ## Take the FW step
        d = x_FW - (y - Pt_y)
        AUrd = AUr @ d
        if step_size == 'linesearch':
            step = logistic_linesearch(AUry,AUrd,1)
        else:
            if step_size == 'simple':
                step = 2 / (k + 2)
        x = y + step * d
        AUrx = AUry + step * AUrd

        # Record the time
        time_vec[k] = time.time()- start

    return time_vec, obj_val

