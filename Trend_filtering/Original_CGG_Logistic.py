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


def Original_CGG_Logistic(r, A, delta, step_size, itermax = 2000):
    ################################################################
    # Use the CGG method for the problem
    #
    # minimize \sum_{i=1}^N log(1 + exp(a_i^T x))
    # s.t. \|D^(r) x\|_1 <= delta
    #
    # A = [a_1^T; a_2^T; ... ; a_N^T] \in R^{ N x n }
    # D^(r) is the r-order discrete difference operator
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
    AUr = A
    for i in range(r):
        AUr = np.cumsum(AUr, axis=1)
    AUrID = AUr[:, 0:n - r] - AUr[:, n - r:n] @ D

    Ur = np.eye(n)
    for i in range(r):
        Ur = np.cumsum(Ur, axis=1)
    UrID = Ur[:, 0:n - r] - Ur[:, n - r:n] @ D

    for k in range(itermax):

        obj_val[k] = np.sum(np.log(1 + np.exp(Ax) ) )

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
                step = 2 / (k + 2)

        x = y + step * d
        Ax = Ay + step * Ad

        # Record the time
        time_vec[k] = time.time()- start

    return time_vec, obj_val

