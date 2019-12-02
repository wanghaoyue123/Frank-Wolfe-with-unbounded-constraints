import numpy as np
import time

### Transformed CGG for least square
def CGF_Leastsquare(r, A, b, delta, step_size, itermax = 2000, cache_length = 300):
    ################################################################
    #
    # Original problem:
    #
    # minimize \|Ax - b\|_2^2
    # s.t. \|D^(r) x\|_1 <= delta
    #
    # A is an (N times n) real matrix
    # b is a vector with length N
    # D^(r) is the r-order discrete difference operator
    #
    #
    # Transformed problem:
    #
    # minimize \| AU^r x - b \|_2^2
    # s.t. |x_1| + ... + |x_{n-r}| <= delta
    #
    # where U = [1, 1, ... 1, 1
    #               1, ... 1, 1
    #                  ... 1, 1
    #                      ....
    #                         1]
    #
    # CGF directly solve the transformed in two steps:
    #
    # First, fully minimize \| AU^r x - b \|_2^2 in the unbounded directions x_{n-r+1}, ..., x_n.
    # The objective function after being minimized in these directions is still a convex quadratic function of x_1,..., x_{n-r}, which is denoted as g(x_1,...,x_{n-r}).
    #
    # Second, minimize g using the ordinary Frank-Wolfe method.
    #
    #
    #
    # Parameters:
    #
    # step_size: It can take two values: 'simple' or 'linesearch'.
    #           'simple' denotes the simple step size rule 2/(k+2) at iteration k for the Frank-Wolfe step,
    #           'linesearch' denotes line search for the Frank-Wolfe step.
    #
    # itermax: The maximum number of iterations.
    #
    # cache_length: A parameter which is only useful when strategy 2 is taken (in the case N << n).
    #               It is the maximum number of vectors that can be kept by the cache.
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


    ## The main algorithm
    time_vec = np.zeros((itermax, 1))
    obj_val = np.zeros((itermax, 1))
    start = time.time()

    AUr = A
    for i in range(r):
        AUr = np.cumsum(AUr, axis=1)

    # Set the initial point
    u_initial = np.zeros((n - r, 1))
    u_initial[0] = delta
    x00 = np.zeros((n, 1))
    x00[0:n - r] = u_initial
    x00[n - r:n] = -D @ u_initial
    x = x00

    ## Main algorithm
    A1 = AUr[:, 0:n - r]
    A2 = AUr[:, n - r:n]
    Q = np.linalg.qr(A2)
    w = x00[0:n - r]

    for k in range(itermax):

        ww = A1 @ w - b
        obj_val[k] = np.linalg.norm(ww - Q[0] @ (Q[0].T @ ww))
        grad = A1.T @ (ww - Q[0] @ (Q[0].T @ ww))
        grad = 2 * grad

        # Compute the FW step
        w_FW_index = np.argmax(np.abs(grad))
        w_FW = np.zeros((n - r, 1))
        w_FW[w_FW_index] = -np.sign(grad[w_FW_index]) * delta
        d_FW = w_FW - w
        A1d_FW = A1 @ d_FW
        QQTA1d_FW = Q[0] @ (Q[0].T @ A1d_FW)
        if step_size == 'linesearch':
            t1 = -ww.T @ (A1d_FW - QQTA1d_FW)
            t2 = np.linalg.norm(A1d_FW - QQTA1d_FW) ** 2
            step = max(min(t1 / t2, 1), 0)
        else:
            step = 2 / (k + 2)
        w = w + step * d_FW
        time_vec[k] = time.time() - start

    return time_vec, obj_val




