import numpy as np
import time

### Original CGG for least square
def Original_CGG_Leastsquare(r, A, b, delta, step_size, itermax = 2000, cache_length = 300):
    ################################################################
    # Use the CGG method for the problem
    #
    # minimize \|Ax - b\|_2^2
    # s.t. \|D^(r) x\|_1 <= delta
    #
    # A is an (N times n) real matrix
    # b is a vector with length N
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
    # cache_length: A parameter which is only useful when strategy 2 is taken (in the case N << n).
    #               It is the maximum number of vectors that can be kept by the cache.
    #
    # Author: Haoyue Wang
    # Email: haoyuew@mit.edu
    # Date: 2019. 11. 24
    # Reference:
    ################################################################

    N = A.shape[0]
    n = A.shape[1]
    L_or = 2 * np.linalg.norm(A, 2) ** 2
    if (N/n) >= 0.2 and n <= 10000:
        strategy = 1
    else:
        strategy = 2

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
    if strategy == 1:
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

        ATA = A.T @ A
        ATb = A.T @ b
        ATAUr = ATA
        for i in range(r):
            ATAUr = np.cumsum(ATAUr, axis=1)
        ATAUrID = ATAUr[:, 0:n - r] - ATAUr[:, n - r:n] @ D
        ATAF = ATA @ F
        ATAx = ATA @ x

        for k in range(itermax):
            Ax_b = Ax - b
            obj_val[k] = np.linalg.norm(Ax_b)

            ## take gradient step in the unbounded direction
            grad = ATAx - ATb
            Ft_grad = F.T @ grad
            Pt_grad = F @ Ft_grad
            APt_grad = AF @ Ft_grad
            ATAPt_grad = ATAF @ Ft_grad

            y = x - (1 / L_or) * Pt_grad
            Ay = Ax - (1 / L_or) * APt_grad
            ATAy = ATAx - (1 / L_or) * ATAPt_grad
            Ay_b = Ay - b

            grad = ATAy - ATb
            Ft_y = F.T @ y
            Pt_y = F @ Ft_y
            APt_y = AF @ Ft_y
            ATAPt_y = ATAF @ Ft_y

            ## Compute the FW step
            tilde_c = grad
            for i in range(r):
                tilde_c = np.cumsum(tilde_c, axis=0)
            cc = tilde_c[0:n - r] - D.T @ tilde_c[n - r:n]
            cc = np.reshape(cc, (n - r,))

            FW_index = np.argmax(np.abs(cc))
            sgn = -np.sign(cc[FW_index])
            x_FW = UrID[:, FW_index] * sgn * delta
            x_FW = np.reshape(x_FW, (n, 1))
            Ax_FW = AUrID[:, FW_index] * sgn * delta
            Ax_FW = np.reshape(Ax_FW, (N, 1))
            ATAx_FW = ATAUrID[:, FW_index] * sgn * delta
            ATAx_FW = np.reshape(ATAx_FW, (n, 1))

            # Take the step
            d = x_FW - (y - Pt_y)
            Ad = Ax_FW - (Ay - APt_y)
            ATAd = ATAx_FW - (ATAy - ATAPt_y)
            if step_size == 'linesearch':
                t1 = -Ay_b.T @ Ad
                t2 = np.linalg.norm(Ad) ** 2
                step = max(min(t1 / t2, 1), 0)
            else:
                step = 2 / (k + 2)

            x = y + step * d
            Ax = Ay + step * Ad
            ATAx = ATAy + step * ATAd

            # Record the time
            time_vec[k] = time.time() - start

    else: # strategy == 2
        time_vec = np.zeros((itermax, 1))
        obj_val = np.zeros((itermax, 1))
        start = time.time()
        x = x00
        Ax = A @ x
        ATAx = A.T @ Ax
        index_set = np.zeros((cache_length,))
        cache_vectors = np.zeros((n, cache_length))
        cache_Avectors = np.zeros((N, cache_length))
        cache_ATAvectors = np.zeros((n, cache_length))
        index_num = 1
        index_set[0] = 0
        cache_vectors[:, 0] = np.reshape(x, (n,))
        cache_Avectors[:, 0] = np.reshape(Ax, (N,))
        cache_ATAvectors[:, 0] = np.reshape(ATAx, (n,))

        AF = A @ F
        AUr = A
        for i in range(r):
            AUr = np.cumsum(AUr, axis=1)
        AUrID = AUr[:, 0:n - r] - AUr[:, n - r:n] @ D

        ATb = A.T @ b
        ATAF = A.T @ AF

        for k in range(itermax):

            Ax_b = Ax - b
            obj_val[k] = np.linalg.norm(Ax_b)

            ## take gradient step in the unbounded direction
            grad = ATAx - ATb
            Ft_grad = F.T @ grad
            Pt_grad = F @ Ft_grad
            APt_grad = AF @ Ft_grad
            ATAPt_grad = ATAF @ Ft_grad

            y = x - (1 / L_or) * Pt_grad
            Ay = Ax - (1 / L_or) * APt_grad
            ATAy = ATAx - (1 / L_or) * ATAPt_grad
            Ay_b = Ay - b

            grad = ATAy - ATb
            Ft_y = F.T @ y
            Pt_y = F @ Ft_y
            APt_y = AF @ Ft_y
            ATAPt_y = ATAF @ Ft_y

            ## Compute the FW step
            tilde_c = grad
            for i in range(r):
                tilde_c = np.cumsum(tilde_c, axis=0)
            cc = tilde_c[0:n - r] - D.T @ tilde_c[n - r:n]
            cc = np.reshape(cc, (n - r,))

            FW_index = np.argmax(np.abs(cc))
            sgn = -np.sign(cc[FW_index])
            index_set_active = index_set[0: index_num]
            check_index = np.abs(FW_index - index_set_active) < 0.1

            if np.sum(check_index) > 0.5:  # Already computed
                position_in_cache = np.where(check_index > 0.5)
                x_FW = sgn * cache_vectors[:, position_in_cache]
                x_FW = np.reshape(x_FW, (n, 1))
                Ax_FW = sgn * cache_Avectors[:, position_in_cache]
                Ax_FW = np.reshape(Ax_FW, (N, 1))
                ATAx_FW = sgn * cache_ATAvectors[:, position_in_cache]
                ATAx_FW = np.reshape(ATAx_FW, (n, 1))


            else:  # haven't been computed

                z = np.zeros((n - r, 1))
                z[FW_index] = sgn * delta
                w = - D @ z
                zw = np.zeros((n, 1))
                zw[0:n - r] = z
                zw[n - r:n] = w
                s = zw[inv_lin]
                for j in range(r):
                    s = np.cumsum(s, axis=0)
                s = s[inv_lin]
                x_FW = np.reshape(s, (n, 1))
                Ax_FW = AUrID[:, FW_index] * sgn * delta
                Ax_FW = np.reshape(Ax_FW, (N, 1))
                ATAx_FW = A.T @ Ax_FW
                ATAx_FW = np.reshape(ATAx_FW, (n, 1))

                ## save the computed vectors in the cache
                index_set[index_num] = FW_index
                cache_vectors[:, index_num] = sgn * np.reshape(x_FW, (n,))
                cache_Avectors[:, index_num] = sgn * np.reshape(Ax_FW, (N,))
                cache_ATAvectors[:, index_num] = sgn * np.reshape(ATAx_FW, (n,))
                index_num = index_num + 1


            d = x_FW - (y - Pt_y)
            Ad = Ax_FW - (Ay - APt_y)
            ATAd = ATAx_FW - (ATAy - ATAPt_y)
            if step_size == 'linesearch':
                t1 = -Ay_b.T @ Ad
                t2 = np.linalg.norm(Ad) ** 2
                step = max(min(t1 / t2, 1), 0)
            else:
                step = 2 / (k + 2)

            x = y + step * d
            Ax = Ay + step * Ad
            ATAx = ATAy + step * ATAd

            # Record the time
            time_vec[k] = time.time() - start

    return time_vec, obj_val




