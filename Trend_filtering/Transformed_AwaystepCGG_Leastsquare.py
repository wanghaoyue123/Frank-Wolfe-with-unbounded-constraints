import numpy as np
import time

### Transformed CGG for least square
def Transformed_AwaystepCGG_Leastsquare(r, A, b, delta, itermax = 2000, cache_length = 300):
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

    ## The main algorithm
    if strategy == 1:
        time_vec = np.zeros((itermax, 1))
        obj_val = np.zeros((itermax, 1))
        start = time.time()

        AUr = A
        for i in range(r):
            AUr = np.cumsum(AUr, axis=1)
        AUr_end = np.reshape(AUr[:, n - r:n], (N, r))
        L_tr = np.linalg.norm(AUr, 2) ** 2

        AUrTAUr = AUr.T @ AUr
        AUrTAUr_end = np.reshape(AUrTAUr[:, n - r:n], (n, r))

        # Set the initial point
        u_initial = np.zeros((n - r, 1))
        u_initial[0] = delta
        x00 = np.zeros((n, 1))
        x00[0:n - r] = u_initial
        x00[n - r:n] = -D @ u_initial
        x = x00

        AUrx = AUr @ x
        AUrTAUrx = AUrTAUr @ x
        AUrTb = AUr.T @ b

        ## The table of active verteices, 1: active
        ver_set = np.zeros((2, n - r))
        al = np.zeros((2, n - r))
        ver_set[0, 0] = 1
        al[0, 0] = 1

        for k in range(itermax):

            obj_val[k] = np.linalg.norm(AUrx - b)

            # Take the gradient step in unbounded direction
            grad = AUrTAUrx - AUrTb
            x[n - r:n] = x[n - r:n] - (1 / L_tr) * grad[n - r:n]
            y = x
            grad_end = np.reshape(grad[n - r:n], (r, 1))
            AUry = AUrx - (1 / L_tr) * (AUr_end @ grad_end)
            AUrTAUry = AUrTAUrx - (1 / L_tr) * (AUrTAUr_end @ grad_end)
            Pt_y = np.zeros((n, 1))
            Pt_y[n - r:n] = y[n - r:n]
            Pt_y_end = np.reshape(Pt_y[n - r:n], (r, 1))
            AUrPt_y = AUr_end @ Pt_y_end
            AUrTAUrPt_y = AUrTAUr_end @ Pt_y_end
            grad = AUrTAUry - AUrTb

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
                AUrx_away = AUr[:, u_away_index] * (-delta)
                AUrx_away = np.reshape(AUrx_away, (N,1))
                AUrTAUrx_away = AUrTAUr[:, u_away_index] * (-delta)
                AUrTAUrx_away = np.reshape(AUrTAUrx_away, (n,1))

            else:
                u_away_index = flat_pivot_index
                u_away_setindex = (0, u_away_index)
                x_away = np.zeros((n, 1))
                x_away[u_away_index] = delta
                AUrx_away = AUr[:, u_away_index] * (delta)
                AUrx_away = np.reshape(AUrx_away, (N, 1))
                AUrTAUrx_away = AUrTAUr[:, u_away_index] * (delta)
                AUrTAUrx_away = np.reshape(AUrTAUrx_away, (n, 1))

            ## FW step
            x_FW = np.zeros((n, 1))
            x_FW_index = np.argmax(np.abs(cc))
            sgn = -np.sign(cc[x_FW_index])
            x_FW[x_FW_index] = sgn * delta
            AUrx_FW = AUr[:, x_FW_index] * sgn * delta
            AUrx_FW = np.reshape(AUrx_FW, (N,1))
            AUrTAUrx_FW = AUrTAUr[:, x_FW_index] * sgn * delta
            AUrTAUrx_FW = np.reshape(AUrTAUrx_FW, (n,1))

            ## Decide the moving direction
            d_away = (y - Pt_y) - x_away
            d_FW = x_FW - (y - Pt_y)
            if grad.T @ d_FW < grad.T @ d_away:
                flag = 1
                d = d_FW
                AUrd = AUrx_FW - (AUry - AUrPt_y)
                AUrTAUrd = AUrTAUrx_FW - (AUrTAUry - AUrTAUrPt_y)
                step_max = 1
            else:
                flag = 0
                d = d_away
                AUrd =  (AUry - AUrPt_y) - AUrx_away
                AUrTAUrd = (AUrTAUry - AUrTAUrPt_y) - AUrTAUrx_away
                alpha = al.ravel()[flat_pivot_index]
                if alpha < 1 - 1e-10:
                    step_max = alpha / (1 - alpha)
                else:
                    step_max = 1e10

            # Compute the step size and take the step
            t1 = -(AUry - b).T @ AUrd
            t2 = np.linalg.norm(AUrd) ** 2
            step = max(min(t1 / t2, step_max), 0)
            x = y + step * d
            AUrx = AUry + step * AUrd
            AUrTAUrx = AUrTAUry + step * AUrTAUrd


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
            time_vec[k] = time.time() - start

    else: #strategy == 2
        time_vec = np.zeros((itermax, 1))
        obj_val = np.zeros((itermax, 1))
        start = time.time()

        AUr = A
        for i in range(r):
            AUr = np.cumsum(AUr, axis=1)
        AUr_end = np.reshape(AUr[:, n - r:n], (N, r))
        L_tr = np.linalg.norm(AUr, 2) ** 2

        AUrTAUr_end = AUr.T @ AUr_end

        # Set the initial point
        u_initial = np.zeros((n - r, 1))
        u_initial[0] = delta
        x00 = np.zeros((n, 1))
        x00[0:n - r] = u_initial
        x00[n - r:n] = -D @ u_initial
        x = x00

        AUrx = AUr @ x
        AUrTAUrx = AUr.T @ AUrx
        AUrTb = AUr.T @ b

        ## The table of active verteices, 1: active
        ver_set = np.zeros((2, n - r))
        al = np.zeros((2, n - r))
        ver_set[0, 0] = 1
        al[0, 0] = 1

        ## Set up the cache
        index_set = np.zeros((cache_length,))
        cache_vectors = np.zeros((n, cache_length))
        cache_Avectors = np.zeros((N, cache_length))
        cache_ATAvectors = np.zeros((n, cache_length))
        index_num = 1
        index_set[0] = 0
        cache_vectors[:, 0] = np.reshape(x, (n,))
        cache_Avectors[:, 0] = np.reshape(AUrx, (N,))
        cache_ATAvectors[:, 0] = np.reshape(AUrTAUrx, (n,))

        for k in range(itermax):

            obj_val[k] = np.linalg.norm(AUrx - b)

            # Take the gradient step in unbounded direction
            grad = AUrTAUrx - AUrTb
            x[n - r:n] = x[n - r:n] - (1 / L_tr) * grad[n - r:n]
            y = x
            grad_end = np.reshape(grad[n - r:n], (r, 1))
            AUry = AUrx - (1 / L_tr) * (AUr_end @ grad_end)
            AUrTAUry = AUrTAUrx - (1 / L_tr) * (AUrTAUr_end @ grad_end)
            Pt_y = np.zeros((n, 1))
            Pt_y[n - r:n] = y[n - r:n]
            Pt_y_end = np.reshape(Pt_y[n - r:n], (r, 1))
            AUrPt_y = AUr_end @ Pt_y_end
            AUrTAUrPt_y = AUrTAUr_end @ Pt_y_end
            grad = AUrTAUry - AUrTb

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

            index_set_active = index_set[0:index_num]

            if flat_pivot_index > n - r - 1:
                u_away_index = flat_pivot_index - (n - r)
                u_away_setindex = (1, u_away_index)

                check_index = np.abs(u_away_index - index_set_active) < 0.1
                position_in_cache = np.where(check_index > 0.5)
                x_away = - cache_vectors[:, position_in_cache]
                x_away = np.reshape(x_away, (n, 1))
                AUrx_away = - cache_Avectors[:, position_in_cache]
                AUrx_away = np.reshape(AUrx_away, (N, 1))
                AUrTAUrx_away = - cache_ATAvectors[:, position_in_cache]
                AUrTAUrx_away = np.reshape(AUrTAUrx_away, (n, 1))


            else:
                u_away_index = flat_pivot_index
                u_away_setindex = (0, u_away_index)

                check_index = np.abs(u_away_index - index_set_active) < 0.1
                position_in_cache = np.where(check_index > 0.5)
                x_away =  cache_vectors[:, position_in_cache]
                x_away = np.reshape(x_away, (n, 1))
                AUrx_away =  cache_Avectors[:, position_in_cache]
                AUrx_away = np.reshape(AUrx_away, (N, 1))
                AUrTAUrx_away = cache_ATAvectors[:, position_in_cache]
                AUrTAUrx_away = np.reshape(AUrTAUrx_away, (n, 1))

            ## FW step
            x_FW = np.zeros((n, 1))
            x_FW_index = np.argmax(np.abs(cc))
            sgn = -np.sign(cc[x_FW_index])

            check_index = np.abs(x_FW_index - index_set_active) < 0.1
            if np.sum(check_index) > 0.5:  # Already computed
                position_in_cache = np.where(check_index > 0.5)
                x_FW = sgn * cache_vectors[:, position_in_cache]
                x_FW = np.reshape(x_FW, (n, 1))
                AUrx_FW = sgn * cache_Avectors[:, position_in_cache]
                AUrx_FW = np.reshape(AUrx_FW, (N, 1))
                AUrTAUrx_FW = sgn * cache_ATAvectors[:, position_in_cache]
                AUrTAUrx_FW = np.reshape(AUrTAUrx_FW, (n, 1))

            else: # haven't been computed
                x_FW[x_FW_index] = sgn * delta
                AUrx_FW = AUr[:, x_FW_index] * sgn * delta
                AUrx_FW = np.reshape(AUrx_FW, (N, 1))
                AUrTAUrx_FW = AUr.T @ AUrx_FW
                AUrTAUrx_FW = np.reshape(AUrTAUrx_FW , (n,1))

                index_set[index_num] = x_FW_index
                cache_vectors[:, index_num] = sgn * np.reshape(x_FW, (n,))
                cache_Avectors[:, index_num] = sgn * np.reshape(AUrx_FW, (N,))
                cache_ATAvectors[:, index_num] = sgn * np.reshape(AUrTAUrx_FW, (n,))
                index_num = index_num + 1

            ## take the step
            d_away = (y - Pt_y) - x_away
            d_FW = x_FW - (y - Pt_y)
            if grad.T @ d_FW < grad.T @ d_away:
                flag = 1
                d = d_FW
                AUrd = AUrx_FW - (AUry - AUrPt_y)
                AUrTAUrd = AUrTAUrx_FW - (AUrTAUry - AUrTAUrPt_y)
                step_max = 1
            else:
                flag = 0
                d = d_away
                AUrd = (AUry - AUrPt_y) - AUrx_away
                AUrTAUrd = (AUrTAUry - AUrTAUrPt_y) - AUrTAUrx_away
                alpha = al.ravel()[flat_pivot_index]
                if alpha < 1 - 1e-10:
                    step_max = alpha / (1 - alpha)
                else:
                    step_max = 1e10

            t1 = -(AUry - b).T @ AUrd
            t2 = np.linalg.norm(AUrd) ** 2
            step = max(min(t1 / t2, step_max), 0)
            x = y + step * d
            AUrx = AUry + step * AUrd
            AUrTAUrx = AUrTAUry + step * AUrTAUrd

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
            time_vec[k] = time.time() - start


    return time_vec, obj_val




