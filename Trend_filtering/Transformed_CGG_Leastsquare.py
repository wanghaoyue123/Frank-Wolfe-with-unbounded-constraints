import numpy as np
import time

### Transformed CGG for least square
def Transformed_CGG_Leastsquare(r, A, b, delta, step_size, itermax = 2000, cache_length = 300):
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
        AUr_end = np.reshape(AUr[:, n-r:n], (N,r))
        L_tr = np.linalg.norm(AUr, 2)**2

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


        for k in range(itermax):

            AUrx_b = AUrx - b
            obj_val[k] = np.linalg.norm(AUrx_b)

            # Take the gradient step in unbounded direction
            grad = AUrTAUrx - AUrTb
            x[n - r:n] = x[n - r:n] - (1 / L_tr) * grad[n - r:n]
            y = x
            grad_end = np.reshape(grad[n - r:n], (r,1))
            AUry = AUrx - (1 / L_tr) * (AUr_end @ grad_end)
            AUrTAUry = AUrTAUrx - (1 / L_tr) * (AUrTAUr_end @ grad_end)
            Pt_y = np.zeros((n, 1))
            Pt_y[n - r:n] = y[n - r:n]
            Pt_y_end = np.reshape(Pt_y[n - r:n], (r,1) )
            AUrPt_y = AUr_end @ Pt_y_end
            AUrTAUrPt_y = AUrTAUr_end @ Pt_y_end
            grad = AUrTAUry - AUrTb

            ## Compute the FW step
            x_FW_index = np.argmax(np.abs(grad[0:n - r]))
            sgn = -np.sign(grad[x_FW_index])
            x_FW = np.zeros((n, 1))
            x_FW[x_FW_index] = sgn * delta
            AUrx_FW = sgn * delta * np.reshape(AUr[:, x_FW_index], (N,1))
            AUrTAUrx_FW = sgn * delta * np.reshape(AUrTAUr[:, x_FW_index], (n,1))

            ## Linear search and take the FW step
            d = x_FW - (y - Pt_y)
            AUrd = AUrx_FW - (AUry - AUrPt_y)
            AUrTAUrd = AUrTAUrx_FW - (AUrTAUry - AUrTAUrPt_y)
            if step_size == 'linesearch':
                t1 = -(AUry-b).T @ AUrd
                t2 = np.linalg.norm(AUrd) ** 2
                step = max(min(t1 / t2, 1), 0)
            else:
                if step_size == 'simple':
                    step = 2 / (k + 2)
            x = y + step * d
            AUrx = AUry + step * AUrd
            AUrTAUrx = AUrTAUry + step * AUrTAUrd

            # Record the time
            time_vec[k] = time.time() - start

    else: # Strategy == 2
        time_vec = np.zeros((itermax, 1))
        obj_val = np.zeros((itermax, 1))
        start = time.time()

        AUr = A
        for i in range(r):
            AUr = np.cumsum(AUr, axis=1)
        AUr_end = np.reshape(AUr[:, n - r:n], (N, r))
        AUrTAUr_end = AUr.T @ AUr_end
        L_tr = np.linalg.norm(AUr, 2) ** 2

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

            AUrx_b = AUrx - b
            obj_val[k] = np.linalg.norm(AUrx_b)

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



            ## Compute the FW step
            x_FW_index = np.argmax(np.abs(grad[0:n - r]))
            sgn = -np.sign(grad[x_FW_index])
            index_set_active = index_set[0: index_num]
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

                x_FW = np.zeros((n, 1))
                x_FW[x_FW_index] = sgn * delta
                AUrx_FW = sgn * delta * np.reshape(AUr[:, x_FW_index], (N, 1))
                AUrTAUrx_FW = AUr.T @ AUrx_FW

                ## save the computed vectors in the cache
                index_set[index_num] = x_FW_index
                cache_vectors[:, index_num] = sgn * np.reshape(x_FW, (n,))
                cache_Avectors[:, index_num] = sgn * np.reshape(AUrx_FW, (N,))
                cache_ATAvectors[:, index_num] = sgn * np.reshape(AUrTAUrx_FW, (n,))
                index_num = index_num + 1

            ## Linear search and take the FW step
            d = x_FW - (y - Pt_y)
            AUrd = AUrx_FW - (AUry - AUrPt_y)
            AUrTAUrd = AUrTAUrx_FW - (AUrTAUry - AUrTAUrPt_y)
            if step_size == 'linesearch':
                t1 = -(AUry - b).T @ AUrd
                t2 = np.linalg.norm(AUrd) ** 2
                step = max(min(t1 / t2, 1), 0)
            else:
                if step_size == 'simple':
                    step = 2 / (k + 2)
            x = y + step * d
            AUrx = AUry + step * AUrd
            AUrTAUrx = AUrTAUry + step * AUrTAUrd

            # Record the time
            time_vec[k] = time.time() - start


    return time_vec, obj_val




