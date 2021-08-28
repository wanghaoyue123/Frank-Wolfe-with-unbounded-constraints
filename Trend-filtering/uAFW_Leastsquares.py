import numpy as np
import time




def uAFW_Leastsquares(r, A, b, delta, itermax = 2000, cache_length = 1000):
    ################################################################
    # Use the uAFW method for the problem
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
    # itermax: The maximum number of iterations.
    #
    # cache_length: A parameter which is only useful when strategy 2 is taken (in the case N << n).
    #               It is the maximum number of vectors that can be kept by the cache.
    #               
    #
    ################################################################
    
    time_vec = np.zeros((itermax, ))
    obj_val = np.zeros((itermax, ))
    G = np.zeros((itermax, ))
    H = np.zeros((itermax, ))
    N = A.shape[0]
    n = A.shape[1]
    L = 1
    start = time.time()
#     L_or = 2 * np.linalg.norm(A, 2) ** 2
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
    
    Z_sp = np.zeros((itermax, n-r))
    
    


    # The main algorithm
    if strategy == 1:
        
#         start = time.time()
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

        ver_set = np.zeros((2, n - r))
        al = np.zeros((2, n - r))
        ver_set[0, 0] = 1
        al[0, 0] = 1

        for k in range(itermax):
            Ax_b = Ax - b
            obj_val[k] = 0.5*np.linalg.norm(Ax_b)**2
            x_ = np.reshape(x, n)
            if r==1:
                Z_sp[k,:] = (np.abs(np.diff(x_))>1e-7)*1
            if r==2:
                Z_sp[k,:] = (np.abs(np.diff(np.diff(x_)))>1e-7)*1


            ## take gradient step in the unbounded direction
            grad = ATAx - ATb
            Ft_grad = F.T @ grad
            Pt_grad = F @ Ft_grad
            APt_grad = AF @ Ft_grad
            ATAPt_grad = ATAF @ Ft_grad

            while True:
                y = x - (1 / L) * Pt_grad
                Ay = Ax - (1 / L) * APt_grad
                ATAy = ATAx - (1 / L) * ATAPt_grad
                Ay_b = Ay - b
                if (1-1e-10)*0.5*np.linalg.norm(Ay_b)**2 <= 0.5*np.linalg.norm(Ax_b)**2 + Ax_b.T@(Ay-Ax) + (L/2)*np.linalg.norm(y-x)**2:
                    break
                L = L*2
                print("L=", L)
            
            grad = ATAy - ATb
            Pt_grad = F @ (F.T@grad)
            H[k] = np.linalg.norm(Pt_grad)
            
            Ft_y = F.T @ y
            Pt_y = F @ Ft_y
            APt_y = AF @ Ft_y
            ATAPt_y = ATAF @ Ft_y

            ####### Computing away step and FW step
            tilde_c = grad
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
                x_away = UrID[:, u_away_index] * (-delta)
                x_away = np.reshape(x_away, (n, 1))
                Ax_away = AUrID[:, u_away_index] * (-delta)
                Ax_away = np.reshape(Ax_away, (N, 1))
                ATAx_away = ATAUrID[:, u_away_index] * (-delta)
                ATAx_away = np.reshape(ATAx_away, (n, 1))
            else:
                u_away_index = flat_pivot_index
                u_away_setindex = (0, u_away_index)
                x_away = UrID[:, u_away_index] * (delta)
                x_away = np.reshape(x_away, (n, 1))
                Ax_away = AUrID[:, u_away_index] * (delta)
                Ax_away = np.reshape(Ax_away, (N, 1))
                ATAx_away = ATAUrID[:, u_away_index] * (delta)
                ATAx_away = np.reshape(ATAx_away, (n, 1))

            ## FW step
            u_FW = np.zeros((n - r, 1))
            u_FW_index = np.argmax(np.abs(cc))
            u_FW[u_FW_index] = -np.sign(cc[u_FW_index]) * delta

            sgn = -np.sign(cc[u_FW_index])
            x_FW = UrID[:, u_FW_index] * delta * sgn
            x_FW = np.reshape(x_FW, (n, 1))
            Ax_FW = AUrID[:, u_FW_index] * delta * sgn
            Ax_FW = np.reshape(Ax_FW, (N, 1))
            ATAx_FW = ATAUrID[:, u_FW_index] * delta * sgn
            ATAx_FW = np.reshape(ATAx_FW, (n, 1))

            ## take the step
            d_away = (y - Pt_y) - x_away
            d_FW = x_FW - (y - Pt_y)
            if grad.T @ d_FW < grad.T @ d_away:
                flag = 1
                d = d_FW
                Ad = Ax_FW - (Ay - APt_y)
                ATAd = ATAx_FW - (ATAy - ATAPt_y)
                step_max = 1
            else:
                flag = 0
                d = d_away
                Ad = (Ay - APt_y) - Ax_away
                ATAd = (ATAy - ATAPt_y) - ATAx_away
                alpha = al.ravel()[flat_pivot_index]
                if alpha < 1 - 1e-10:
                    step_max = alpha / (1 - alpha)
                else:
                    step_max = 1e10

            t1 = -Ay_b.T @ Ad
            t2 = np.linalg.norm(Ad) ** 2
            step = max(min(t1 / t2, step_max), 0)
            # step = min(2/(k+2), step_max)
            x = y + step * d
            Ax = Ay + step * Ad
            ATAx = ATAy + step * ATAd
            
            G[k] = -(grad.T @ d_FW)

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

    else: # strategy == 2

        x = x00
        Ax = A @ x
        ATAx = A.T @ Ax
        AF = A @ F
        AUr = A
        for i in range(r):
            AUr = np.cumsum(AUr, axis=1)
        AUrID = AUr[:, 0:n - r] - AUr[:, n - r:n] @ D
        ATb = A.T @ b
        ATAF = A.T @ AF

        ## The table of active verteices, 1: active; 0: nonactive
        ver_set = np.zeros((2, n - r))
        al = np.zeros((2, n - r))
        ver_set[0, 0] = 1
        al[0, 0] = 1

        ## Cache for the visited vertices
        index_set = np.zeros((cache_length,))
        cache_vectors = np.zeros((n, cache_length))
        cache_Avectors = np.zeros((N, cache_length))
        cache_ATAvectors = np.zeros((n, cache_length))
        index_num = 1
        index_set[0] = 0
        cache_vectors[:, 0] = np.reshape(x, (n,))
        cache_Avectors[:, 0] = np.reshape(Ax, (N,))
        cache_ATAvectors[:, 0] = np.reshape(ATAx, (n,))

        for k in range(itermax):
            Ax_b = Ax - b
            obj_val[k] = 0.5*np.linalg.norm(Ax_b)**2
            x_ = np.reshape(x, n)
            if r==1:
                Z_sp[k,:] = (np.abs(np.diff(x_))>1e-7)*1
            if r==2:
                Z_sp[k,:] = (np.abs(np.diff(np.diff(x_)))>1e-7)*1

            ## take gradient step in the unbounded direction
            grad = ATAx - ATb
            Ft_grad = F.T @ grad
            Pt_grad = F @ Ft_grad
            APt_grad = AF @ Ft_grad
            ATAPt_grad = ATAF @ Ft_grad

            while True:
                y = x - (1 / L) * Pt_grad
                Ay = Ax - (1 / L) * APt_grad
                ATAy = ATAx - (1 / L) * ATAPt_grad
                Ay_b = Ay - b
                if (1-1e-10)*0.5*np.linalg.norm(Ay_b)**2 <= 0.5*np.linalg.norm(Ax_b)**2 + Ax_b.T@(Ay-Ax) + (L/2)*np.linalg.norm(y-x)**2:
                    break
                L = L*2
            
            grad = ATAy - ATb
            Pt_grad = F @ (F.T@grad)
            H[k] = np.linalg.norm(Pt_grad)
            
            Ft_y = F.T @ y
            Pt_y = F @ Ft_y
            APt_y = AF @ Ft_y
            ATAPt_y = ATAF @ Ft_y

            ####### Computing away step and FW step
            tilde_c = grad
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
            index_set_active = index_set[0:index_num]
            if flat_pivot_index > n - r - 1:
                u_away_index = flat_pivot_index - (n - r)
                u_away_setindex = (1, u_away_index)

                check_index = np.abs(u_away_index - index_set_active) < 0.1
                position_in_cache = np.where(check_index > 0.5)
                # position_in_cache = position_in_cache[0]
                x_away = - cache_vectors[:, position_in_cache]
                x_away = np.reshape(x_away, (n, 1))
                Ax_away = - cache_Avectors[:, position_in_cache]
                Ax_away = np.reshape(Ax_away, (N, 1))
                ATAx_away = - cache_ATAvectors[:, position_in_cache]
                ATAx_away = np.reshape(ATAx_away, (n, 1))
            else:

                u_away_index = flat_pivot_index
                u_away_setindex = (0, u_away_index)
                check_index = np.abs(u_away_index - index_set_active) < 0.1
                position_in_cache = np.where(check_index > 0.5)
                # position_in_cache = position_in_cache[0]
                x_away = cache_vectors[:, position_in_cache]
                x_away = np.reshape(x_away, (n, 1))
                Ax_away = cache_Avectors[:, position_in_cache]
                Ax_away = np.reshape(Ax_away, (N, 1))
                ATAx_away = cache_ATAvectors[:, position_in_cache]
                ATAx_away = np.reshape(ATAx_away, (n, 1))

            ## FW step
            u_FW = np.zeros((n - r, 1))
            u_FW_index = np.argmax(np.abs(cc))
            u_FW[u_FW_index] = -np.sign(cc[u_FW_index]) * delta
            sgn = -np.sign(cc[u_FW_index])

            check_index = np.abs(u_FW_index - index_set_active) < 0.1
            if np.sum(check_index) > 0.5:  # Already computed
                position_in_cache = np.where(check_index > 0.5)
                # position_in_cache = position_in_cache[0]
                x_FW = sgn * cache_vectors[:, position_in_cache]
                x_FW = np.reshape(x_FW, (n, 1))
                Ax_FW = sgn * cache_Avectors[:, position_in_cache]
                Ax_FW = np.reshape(Ax_FW, (N, 1))
                ATAx_FW = sgn * cache_ATAvectors[:, position_in_cache]
                ATAx_FW = np.reshape(ATAx_FW, (n, 1))


            else:  # haven't been computed

                z = np.zeros((n - r, 1))
                z[u_FW_index] = sgn * delta
                w = - D @ z
                zw = np.zeros((n, 1))
                zw[0:n - r] = z
                zw[n - r:n] = w
                s = zw[inv_lin]
                for j in range(r):
                    s = np.cumsum(s, axis=0)
                s = s[inv_lin]
                x_FW = np.reshape(s, (n, 1))

                Ax_FW = AUrID[:, u_FW_index] * sgn * delta
                Ax_FW = np.reshape(Ax_FW, (N, 1))
                ATAx_FW = A.T @ Ax_FW
                ATAx_FW = np.reshape(ATAx_FW, (n, 1))

                index_set[index_num] = u_FW_index
                cache_vectors[:, index_num] = sgn * np.reshape(x_FW, (n,))
                cache_Avectors[:, index_num] = sgn * np.reshape(Ax_FW, (N,))
                cache_ATAvectors[:, index_num] = sgn * np.reshape(ATAx_FW, (n,))
                index_num = index_num + 1

            ## take the step
            d_away = (y - Pt_y) - x_away
            d_FW = x_FW - (y - Pt_y)
            if grad.T @ d_FW < grad.T @ d_away:
                flag = 1
                d = d_FW
                Ad = Ax_FW - (Ay - APt_y)
                ATAd = ATAx_FW - (ATAy - ATAPt_y)
                step_max = 1
            else:
                flag = 0
                d = d_away
                Ad = (Ay - APt_y) - Ax_away
                ATAd = (ATAy - ATAPt_y) - ATAx_away
                alpha = al.ravel()[flat_pivot_index]
                if alpha < 1 - 1e-10:
                    step_max = alpha / (1 - alpha)
                else:
                    step_max = 1e10

            t1 = -Ay_b.T @ Ad
            t2 = np.linalg.norm(Ad) ** 2
            step = max(min(t1 / t2, step_max), 0)
            # step = min(2/(k+2), step_max)
            x = y + step * d
            Ax = Ay + step * Ad
            ATAx = ATAy + step * ATAd
            
            G[k] = -(grad.T @ d_FW)

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
    
    x= np.reshape(x, (n,))
    
    return x, time_vec, obj_val, G, H, Z_sp






