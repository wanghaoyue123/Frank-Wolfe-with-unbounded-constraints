function [time_scs_ave, train_error_scs_ave, test_error_scs_ave, time_CGG_ave, train_error_CGG_ave, test_error_CGG_ave] ...
    = MCside_func(m, n, r, r1, nnzr, snr, rel_delta, tol, itermax )
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Matrix completion with side information:
    %
    % Minimize_X     \| P_{Omega}(X - B) \|_2
    %     s.t.      \|P X \|_nuc <= delta
    %
    % where X \in R^{m x n}
    % Omega denotes the set of observed entries, which is randomly picked from
    % all entries.
    %
    % P_{Omega}\in R^{m x m} is the projection onto the observed entries.
    %
    % P\in R^{m x m} is the projection onto the side information subspace.
    %
    % B is a given matrix which is generated by:
    % 
    % B = P1 * Z + U * V' + eps,
    %
    % where P1\in R^{m x r1} is the matrix whose columns are an orthogonal basis of ker(P) (equivalently, P = I - P1*P1'),
    % U \in R^{m x r}, V \in R^{r x n}, Z \in R^{r1 x n} are i.i.d. Gaussian
    % matrices, representing the underlying low rank structure and the side
    % information structure of the data.
    % eps \in R^{m x n} is the noise term, which is also i.i.d. Gaussian.
    % 
    %    
    % Input: 
    %
    % m, n, r, r1: the size parameters of the matrices.
    %
    % nnzr: the ratio of the observed entries, that is, nnzr = (number of
    % observed)/(m*n).
    %
    % snr: the signal-to-noise ratio.
    %
    % rel_delta: the relative value of delta w.r.t. \| U * V' \|_nuc.
    %
    % tol: the tolerance precision of the relative gap of the solution (using a solution from SCS solver as a benchmark)
    %
    % itermax: the maximum number of iterations for CGG
    % 
    %
    %
    %
    % Output:
    %
    % time_scs_ave: the average time (among 3 independent trials) used by SCS to reach the target
    % tolerance tol.
    %
    % train_error_scs_ave: the average training error of the solution by
    % SCS.
    %
    % test_error_scs_ave: the average test error of the solution by SCS.
    %
    % time_CGG_ave: the average time (among 3 independent trials) used by CGG to reach the target
    % tolerance tol.
    %
    % train_error_CGG_ave: the average training error of the solution by
    % CGG.
    %
    % test_error_CGG_ave: the average test error of the solution by CGG.
    % 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    mn = m*n;
    time_scs_ave = 0;
    time_CGG_ave = 0;
    train_error_scs_ave = 0;
    train_error_CGG_ave = 0;
    test_error_scs_ave = 0;
    test_error_CGG_ave = 0;
    repeat = 3;
    fail = 0;
    
    
    for k = 1:3
        % Generate the data
        U = randn(m,r);
        V = randn(n,r);
        Z = randn(r1,n);
        Gauss = randn(m,r1);
        [P1,R] = qr(Gauss);
        P1 = P1(:,1:r1);
        P = eye(m) - P1 * P1';
        eps = randn(m,n);
        eps = (sqrt(r)/snr)* eps;

        % Generate the data matrix B
        B = P1 * Z + U * V' + eps;
        P1ZUV = P1 * Z + U * V';
        PUVT = P * U * V';
        svdPUVT = svd(PUVT);
        true_nuc = sum(svdPUVT);
        delta = rel_delta * true_nuc;

        % Randomly pick the observed entries
        % row, col, val denote the row indices, column indices and values of the
        % observed entries repectively. 
        % ind contains the 'vectorized' indices of the observed entries
        N =  nnzr * mn ;
        ind = randperm(mn, N)';
        row = mod(ind,m);
        row(row == 0) = m;
        col = round( (ind-row)/m +1 );
        val = B(ind);
        norm_val = norm(val);
        len = length(row);

        Pom = sparse(row,col,ones(len,1),m,n);
        Pom = full(Pom);
        B_om = B.* Pom;

        L = 1;
        Q_SIM = P1;
        tol_acc = 1e-5;

        [time_scs, cvxopt, train_error_scs, test_error_scs] = SCS_SIMC(B_om, Pom, P, delta, norm_val, P1ZUV, ind, tol_acc);
        [time_CGG, train_error_CGG, test_error_CGG, time_vec, gap_vec ] = CGG_SIMC(P1ZUV, P1, row, col, val, ind, delta, cvxopt, 0, tol,  itermax);
        
        if time_CGG < 0
            fail = 1;
        else
            time_CGG_ave =  time_CGG_ave + time_CGG;
        end
        
        time_scs_ave = time_scs_ave + time_scs;
        train_error_scs_ave = train_error_scs_ave + train_error_scs;
        train_error_CGG_ave = train_error_CGG_ave + train_error_CGG;
        test_error_scs_ave = test_error_scs_ave + test_error_scs;
        test_error_CGG_ave = test_error_CGG_ave + test_error_CGG;
        
            
    end
    
    
    if fail == 0
        time_CGG_ave = time_CGG_ave / repeat;
    else
        time_CGG_ave  = -1;
    end
    
    time_scs_ave = time_scs_ave / repeat;
    train_error_scs_ave = train_error_scs_ave / repeat;
    train_error_CGG_ave = train_error_CGG_ave / repeat;
    test_error_scs_ave = test_error_scs_ave / repeat;
    test_error_CGG_ave = test_error_CGG_ave / repeat;
    
    


end

