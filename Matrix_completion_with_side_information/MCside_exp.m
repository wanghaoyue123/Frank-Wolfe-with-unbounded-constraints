function MCside_exp(m, n, r, r1, nnzr, snr, rel_delta, tol, itermax )
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
    % The time and relative gap information are written into txt file in the directory "MC_output_data"
    % 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    mn = m*n;

    
    for k = 1:1
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

        [time_scs, cvxopt, feas_scs, train_error_scs, test_error_scs] = SCS_SIMC(B_om, Pom, P, delta, norm_val, P1ZUV, ind, tol_acc);
        
        SCS_str1 = sprintf('MC_output_data/delta=%0.2f_snr=%d_nnzr=%0.2f_accurate_num%d.txt', rel_delta, snr, nnzr, k)
        % dlmwrite(SCS_str1, cvxopt);
        
        [time_vec1, gap_vec1 ] = uFW_SIMC(P1ZUV, P1, row, col, val, ind, delta, cvxopt, 0, tol,  itermax);
        time_str1 = sprintf('MC_output_data/delta=%0.2f_snr=%d_nnzr=%0.2f_time_simple_num%d.txt', rel_delta, snr, nnzr, k)
        gap_str1 = sprintf('MC_output_data/delta=%0.2f_snr=%d_nnzr=%0.2f_gap_simple_num%d.txt', rel_delta, snr, nnzr, k)
        dlmwrite(time_str1, time_vec1);
        dlmwrite(gap_str1, gap_vec1);
        
        % [time_vec2, gap_vec2 ] = uFW_SIMC(P1ZUV, P1, row, col, val, ind, delta, cvxopt, 1, tol,  itermax);
        % time_str2 = sprintf('MC_output_data/delta=%0.2f_snr=%d_nnzr=%0.2f_time_linesearch_num%d.txt', rel_delta, snr, nnzr, k)
        % gap_str2 = sprintf('MC_output_data/delta=%0.2f_snr=%d_nnzr=%0.2f_gap_linesearch_num%d.txt', rel_delta, snr, nnzr, k)
        % dlmwrite(time_str2, time_vec2);
        % dlmwrite(gap_str2, gap_vec2);
        
        [time_scs, cvxopt1, feas_scs, train_error_scs1, test_error_scs1] = SCS_SIMC(B_om, Pom, P, delta, norm_val, P1ZUV, ind, tol);
        SCS_str2 = sprintf('MC_output_data/delta=%0.2f_snr=%d_nnzr=%0.2f_SCS_num%d.txt', rel_delta, snr, nnzr, k)
        dlmwrite(SCS_str2, [time_scs; (cvxopt1 - cvxopt)/(1+cvxopt); feas_scs])
    end
    

    
    


end

