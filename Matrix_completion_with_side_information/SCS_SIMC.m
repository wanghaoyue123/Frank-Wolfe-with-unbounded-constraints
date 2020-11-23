function [time_scs, cvxopt, feas_scs,  train_error, test_error] = SCS_SIMC(B_om, P_om, P, delta,  norm_val, P1ZUV, ind, tol)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Solve the matrix completion with side information with SCS
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    [m,n] = size(P1ZUV);
    tic
    cvx_begin
        cvx_solver SCS
    %     cvx_precision low
        cvx_solver_settings('eps',tol)
        variable X(m,n)
        minimize(norm(P_om.*X - B_om,'fro'))
        subject to
            norm_nuc(P*X) <= delta
    cvx_end
    time_scs = toc;

    Pom_X = X(ind);
    P1ZUVind = P1ZUV(ind);
    cvxopt = norm(P_om.* X - B_om,'fro');
    feas_scs = (sum(svd(P*X)))/ delta;

    train_error = cvxopt^2 / norm_val^2;
    test_error = ( norm(P1ZUV - X ,'fro')^2 - norm(P1ZUVind - Pom_X)^2 ) / (  norm(P1ZUV ,'fro')^2 - norm(P1ZUVind)^2 );

end

