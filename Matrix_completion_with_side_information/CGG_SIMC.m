function [time_CGG,  train_error, test_error, time_vec, gap_vec] = CGG_SIMC(P1ZUV, P1, row, col, val, ind, delta, cvxopt, step_size, tol, itermax)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % CGG algorithm for Matrix completion with side information.
    %
    % Input:
    %
    % P1ZUV\in R^{m x n}: P1 * Z + U * V'
    %
    % P1\in R^{m x r1}: the matrix whose columns are an orthogonal basis of ker(P) (equivalently, P = I - P1*P1')
    %
    % row, col, val: the row indices, column indices and values of the
    % observed entries repectively. 
    %
    % ind: the 'vectorized' indices of the observed entries.
    %
    % delta: the parameter contoling the diameter of the constraint set.
    %
    % cvxopt: optimal value computed by the SCS
    %
    % step_size: can take two values: 'simple' or 'linesearch'.
    %
    % tol: the tolerance precision of the relative gap of the solution (using a solution from SCS solver as a benchmark)
    %
    % itermax: maximum number of iterations for CGG.
    %
    %
    %
    % Output:
    %
    % time_CGG: the time required for CGG to reach the target tolerance tol
    %
    % train_error, test_error: the training error and test error of the
    % solution of CGG stopped at the tolerance tol
    %
    % time_vec: the vector recording the passed time in each iteration of
    % CGG (If the algorithm break because reaching the tolerance at iteration k, then for all j >= k+1, time_vec(j) = 0)
    %
    % gap_vec: the vector recording the relative gap in each iteration of
    % CGG (If the algorithm break because reaching the tolerance at iteration k, then for all j >= k+1, gap_vec(j) = 0)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [m,n] = size(P1ZUV);
    L = 1;
    data = val;

    % Set up the storage
    X = zeros(m,n);
    X1 = zeros(m,n);
    X2U = zeros(m,1);
    X2V = zeros(n,1);
    X3U = P1;
    X3V = zeros(n,1);
    w = 1;
    PomP1 = P1(row,:);

    Pom_X = w * X1(ind) + sum(X2U(row,:).*X2V(col,:), 2) + sum( PomP1 .* X3V(col,:), 2);
    QTX = P1' * X;

    obj_val = zeros(itermax,1);
    time_vec = zeros(itermax,1);
    gap_vec = zeros(itermax,1);
    flag = 0;
    tic

    for k = 1:itermax

        % Compute the objective value and check the optimality gap
        Pom_X_data = Pom_X - data;
        obj_val(k) = norm(Pom_X_data);
        gap_vec(k) = (obj_val(k)- cvxopt)/(1+cvxopt);
        if flag == 0 &&  gap_vec(k) <tol
            time_CGG = toc;
            flag = 1;
            break;
        end

        % Take the gradient step in the unbounded subspace
        grads = sparse(row,col,Pom_X_data,m,n);
        QTgrad = P1' * grads;
        PomQQTgrad = sum(PomP1 .* (QTgrad(:,col)'),2);
        Pom_Y = Pom_X - PomQQTgrad;
        QTY = QTX - QTgrad;
        grads = sparse(row,col,Pom_Y - data,m,n);

        % Compute the FW step in the bounded subspace
        [u,sig,v] = svds(@(y,tflag) Afunc(y,tflag,P1,grads),[m,n],1 );
        Pom_d = (-delta) * u(row).* v(col) - Pom_Y + sum(PomP1 .* (QTY(:,col)'),2);

        % Compute the step size
        if step_size == 1
            t1 = (data - Pom_Y)' * Pom_d;
            t2 = norm(Pom_d)^2;
            al =  max(min(t1/t2,1),0);
        elseif step_size == 0
            al = 2/(k+2);
        end

        % Take the step and reorganize the storage
        w_old = w;
        w = w * (1-al);
        X3VT = -(1/L)* QTgrad + X3V' + al* P1'* w_old* X1 + al* P1'*X2U * X2V' ;
        X3V = X3VT';
        X2U = [X2U,u];
        X2V = [(1-al)*X2V, -al * delta * v];

        Pom_X = w * X1(ind) + sum(X2U(row,:).*X2V(col,:), 2) + sum( PomP1 .* X3V(col,:), 2);
        QTX =  QTX + al *(-delta)* (P1'*u)*v' - (1/L)* QTgrad;

        if mod(k,20) == 0
           X1 = w * X1 + X2U * X2V';
           w = 1;
           X2U = zeros(m,1);
           X2V = zeros(n,1);
        end

        % Record the time
        time_vec(k) = toc;
    end
    
    if flag == 0
        time_CGG = -1;
    end
    
    X = w * X1 + X2U * X2V' + X3U * X3V';

    P1ZUVind = P1ZUV(ind);
    test_error = ( norm(P1ZUV - X ,'fro')^2 - norm(P1ZUVind - Pom_X)^2 ) / (  norm(P1ZUV ,'fro')^2 - norm(P1ZUVind)^2 );
    train_error = min(obj_val(1:k))^2 / norm(data)^2;
    
 
end

