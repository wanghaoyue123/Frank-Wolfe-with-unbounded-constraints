rmpath sloan/matlab/2017b/cvx/2.1
addpath cvx
addpath scs-matlab



cvx_setup;



m = 700
n = 700
r = 5
r1 = 5



rel_delta = 0.8
snr = 3
nnzr = 0.1


tol = 1e-3
itermax = 20000


MCside_exp(m, n, r, r1, nnzr, snr, rel_delta, tol, itermax );