


%%%%%% Need to set up cvx to use scs
% rmpath sloan/matlab/2017b/cvx/2.1
% addpath /home/haoyuew/test_proj/To_open_MC/scs-matlab
% addpath /home/haoyuew/test_proj/To_open_MC/cvx
% cvx_setup;
%%%%%%


m = 50
n = 50
r = 5
r1 = 5

rel_delta = 0.8
snr = 1
nnzr = 0.3


tol = 1e-3
itermax = 500
rep = 1
save_path = 'Output/'
MCside_exp(m, n, r, r1, nnzr, snr, rel_delta, tol, itermax, save_path, rep);








