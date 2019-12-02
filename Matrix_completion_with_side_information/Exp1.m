% rmpath sloan/matlab/2017b/cvx/2.1
% addpath cvx
% addpath scs-matlab
cvx_setup;


Data_table = zeros(12, 6);

m1 = 300;
n1 = 300;
m2 = 300;
n2 = 300;

snr1 = 10;
snr2 = 3;
snr3 = 1;

nnzr1 = 0.1;
nnzr2 = 0.3;
nnzr3 = 0.5;

tol0 = 1e-2;
tol1 = 1e-4;
tol2 = 1e-4;

relative_delta1 = 0.8;
relative_delta2 = 1.2;

r = 5;
r1 = 5;

itermax = 10000;

% MCside_func(m, n, r, r1, nnzr, snr, rel_delta, tol , itermax )



[a1,a2,a3,a4,a5,a6] = MCside_func(m2, n2, r, r1, nnzr1,  snr2, relative_delta1, tol1, itermax);
Data_table(1,:) = [a1,a2,a3,a4,a5,a6];

% [a1,a2,a3,a4,a5,a6] = MCside_func(m2, n2, r, r1, nnzr2,  snr2, relative_delta1, tol1, itermax);
% Data_table(2,:) = [a1,a2,a3,a4,a5,a6];
% 
% [a1,a2,a3,a4,a5,a6] = MCside_func(m2, n2, r, r1, nnzr3,  snr2, relative_delta1, tol1, itermax);
% Data_table(3,:) = [a1,a2,a3,a4,a5,a6];
% 
% [a1,a2,a3,a4,a5,a6] = MCside_func(m2, n2, r, r1, nnzr1,  snr3, relative_delta1, tol1, itermax);
% Data_table(4,:) = [a1,a2,a3,a4,a5,a6];
% 
% [a1,a2,a3,a4,a5,a6] = MCside_func(m2, n2, r, r1, nnzr2,  snr3, relative_delta1, tol1, itermax);
% Data_table(5,:) = [a1,a2,a3,a4,a5,a6];
% 
% [a1,a2,a3,a4,a5,a6] = MCside_func(m2, n2, r, r1, nnzr3,  snr3, relative_delta1, tol1, itermax);
% Data_table(6,:) = [a1,a2,a3,a4,a5,a6];
% 
% 
% 
% 
% [a1,a2,a3,a4,a5,a6] = MCside_func(m2, n2, r, r1, nnzr1,  snr2, relative_delta2, tol1, itermax);
% Data_table(7,:) = [a1,a2,a3,a4,a5,a6];
% 
% [a1,a2,a3,a4,a5,a6] = MCside_func(m2, n2, r, r1, nnzr2,  snr2, relative_delta2, tol1, itermax);
% Data_table(8,:) = [a1,a2,a3,a4,a5,a6];
% 
% [a1,a2,a3,a4,a5,a6] = MCside_func(m2, n2, r, r1, nnzr3,  snr2, relative_delta2, tol1, itermax);
% Data_table(9,:) = [a1,a2,a3,a4,a5,a6];
% 
% [a1,a2,a3,a4,a5,a6] = MCside_func(m2, n2, r, r1, nnzr1,  snr3, relative_delta2, tol1, itermax);
% Data_table(10,:) = [a1,a2,a3,a4,a5,a6];
% 
% [a1,a2,a3,a4,a5,a6] = MCside_func(m2, n2, r, r1, nnzr2,  snr3, relative_delta2, tol1, itermax);
% Data_table(11,:) = [a1,a2,a3,a4,a5,a6];
% 
% [a1,a2,a3,a4,a5,a6] = MCside_func(m2, n2, r, r1, nnzr3,  snr3, relative_delta2, tol1, itermax);
% Data_table(12,:) = [a1,a2,a3,a4,a5,a6];



% row_index = {
%    sprintf( 'nnzr = %.1f, snr = %.1f, relative delta = %.1f', nnzr1, snr2, relative_delta1 ),
%    sprintf( 'nnzr = %.1f, snr = %.1f, relative delta = %.1f', nnzr2, snr2, relative_delta1 ),
%    sprintf( 'nnzr = %.1f, snr = %.1f, relative delta = %.1f', nnzr3, snr2, relative_delta1 ),
%    sprintf( 'nnzr = %.1f, snr = %.1f, relative delta = %.1f', nnzr1, snr3, relative_delta1 ),
%    sprintf( 'nnzr = %.1f, snr = %.1f, relative delta = %.1f', nnzr2, snr3, relative_delta1 ),
%    sprintf( 'nnzr = %.1f, snr = %.1f, relative delta = %.1f', nnzr3, snr3, relative_delta1 ),
%    sprintf( 'nnzr = %.1f, snr = %.1f, relative delta = %.1f', nnzr1, snr2, relative_delta2 ),
%    sprintf( 'nnzr = %.1f, snr = %.1f, relative delta = %.1f', nnzr2, snr2, relative_delta2 ),
%    sprintf( 'nnzr = %.1f, snr = %.1f, relative delta = %.1f', nnzr3, snr2, relative_delta2 ),
%    sprintf( 'nnzr = %.1f, snr = %.1f, relative delta = %.1f', nnzr1, snr3, relative_delta2 ),
%    sprintf( 'nnzr = %.1f, snr = %.1f, relative delta = %.1f', nnzr2, snr3, relative_delta2 ),
%    sprintf( 'nnzr = %.1f, snr = %.1f, relative delta = %.1f', nnzr3, snr3, relative_delta2 ) 
% };

row_index = {
   sprintf( '(%.1f, %.1f, %.1f)', nnzr1, snr2, relative_delta1 ),
   sprintf( '(%.1f, %.1f, %.1f)', nnzr2, snr2, relative_delta1 ),
   sprintf( '(%.1f, %.1f, %.1f)', nnzr3, snr2, relative_delta1 ),
   sprintf( '(%.1f, %.1f, %.1f)', nnzr1, snr3, relative_delta1 ),
   sprintf( '(%.1f, %.1f, %.1f)', nnzr2, snr3, relative_delta1 ),
   sprintf( '(%.1f, %.1f, %.1f)', nnzr3, snr3, relative_delta1 ),
   sprintf( '(%.1f, %.1f, %.1f)', nnzr1, snr2, relative_delta2 ),
   sprintf( '(%.1f, %.1f, %.1f)', nnzr2, snr2, relative_delta2 ),
   sprintf( '(%.1f, %.1f, %.1f)', nnzr3, snr2, relative_delta2 ),
   sprintf( '(%.1f, %.1f, %.1f)', nnzr1, snr3, relative_delta2 ),
   sprintf( '(%.1f, %.1f, %.1f)', nnzr2, snr3, relative_delta2 ),
   sprintf( '(%.1f, %.1f, %.1f)', nnzr3, snr3, relative_delta2 ) 
};

col_index = {'scs time', 'scs training error', 'scs test error', 'CGG time', 'CGG training error', 'CGG test error'};

top_left = '(nnzr, snr, relative delta)';
label = 'MC-side-information';
caption = sprintf('Experiment on a problem with m = %d, n = %d, r = %d, r1 = %d, tol = %.1e. The CGG time takes value -1 means it doesnot reach the %.1e tolerance within %d iterations.' , m2, n2, r, r1, tol1, tol1, itermax);

Latex_table(Data_table, row_index, col_index, top_left, label, caption, 'MyTex/my_tex4.tex');
