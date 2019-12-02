function y = Afun(x,tflag,B,C,n)
if strcmp(tflag,'notransp')
    y = [B*x(n+1:end); C*x(1:n)];
else
    y = [C'*x(n+1:end); B'*x(1:n)];
end