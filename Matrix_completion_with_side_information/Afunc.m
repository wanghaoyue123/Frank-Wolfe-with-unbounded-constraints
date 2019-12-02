function z = Afunc(y,tflag,Q,G)

if strcmp(tflag,'notransp')
    Gy = G * y;
    z1 = Q * (Q' * Gy);
    z = Gy - z1;
else
    QQy = Q*(Q'*y);
    z = G' * (y - QQy);
end



end

