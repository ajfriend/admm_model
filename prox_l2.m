function x = prox_l2(x0,rho)
% x = argmin norm(x) + rho/2*norm(x-x0)^2

if norm(x0) > 1/rho
    x = (1 - 1/(rho*norm(x0)))*x0;
else
    x = 0*x0;
end