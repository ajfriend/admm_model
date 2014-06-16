function x = prox_hinge(x0,rho)
% x = argmin sum((1-x)_+) + rho/2*norm(x-x0)^2

x = x0;
ind = x0 < 1;
x(ind) = min(x(ind)+1/rho,1);