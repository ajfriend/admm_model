function x = prox_l1(x0,rho)
% x = argmin norm(x,1) + rho/2*norm(x-x0)^2

ind = abs(x0) > 1/rho;
x = 0*x0;
x(ind) = x0(ind) - sign(x0(ind))/rho;