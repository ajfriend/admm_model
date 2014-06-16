function x = prox_logistic(x0,rho)
% x = argmin sum(log(1+exp(-x))) + rho/2*norm(x-x0)^2
% solution method: newtons method to find f(x) == 0 for the function
% f(x) = -1/(1+exp(x)) + rho(x-x0)
% f'(x) = exp(x)/(1+exp(x))^2 + rho

% make a good initial guess
x = (1/2 + rho*x0)./(rho+1/10);
ind = x > 5;
ind2 = x < -5;
x(ind) = x0(ind);
x(ind2) = x0(ind2) + 1/rho;

for i=1:5
    f = -1./(1+exp(x)) + rho*(x-x0);
    g = exp(x)./(1+exp(x)).^2 + rho;
    x = x - f./g;
end

end