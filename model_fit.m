function x = model_fit(prox_l, prox_r, A, b, rho)
% minimize l(A*x + b) + r(x)
% prox_l(y0, rho) = argmin l(y) + rho/2*norm(y-y0)^2
% prox_r(x0, rho) = argmin r(x) + rho/2*norm(x-x0)^2


[m,n] = size(A);

x2 = zeros(n,1);
ux = zeros(n,1);

y2 = ones(m,1);
uy = zeros(m,1);

for i = 1:1000
    
    % project onto A*x1 + b == y1
    x0 = x2 - ux;
    y0 = y2 - uy;
    x1 = (eye(n) + A'*A)\(A'*y0 - A'*b + x0);
    y1 = A*x1 + b;
    
    
    % prox_l
    y0 = y1 + uy;
    y2 = prox_l(y0, rho);
    
    % prox_r
    x0 = x1 + ux;
    x2 = prox_r(x0, rho);
    
    ux = ux + x1 - x2;
    uy = uy + y1 - y2;
    
end

x = x2;