% lasso example
% minimize norm(A*x-b,2) + gamma*norm(x,1)

m = 30;
n = 100;
k = 10;
gamma = 2;

rng(0)
A = randn(m,n);
x = randn(n,1);
[~,I] = sort(x);
x = 0*x;
x(I(1:k),1) = randn(k,1);

b = A*x;
xtrue = x;

% minimize norm(A*x-b,2) + norm(x,1)
rho = 1;
scaled_l1 = @(x0, rho) prox_l1(x0,rho/gamma);
x = model_fit(@prox_l2, scaled_l1, A, -b, rho);


% compare with cvx
cvx_begin
    variable xcvx(n)
    minimize( norm(A*xcvx-b,2) + gamma*norm(xcvx,1) )
cvx_end

[xtrue, x, xcvx]