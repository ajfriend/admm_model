% svm

n = 2;
m = 100;

% generate two clusters with centers a fixed distance apart.
% the average of the centers is not the origin
center_dist = 1.5;
center = randn(1,n);
xc1 = randn(1,n);
xc1 = center_dist*xc1/norm(xc1);
xc2 = -xc1;
xc1 = xc1 + center;
xc2 = xc2 + center;

% generate a point cloud around the centers
p1 = repmat(xc1,m,1) + randn(m,n);
p2 = repmat(xc2,m,1) + randn(m,n);


p = [p1;p2]; %locations
q = [ones(m,1);-ones(m,1)]; %labels

figure
hold on
plot(p1(:,1), p1(:,2), 'r.');
plot(p2(:,1), p2(:,2), 'b.');

% stuff the matrix for the SVM
A = [repmat(q,1,n).*p, q];


[m,n] = size(A);
gamma = 1;
rho = 1;
% average the hinge loss by number of examples
my_hinge = @(x0,rho) prox_hinge(x0,rho*length(x0));
% the SVM model only regularizes the first n-1 elements of x
my_reg = @(x0,rho) [prox_l2(x0(1:end-1),rho/gamma); x0(end)];
% call the ADMM solver
x = model_fit(my_hinge, my_reg, A, zeros(m,1), rho);


% plot the SVM result from ADMM
line([-5,5], -x(end)/x(2) - x(1)*[-5,5]/x(2),'Color','r');


% compare the ADMM result with CVX
cvx_begin
    variable xcvx(n)
    minimize( sum(pos(1-A*xcvx))/m + gamma*norm(xcvx(1:end-1)) )
cvx_end

line([-5,5], -xcvx(end)/xcvx(2) - xcvx(1)*[-5,5]/xcvx(2),'Color','g');
axis([-5,5,-5,5])
