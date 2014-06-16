# admm_model

## admm iteration
```
minimize    f(x) + g(z)
subject to  x == z
```
has the ADMM iteration

- x^{k+1} := prox_{f/rho}( z^k - u^k )
- z^{k+1} := prox_{g/rho}( x^{k+1} + u^k )
- u^{k+1} := u^k + x^{k+1} - z^{k+1}

## projecting onto A*x + b = y

To project (t0,s0) onto the equality constraints A*t == s, we solve the problem

minimize 1/2*||t-t0||_2^2 + 1/2*||s-s0||_2^2
subject to A*t == s

I write the Lagrangian as 1/2*||t-t0||_2^2 + 1/2*||s-s0||_2^2 + y^'*(A*t - s) to get the optimality conditions:

t - t0 + A'*y == 0
s - s0 - y == 0
A*t == s

We can solve for s or t, based on which of A'*A and A*A' we want to form and invert.

Solving for t gives:

t := (I + A'*A)\(A'*s0 + t0)
s := A*t

Solving for s gives:

s := (I + A*A')\(A*A'*s0 + A*t0)
t := t0 - A'*(s-s0)

## prox operator for logistic loss

