# Solving the Bellman Equation with a Simple Univariate Diffusion
## Setup
Take the stochastic process
$$
d x_t = \mu(x_t)dt + \sigma(x_t)d W_t
$$
where $W_t$ is Brownian motion and reflecting barriers at $x \in (\underline{x}, \bar{x})$ where $-\infty < \underline{x} < \bar{x} < \infty$.

The partial differential operator (infinitesimal generator) associated with the stochastic process is
$$
	\mathcal{A} \equiv \mu(x)\partial_x + \frac{\sigma(x)^2}{2}\partial_{xx}
$$

Then, if the payoff in state $x$ is $u(x)$, and payoffs are discounted at rate $\rho$, then the Bellman equation is,
$$
\rho v(x) = u(x) + \mathcal{A}v(x)
$$
With boundary values $v'(\underline{x}) = 0$ and $v'(\bar{x}) = 0$

## Solving the discretized problem
Create a grid on $x$ where $i = 1, \ldots I$ and  and $x_1 = \underline{x} = x_1, x2, \ldots, x_I$, define $v \in \mathbb{R}^I$ such that $v_i = v(x_i)$, and finally $u \in \mathbb{R}^I$ such that $u_i = u(x_i)$.

Let the upwind discretized $\mathcal{A}$, subject to the boundary conditions, be $A$, then the solution to the bellman equation is the solution to the following linear system of equations,
$$
\rho v = u + A v
$$

```julia
f(x) = x
```
