# Examples of Differential Operators for Economics
This provides examples of important differential operators, boundary values, and associated PDEs for important economics and finance applications.

## General notation
Define the following:

- $x_t$ be a stochastic process for a univariate function defined on a continuous domain $x \in (\underline{x}, \bar{x})$ where $-\infty < \underline{x} < \bar{x} < \infty$.  We will assume throughout that the domain is time-invariant.

- $W_t$ is Brownian motion

- Denote $k = 1,\ldots K$ as possible discrete states which may evolve according to a continuous-time Markov chain.

- Where appropriate, denote the stochastic process for the joint distribution as $x_{k,t}$

- Denote the drift of a process as $\mu(x), \mu(t,x),$ or $\mu_k(t,x)$ as appropriate.

- Similarly, denote the variance of the diffusion component of a process as $\sigma(x), \sigma(t,x),$ or $\sigma_k(t,x)$ as appropriate.

- For the switching probabilities of the markov chain, denote $\lambda_{kj}(t)$ as the arrival rate of changes between the $k$ and $j$ state.  Denote the time $t$ intensity matrix of this markov chain as $\mathbb{Q}(t)$ where for $k \neq j$, $\mathbb{Q}_{kj}(t) = \lambda_{kj}(t)$.

- When solving HJBE-style equations, the payoff in a particular state is denoted $b(x), b(t,x),$ or $b_k(t,x)$ as appropriate.

- The discount rate is denoted $r$ or $r(t)$ if time-varying.

- When thinking about time-varying problems, denote $T$ as the stationary point at which all parameters converge.  In particular

	- $\mu_k(t,x) = \mu_k(T,x)$ for all $t \geq T$

	- $\sigma_k(t,x) = \sigma_k(T,x)$ for all $t \geq T$

	- $r(t) = r(T)$ for all $t \geq T$

	- $b_k(t,x) = b_k(T,x)$ for all $t \geq T$

	- $\mathbb{Q}(t) = \mathbb{Q}(T)$ for all $t \geq T$

- Create a grid on $x$ where $i = 1, \ldots I$ and  and $x_1 = \underline{x} = x_1, x2, \ldots, x_I

- Denote $b_i = b(x_i)$ and when applied to the whole vector, drop the subscript.  i.e. $b \equiv \{b(x_i)\}_{i=1}^I$
- When solving for functions such as $v(x)$, denote the vectorized solution as $v_i = v(x_i)$ and the whole vector as $v = \{v_i\}_{i=1}^I$.

## Decomposing the Operators for Finite Differences
As we will only look at linear differential operators, we should be able to decompose them.  Denote the general process as $\mathcal{A}$ then we can define the following operators in preparation for using finite-differences

- $\mathcal{A}^{UW}_{1, s}$ as the upwind first order differential operator on the $x$ space.  That is,
	$$
	\mathcal{A}^{UW}_{1, s} \equiv \partial_x
	$$

	- This will use upwind finite differences and a first-order scheme, such that wherever $s(t,x) < 0$ it uses forward-differences, and vice-versa

	- Note that this operator is not directly using the drift, just the sign.

- $\mathcal{A}^{B}_{1}$ as the backwards first order differential operator and $\mathcal{A}^{F}_{1}$ as the forwards first-order differential operator.

- $\mathcal{A}_2$ is the second-order central differences operator


## Simple Time-Invariant Reflected Univariate Diffusion
Take the stochastic process
$$
d x_t = \mu(x_t)dt + \sigma(x_t)d W_t
$$
where $W_t$ is Brownian motion and reflecting barriers at $x \in (\underline{x}, \bar{x})$ where $-\infty < \underline{x} < \bar{x} < \infty$.

The partial differential operator (infinitesimal generator) associated with the stochastic process is
$$
	\mathcal{A} \equiv \mu(x)\partial_x + \frac{\sigma(x)^2}{2}\partial_{xx}
$$

Then, if the payoff in state $x$ is $b(x)$, and payoffs are discounted at rate $r$, then the Bellman equation is,
$$
r v(x) = b(x) + \mathcal{A}v(x)
$$
With boundary values $v'(\underline{x}) = 0$ and $v'(\bar{x}) = 0$

**Discretizing**: To map to the general setup:

- The operator above can be decomposed as,
$$
\mathcal{A} = \mu(x) \mathcal{A}^{UW}_{1,\, (\mu(x)>0)} + \sigma(x)\mathcal{A}_2
$$

- The boundary values are just the Neumann0 boundaries at $\underline{x}$, i.e. $\partial_x v(\underline{x}) = 0$ and $\partial_x v(\bar{x}) = 0$

- If $\mathcal{A}$ is then discretized to a matrix $A$, then the bellman equation can be solved as the system,
$$
r v = b + A v
$$
