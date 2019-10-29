# # Poisson Equation
#
# We want to solve the [poisson equation](https://en.wikipedia.org/wiki/Poisson_equation) on the unit interval. It is given by
# `Δu = f` with boundary conditions `u(0) = a` and `u(1) = b`
# First of all let us choose some values for the parameters and remark, that there is an exact solution:
f = 1.0
a = -1.0
b = 2.0

u_analytic(x) = f/2*x^2 + (b-a-f/2) * x + a

# We would like to recompute this solution numerically
using DiffEqOperators
using DiffEqOperators: DirichletBC


nknots = 10
h = 1.0/(nknots+1)
ord_deriv = 2
ord_approx = 2

Δ = CenteredDifference(ord_deriv, ord_approx, h, nknots)
spacings = fill(h, nknots)
bc = DirichletBC([a,b], spacings)

# Before solving the equation, lets take a look at Δ and bc:
# display(Array(Δ))
# display(bc*zeros(nknots))
# We see that `Δ` is a (lazy) matrix with the laplace stencil extended over the boundaries.
# And `bc` acts by padding the values just outside the boundaries.

u = (Δ*bc) \ fill(f, nknots)
knots = cumsum(spacings)

# Since we used a second order approximation and the analytic solution itself was a second order
# polynomial, we expect that they are equal up to rounding errors:
using Test
@test u ≈ u_analytic.(knots)
