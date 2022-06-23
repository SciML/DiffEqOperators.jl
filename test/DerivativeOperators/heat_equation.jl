# # Heat Equation
#
# This example demonstrates how to combine `OrdinaryDiffEq` with `DiffEqOperators` to solve a time-dependent PDE.
# We consider the heat equation on the unit interval, with Dirichlet boundary conditions:
# ∂ₜu = Δu
# u(x=0,t)  = a
# u(x=1,t)  = b
# u(x, t=0) = u₀(x)
#
# For `a = b = 0` and `u₀(x) = sin(2πx)` a solution is given by:
u_analytic(x, t) = sin(2 * π * x) * exp(-t * (2 * π)^2)
#
# We want to reproduce it numerically
#
using DiffEqOperators, OrdinaryDiffEq

nknots = 100
h = 1.0 / (nknots + 1)
knots = range(h, step = h, length = nknots)
ord_deriv = 2
ord_approx = 2

const Δ = CenteredDifference(ord_deriv, ord_approx, h, nknots)
const bc = Dirichlet0BC(Float64)

t0 = 0.0
t1 = 0.03
u0 = u_analytic.(knots, t0)

step(u, p, t) = Δ * bc * u
prob = ODEProblem(step, u0, (t0, t1))
alg = KenCarp4()
sol = solve(prob, alg)

using Test
@test u_analytic.(knots, t1)≈sol[end] rtol=1e-3

# Because the creation of boundary conditions is cheap, we can
# implement time-dependent boundary conditions as follows:
# function step(u,p,t)
#    bc = DirichletBC(cos(t*1.5), sin(t))
#    Δ*bc*u
# end
