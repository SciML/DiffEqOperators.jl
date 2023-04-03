# DiffEqOperators.jl

[![Join the chat at https://julialang.zulipchat.com #sciml-bridged](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
[![Global Docs](https://img.shields.io/badge/docs-SciML-blue.svg)](https://docs.sciml.ai/DiffEqOperators/stable/)

[![codecov](https://codecov.io/gh/SciML/DiffEqOperators.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/SciML/DiffEqOperators.jl)
[![Build Status](https://github.com/SciML/DiffEqOperators.jl/workflows/CI/badge.svg)](https://github.com/SciML/DiffEqOperators.jl/actions?query=workflow%3ACI)
[![Build status](https://badge.buildkite.com/0bc9acab7cf614b556a704cfe8422a5e3ab0cfbf3dbec83af7.svg)](https://buildkite.com/julialang/diffeqoperators-dot-jl)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

## This Package is in the Process of being Deprecated
## Alternatives:
- For automated finite difference discretization of symbolically-defined PDEs, see [MethodOfLines.jl](https://docs.sciml.ai/MethodOfLines/stable/).
- For MatrixFreeOperators, and other non-derivative operators, see [SciMLOperators.jl](https://docs.sciml.ai/SciMLOperators/stable/).
- For VecJacOperators and JacVecOperators, see [SparseDiffTools.jl](https://github.com/SciML/SparseDiffTools.jl).

# README
DiffEqOperators.jl is a package for finite difference discretization of partial
differential equations. It allows building lazy operators for high order non-uniform finite differences in an arbitrary number of dimensions, including vector calculus operators.

For automatic Method of Lines discretization of PDEs, better suited to nonlinear systems of equations and more complex boundary conditions, please see [MethodOfLines.jl](https://docs.sciml.ai/MethodOfLines/stable/)

For the operators, both centered and
[upwind](https://en.wikipedia.org/wiki/Upwind_scheme) operators are provided,
for domains of any dimension, arbitrarily spaced grids, and for any order of accuracy.
The cases of 1, 2, and 3 dimensions with an evenly spaced grid are optimized with a
convolution routine from `NNlib.jl`. Care is taken to give efficiency by avoiding
unnecessary allocations, using purpose-built stencil compilers, allowing GPUs
and parallelism, etc. Any operator can be concretized as an `Array`, a
`BandedMatrix` or a sparse matrix.

## Documentation

For information on using the package,
[see the stable documentation](https://docs.sciml.ai/DiffEqOperators/stable/). Use the
[in-development documentation](https://docs.sciml.ai/DiffEqOperators/dev/) for the version of
the documentation which contains the unreleased features.

## Example 1: Finite Difference Operator Solution for the Heat Equation

```julia
using DiffEqOperators, OrdinaryDiffEq

# # Heat Equation
# This example demonstrates how to combine `OrdinaryDiffEq` with `DiffEqOperators` to solve a time-dependent PDE.
# We consider the heat equation on the unit interval, with Dirichlet boundary conditions:
# ∂ₜu = Δu
# u(x=0,t)  = a
# u(x=1,t)  = b
# u(x, t=0) = u₀(x)
#
# For `a = b = 0` and `u₀(x) = sin(2πx)` a solution is given by:
u_analytic(x, t) = sin(2*π*x) * exp(-t*(2*π)^2)

nknots = 100
h = 1.0/(nknots+1)
knots = range(h, step=h, length=nknots)
ord_deriv = 2
ord_approx = 2

const Δ = CenteredDifference(ord_deriv, ord_approx, h, nknots)
const bc = Dirichlet0BC(Float64)

t0 = 0.0
t1 = 0.03
u0 = u_analytic.(knots, t0)

step(u,p,t) = Δ*bc*u
prob = ODEProblem(step, u0, (t0, t1))
alg = KenCarp4()
sol = solve(prob, alg)
```

