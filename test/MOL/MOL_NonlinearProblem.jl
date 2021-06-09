# 1D diffusion problem

# Packages and inclusions
using ModelingToolkit,DiffEqOperators,LinearAlgebra,Test,OrdinaryDiffEq
using ModelingToolkit: Differential
using DifferentialEquations


# Handrolled discrete laplace problem
@test begin
    @variables a b c d
    eqs = [0 ~ a - 1,
           0 ~ a + c - 2*b,
           0 ~ b + d - 2*c,
           0 ~ d - 1]
    ns = NonlinearSystem(eqs, [a, b, c, d], [])
    f = eval(generate_function(ns, [a, b, c, d])[2])
    prob = NonlinearProblem(ns, zeros(4), [])
    sol = solve(prob)
    @test sol.u ≈ ones(4)
end

# Laplace's Equation, same as above but with MOL discretization
@test begin
    @parameters x
    @variables u(..)
    Dxx = Differential(x)^2

    eq = Dxx(u(x)) ~ 0
    dx = 1/3

    bcs = [u(0) ~ 1,
           u(1) ~ 1]

    # Space and time domains
    domains = [x ∈ IntervalDomain(0.0,1.0)]

    pdesys = PDESystem([eq],bcs,domains,[x],[u(x)])
    discretization = MOLFiniteDifference([x=>dx], nothing, centered_order=2)
    prob = discretize(pdesys,discretization)
    sol = solve(prob)

    @test sol.u ≈ ones(4)
end

# Laplace's Equation, linear solution
@test begin
    @parameters x
    @variables u(..)
    Dxx = Differential(x)^2

    eq = Dxx(u(x)) ~ 0
    dx = 0.1

    bcs = [u(0) ~ 1,
           u(1) ~ 2]

    # Space and time domains
    domains = [x ∈ IntervalDomain(0.0,1.0)]

    pdesys = PDESystem([eq],bcs,domains,[x],[u(x)])
    discretization = MOLFiniteDifference([x=>dx], nothing, centered_order=2)
    prob = discretize(pdesys,discretization)
    sol = solve(prob)

    @test sol.u ≈ 1.0:0.1:2.0
end
