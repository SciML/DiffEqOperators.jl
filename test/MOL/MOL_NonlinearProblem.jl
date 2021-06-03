# 1D diffusion problem

# Packages and inclusions
using ModelingToolkit,DiffEqOperators,LinearAlgebra,Test,OrdinaryDiffEq
using ModelingToolkit: Differential

# Laplace's Equation
@test_broken begin
    @parameters x
    @variables u(..)
    Dxx = Differential(x)^2

    eq = Dxx(u(x)) ~ 0
    dx = 0.1

    bcs = [u(0) ~ 1,
           u(1) ~ 1]

    # Space and time domains
    domains = [x ∈ IntervalDomain(0.0,1.0)]

    pdesys = PDESystem([eq],bcs,domains,[x],[u(x)])
    discretization = MOLFiniteDifference([x=>dx], nothing, centered_order=2)
    prob = discretize(pdesys,discretization)
end

