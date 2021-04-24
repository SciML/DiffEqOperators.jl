# 1D diffusion problem

# TODO: Add more complex tests.

# Packages and inclusions
using ModelingToolkit,DiffEqOperators,LinearAlgebra,Test,OrdinaryDiffEq
using ModelingToolkit: Differential

@testset "Kuramoto–Sivashinsky equation" begin
    @parameters x, t
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dx2 = Differential(x)^2
    Dx3 = Differential(x)^3
    Dx4 = Differential(x)^4

    α = 1
    β = 4
    γ = 1
    eq = Dt(u(x,t)) ~ -u(x,t)*Dx(u(x,t)) - α*Dx2(u(x,t)) - β*Dx3(u(x,t)) - γ*Dx4(u(x,t))

    u_analytic(x,t;z = -x/2+t) = 11 + 15*tanh(z) -15*tanh(z)^2 - 15*tanh(z)^3
    du(x,t;z = -x/2+t) = 15/2*(tanh(z) + 1)*(3*tanh(z) - 1)*sech(z)^2

    bcs = [u(x,0) ~ u_analytic(x,0),
           u(-10,t) ~ u_analytic(-10,t),
           u(10,t) ~ u_analytic(10,t),
           Dx(u(-10,t)) ~ du(-10,t),
           Dx(u(10,t)) ~ du(10,t)]

    # Space and time domains
    domains = [x ∈ IntervalDomain(-10.0,10.0),
               t ∈ IntervalDomain(0.0,1.0)]
    # Discretization
    dx = 0.4; dt = 0.2

    discretization = MOLFiniteDifference([x=>dx],t;centered_order=4)
    pde_system = PDESystem(eq,bcs,domains,[x,t],[u(x,t)])
    prob = discretize(pde_system,discretization)

    # TODO: finish this.
end
