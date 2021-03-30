# 2D diffusion problem

# Packages and inclusions
using ModelingToolkit,DiffEqOperators,LinearAlgebra,Test,OrdinaryDiffEq
using ModelingToolkit: Differential

# Tests
@testset "Test 00: Dt(u(t,x)) ~ Dxx(u(t,x,y)) + Dyy(u(t,x,y))" begin
    @parameters t x y
    @variables u(..)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2
    Dt = Differential(t)
    t_min= 0.
    t_max = 2.0
    x_min = 0.
    x_max = 2.
    y_min = 0.
    y_max = 2.

    # 3D PDE
    eq  = Dt(u(t,x,y)) ~ Dxx(u(t,x,y)) + Dyy(u(t,x,y))

    analytic_sol_func(t,x,y) = exp(x+y)*cos(x+y+4t)
    # Initial and boundary conditions
    bcs = [u(t_min,x,y) ~ analytic_sol_func(t_min,x,y),
        u(t,x_min,y) ~ analytic_sol_func(t,x_min,y),
        u(t,x_max,y) ~ analytic_sol_func(t,x_max,y),
        u(t,x,y_min) ~ analytic_sol_func(t,x,y_min),
        u(t,x,y_max) ~ analytic_sol_func(t,x,y_max)]

    # Space and time domains
    domains = [t ∈ IntervalDomain(t_min,t_max),
            x ∈ IntervalDomain(x_min,x_max),
            y ∈ IntervalDomain(y_min,y_max)]
    pdesys = PDESystem([eq],bcs,domains,[t,x,y],[u(t,x,y)])

    # Method of lines discretization
    dx = 0.1; dy = 0.2
    discretization = MOLFiniteDifference([x=>dx,y=>dy],t)
    prob = ModelingToolkit.discretize(pdesys,discretization)
    sol = solve(prob,Tsit5())

    # Test against exact solution
    # TODO: do this properly when sol[u] with reshape etc works
    @test sol.u[1][1] ≈ analytic_sol_func(sol.t[1],0.1,0.2)
    @test sol.u[1][2] ≈ analytic_sol_func(sol.t[1],0.2,0.2)
end
