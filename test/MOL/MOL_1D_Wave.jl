# 1D wave equation problems

# Packages and inclusions
using ModelingToolkit,DiffEqOperators,LinearAlgebra,Test,OrdinaryDiffEq, DomainSets
using ModelingToolkit: Differential

# Tests
# @testset "Test 00: Dtt(u(t,x)) ~ Dxx(u(t,x))" begin
    # Method of Manufactured Solutions
    u_exact = (x,t) -> exp.(-t) * cos.(x)

    # Parameters, variables, and derivatives
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dtt = Differential(t)^2
    Dxx = Differential(x)^2

    # 1D PDE and boundary conditions
    eq  = Dtt(u(t,x)) ~ -Dxx(u(t,x))
    bcs = [u(0,x) ~ cos(x),
           Dt(u(0,x)) ~ -cos(x),
           u(t,0) ~ exp(-t),
           u(t,Float64(π)) ~ -exp(-t)]

    # Space and time domains
    domains = [t ∈ Interval(0.0,1.0),
               x ∈ Interval(0.0,Float64(π))]

    # PDE system
    pdesys = PDESystem(eq,bcs,domains,[t,x],[u(t,x)])

    # Method of lines discretization
    dx = range(0.0,Float64(π),length=30)
    order = 2
    discretization = MOLFiniteDifference([x=>dx],t)

    # Convert the PDE problem into an ODE problem
    prob = discretize(pdesys,discretization)
    sys, tspan = SciMLBase.symbolic_discretize(pdesys,discretization)
    # Solve ODE problem
    sol = solve(prob,Tsit5(),saveat=0.1)
    x_sol = dx[2:end-1]
    t_sol = sol.t

    # Test against exact solution
    # for i in 1:length(sol)
    i = 10
        exact = u_exact(x_sol, t_sol[i])
        # non-differential states only (ode_sys_lowering creates additional differential states)
        u_approx = sol.u[i][1:28]
        @test all(isapprox.(u_approx, exact, atol=0.01))
    # end
# end

