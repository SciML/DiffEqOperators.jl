# Logistic Equation (test incomplete)

# Packages and inclusions
using ModelingToolkit,DiffEqOperators,DiffEqBase,LinearAlgebra,Test

# Tests
@testset "Test 00: Dt(u(t)) ~ u(t)*(1.0-u(t))" begin
    # Parameters, variables, and derivatives
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)

    # 1D PDE and boundary conditions
    eq  = Dt(u(t,x)) ~ 0.0*Dx(u(t,x))+u(t,x)*(1.0-u(t,x))
    bcs = [ u(0,x) ~ 0.5+x*0.0,
            u(t,0) ~ 0.5,
            u(t,1) ~ 0.5]

    # Space and time domains
    domains = [t ∈ IntervalDomain(0.0,5.0),
               x ∈ IntervalDomain(0.0,1.0)]

    # PDE system
    pdesys = PDESystem(eq,bcs,domains,[t,x],[u])

    # Method of lines discretization
    dx = 0.1
    order = 1
    discretization = MOLFiniteDifference(dx,order)

    # Convert the PDE problem into an ODE problem
    prob = discretize(pdesys,discretization)

    # Solve ODE problem
    using OrdinaryDiffEq

    sol = solve(prob,Euler(),dt=0.01,saveat=0.1)

    # Plot and save results
    #using Plots
    #time = domains[1].domain.lower:0.1:domains[1].domain.upper

    #plot(time,sol[5,1,:])
    #savefig("MOL_0D_Logistic.png")

    # Test
    # x_interval = domains[2].domain.lower+dx:dx:domains[2].domain.upper-dx
    # u = @. (0.5/(0.2*sqrt(2.0*3.1415)))*exp(-(x_interval-(0.75+0.6))^2/(2.0*0.2^2))
    # t_f = size(sol,3)
    # @test sol[t_f] ≈ u atol = 0.1;

end
