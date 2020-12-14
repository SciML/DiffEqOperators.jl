# 1D diffusion problem

# TODO: Add more complex tests.

# Packages and inclusions
using ModelingToolkit,DiffEqOperators,DiffEqBase,LinearAlgebra,Test

# Tests
@testset "Test 00: Dt(u(t,x)) ~ Dxx(u(t,x))" begin
    # Parameters, variables, and derivatives
    @parameters t x
    @variables u(..)
    @derivatives Dt'~t
    @derivatives Dxx''~x

    # 1D PDE and boundary conditions
    eq  = Dt(u(t,x)) ~ Dxx(u(t,x))
    bcs = [u(0,x) ~ -x*(x-1)*sin(x),
           u(t,0) ~ 0.0,
           u(t,1) ~ 0.0]

    # Space and time domains
    domains = [t ∈ IntervalDomain(0.0,1.0),
               x ∈ IntervalDomain(0.0,1.0)]

    # PDE system
    pdesys = PDESystem(eq,bcs,domains,[t,x],[u])

    # Method of lines discretization
    dx = 0.1
    order = 2
    discretization = MOLFiniteDifference(dx,order)

    # Convert the PDE problem into an ODE problem
    prob = discretize(pdesys,discretization)

    # Solve ODE problem
    using OrdinaryDiffEq
    sol = solve(prob,Tsit5(),saveat=0.1)

    # Plot and save results
    # using Plots
    # plot(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,1]))
    # plot!(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,2]))
    # plot!(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,3]))
    # plot!(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,4]))
    # savefig("MOL_1D_Linear_Diffusion_Test00.png")

    # Test
    n = size(sol,1)
    t_f = size(sol,3)

    @test sol[:,1,t_f] ≈ zeros(n) atol = 0.001;
end

@testset "Test 01: Dt(u(t,x)) ~ D*Dxx(u(t,x))" begin
    # Parameters, variables, and derivatives
    @parameters t x D
    @variables u(..)
    @derivatives Dt'~t
    @derivatives Dxx''~x

    D = 1.1

    # 1D PDE and boundary conditions
    eq  = Dt(u(t,x)) ~ D*Dxx(u(t,x))
    bcs = [u(0,x) ~ -x*(x-1)*sin(x),
           u(t,0) ~ 0.0,
           u(t,1) ~ 0.0]

    # Space and time domains
    domains = [t ∈ IntervalDomain(0.0,1.0),
               x ∈ IntervalDomain(0.0,1.0)]

    # PDE system
    pdesys = PDESystem(eq,bcs,domains,[t,x,D],[u])

    # Method of lines discretization
    dx = 0.1
    order = 2
    discretization = MOLFiniteDifference(dx,order)

    # Convert the PDE problem into an ODE problem
    prob = discretize(pdesys,discretization)

    # Solve ODE problem
    using OrdinaryDiffEq
    sol = solve(prob,Tsit5(),saveat=0.1)

    # Plot and save results
    # using Plots
    # plot(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,1]))
    # plot!(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,2]))
    # plot!(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,3]))
    # plot!(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,4]))
    # savefig("MOL_1D_Linear_Diffusion_Test01.png")

    # Test
    n = size(sol,1)
    t_f = size(sol,3)
    @test sol[:,1,t_f] ≈ zeros(n) atol = 0.001;
end

@testset "Test 02: Dt(u(t,x)) ~ Dx(D(t,x))*Dx(u(t,x))+D(t,x)*Dxx(u(t,x))" begin
    # Parameters, variables, and derivatives
    @parameters t x
    @variables u(..) D(..)
    @derivatives Dt'~t
    @derivatives Dx'~x
    @derivatives Dxx''~x

    # 1D PDE and boundary conditions

    eq  = [ Dt(u(t,x)) ~ Dx(D(t,x))*Dx(u(t,x))+D(t,x)*Dxx(u(t,x)),
            D(t,x) ~ 0.999 + 0.001 * t * x  ]

    bcs = [u(0,x) ~ -x*(x-1)*sin(x),
           u(t,0) ~ 0.0,
           u(t,1) ~ 0.0,
           D(0,x) ~ 0.999,
           D(t,0) ~ 0.999,
           D(t,1) ~ 0.999 + 0.001 * t ]

    # Space and time domains
    domains = [t ∈ IntervalDomain(0.0,1.0),
               x ∈ IntervalDomain(0.0,1.0)]

    # PDE system
    pdesys = PDESystem(eq,bcs,domains,[t,x],[u,D])

    # Method of lines discretization
    dx = 0.1
    order = 2
    discretization = MOLFiniteDifference(dx,order)

    # Convert the PDE problem into an ODE problem
    prob = discretize(pdesys,discretization)

    # Solve ODE problem
    using OrdinaryDiffEq
    sol = solve(prob,Tsit5(),saveat=0.1)
    
    # Plot and save results
    # using Plots
    # plot(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,1]))
    # plot!(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,2]))
    # plot!(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,3]))
    # plot!(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,4]))
    # savefig("MOL_1D_Linear_Diffusion_Test02.png")

    # Test
    n = size(sol,1)
    t_f = size(sol,3)
    @test sol[:,1,t_f] ≈ zeros(n) atol = 0.1;
end
