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
    # plot(prob.space,Array(prob.extrapolation*sol[1]))
    # plot!(prob.space,Array(prob.extrapolation*sol[2]))
    # plot!(prob.space,Array(prob.extrapolation*sol[3]))
    # plot!(prob.space,Array(prob.extrapolation*sol[4]))
    # savefig("MOL_1D_Linear_Diffusion_Test00.png")

    # Test
    n = size(sol)[1]
    t_f = size(sol)[2]

    println("prob.space:",prob.space)
    println("prob.extrapolation*sol:",prob.extrapolation*sol[t_f])

    @test sol[t_f] ≈ zeros(n) atol = 0.001;
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

#    # Plot and save results
#    using Plots
#    plot(prob.space,Array(prob.extrapolation*sol[1]))
#    plot!(prob.space,Array(prob.extrapolation*sol[2]))
#    plot!(prob.space,Array(prob.extrapolation*sol[3]))
#    plot!(prob.space,Array(prob.extrapolation*sol[4]))
#    savefig("MOL_1D_Linear_Diffusion_Test01.png")

    # Test
    n = size(sol)[1]
    t_f = size(sol)[2]
    @test sol[t_f] ≈ zeros(n) atol = 0.001;
end

@testset "Test 02: Dt(u(t,x)) ~ Dxx(D*u(t,x))" begin
    # Parameters, variables, and derivatives
    @parameters t x D
    @variables u(..)
    @derivatives Dt'~t
    @derivatives Dxx''~x

    D = 1.2

    # 1D PDE and boundary conditions
    eq  = Dt(u(t,x)) ~ Dxx(D*u(t,x))
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

#    # Plot and save results
#    using Plots
#    plot(prob.space,Array(prob.extrapolation*sol[1]))
#    plot!(prob.space,Array(prob.extrapolation*sol[2]))
#    plot!(prob.space,Array(prob.extrapolation*sol[3]))
#    plot!(prob.space,Array(prob.extrapolation*sol[4]))
#    savefig("MOL_1D_Linear_Diffusion_Test02.png")

    # Test
    n = size(sol)[1]
    t_f = size(sol)[2]
    @test sol[t_f] ≈ zeros(n) atol = 0.001;
end

@testset "Test 03: Dt(u(t,x)) ~ Dx(D(t,x)*Dx(u(t,x)))" begin
    # Parameters, variables, and derivatives
    @parameters t x
    @variables u(..) D(..)
    @derivatives Dt'~t
    @derivatives Dx'~x

    # 1D PDE and boundary conditions
    # TODO: use D(t,x) ~ t*x
    eq  = [ Dt(u(t,x)) ~ Dx(D(t,x)*Dx(u(t,x))),
            D(t,x) ~ 1/(1+exp(-x*t)) ]
    bcs = [u(0,x) ~ -x*(x-1)*sin(x),
           u(t,0) ~ 0.0,
           u(t,1) ~ 0.0]

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
#    using Plots
#    plot(prob.space,Array(prob.extrapolation*sol[1]))
#    plot!(prob.space,Array(prob.extrapolation*sol[2]))
#    plot!(prob.space,Array(prob.extrapolation*sol[3]))
#    plot!(prob.space,Array(prob.extrapolation*sol[4]))
#    savefig("MOL_1D_Linear_Diffusion_Test03.png")

    # Test
    n = size(sol)[1]
    t_f = size(sol)[2]
    @test sol[t_f] ≈ zeros(n) atol = 0.01;
end
