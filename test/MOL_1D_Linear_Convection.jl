# 1D linear convection problem

# Packages and inclusions
using ModelingToolkit,DiffEqOperators,DiffEqBase,LinearAlgebra,Test

# Tests

@testset "Test 00: Dt(u(t,x)) ~ -Dx(u(t,x))" begin
    # Parameters, variables, and derivatives
    @parameters t x
    @variables u(..)
    @derivatives Dt'~t
    @derivatives Dx'~x

    # 1D PDE and boundary conditions
    eq  = Dt(u(t,x)) ~ -Dx(u(t,x))
    bcs = [u(0,x) ~ (0.5/(0.2*sqrt(2.0*3.1415)))*exp(-(x  -0.75)^2/(2.0*0.2^2)),
           u(t,0) ~ 0.0,
           u(t,2) ~ 0.0]

    # Space and time domains
    domains = [t ∈ IntervalDomain(0.0,0.6),
               x ∈ IntervalDomain(0.0,2.0)]

    # PDE system
    pdesys = PDESystem(eq,bcs,domains,[t,x],[u])

    # Method of lines discretization
    dx = 2/80
    order = 1
    discretization = MOLFiniteDifference(dx,order)

    # Convert the PDE problem into an ODE problem
    prob = discretize(pdesys,discretization)

    # Solve ODE problem
    using OrdinaryDiffEq
    sol = solve(prob,Euler(),dt=.025,saveat=0.1)

    # Plot and save results
    # using Plots
    # plot(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,1]))
    # plot!(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,2]))
    # plot!(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,3]))
    # plot!(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,4]))
    # plot!(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,5]))
    # plot!(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,6]))
    # savefig("MOL_1D_Linear_Convection_Test00.png")

    # Test
    x_interval = domains[2].domain.lower+dx:dx:domains[2].domain.upper-dx
    u = @. (0.5/(0.2*sqrt(2.0*3.1415)))*exp(-(x_interval-(0.75+0.6))^2/(2.0*0.2^2))
    t_f = size(sol,3)

    @test sol[:,1,t_f] ≈ u atol = 0.1;
    

end

@testset "Test 01: Dt(u(t,x)) ~ -Dx(u(t,x)) + 0.01" begin

    # Parameters, variables, and derivatives
    @parameters t x
    @variables u(..)
    @derivatives Dt'~t
    @derivatives Dx'~x

    # 1D PDE and boundary conditions
    eq  = Dt(u(t,x)) ~ -Dx(u(t,x)) + 0.01
    bcs = [u(0,x) ~ (0.5/(0.2*sqrt(2.0*3.1415)))*exp(-(x  -0.75)^2/(2.0*0.2^2)),
           u(t,0) ~ 0.0,
           u(t,2) ~ 0.0]

    # Space and time domains
    domains = [t ∈ IntervalDomain(0.0,0.6),
               x ∈ IntervalDomain(0.0,2.0)]

    # PDE system
    pdesys = PDESystem(eq,bcs,domains,[t,x],[u])

    # Method of lines discretization
    dx = 2/80
    order = 1
    discretization = MOLFiniteDifference(dx,order)

    # Convert the PDE problem into an ODE problem
    prob = discretize(pdesys,discretization)

    # Solve ODE problem
    using OrdinaryDiffEq
    sol = solve(prob,Euler(),dt=.025,saveat=0.1)

    # Plot and save results
    # using Plots
    # plot(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,1]))
    # plot!(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,2]))
    # plot!(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,3]))
    # plot!(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,4]))
    # plot!(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,5]))
    # plot!(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,6]))
    # savefig("MOL_1D_Linear_Convection_Test01.png")

    # Test
    x_interval = domains[2].domain.lower+dx:dx:domains[2].domain.upper-dx
    u = @. (0.5/(0.2*sqrt(2.0*3.1415)))*exp(-(x_interval-(0.75+0.6))^2/(2.0*0.2^2))
    t_f = size(sol,3)
    @test sol[:,1,t_f] ≈ u atol = 0.1;

end

@testset "Test 02: Dt(u(t,x)) ~ -v*Dx(u(t,x))" begin
    # Parameters, variables, and derivatives
    @parameters t x v
    @variables u(..)
    @derivatives Dt'~t
    @derivatives Dx'~x

    v = 1.0

    # 1D PDE and boundary conditions
    eq  = Dt(u(t,x)) ~ -v*Dx(u(t,x))
    bcs = [u(0,x) ~ (0.5/(0.2*sqrt(2.0*3.1415)))*exp(-(x-0.75)^2/(2.0*0.2^2)),
           u(t,0) ~ 0.0,
           u(t,2) ~ 0.0]

    # Space and time domains
    domains = [t ∈ IntervalDomain(0.0,0.6),
               x ∈ IntervalDomain(0.0,2.0)]

    # PDE system
    pdesys = PDESystem(eq,bcs,domains,[t,x],[u])

    # Method of lines discretization
    dx = 2/80
    order = 1
    discretization = MOLFiniteDifference(dx,order)

    # Convert the PDE problem into an ODE problem
    prob = discretize(pdesys,discretization)

    # Solve ODE problem
    using OrdinaryDiffEq
    sol = solve(prob,Euler(),dt=.025,saveat=0.1)

    # Plot and save results
    # using Plots
    # plot(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,1]))
    # plot!(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,2]))
    # plot!(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,3]))
    # plot!(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,4]))
    # plot!(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,5]))
    # plot!(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,6]))
    # savefig("MOL_1D_Linear_Convection_Test02.png")

    # Test
    x_interval = domains[2].domain.lower+dx:dx:domains[2].domain.upper-dx
    u = @. (0.5/(0.2*sqrt(2.0*3.1415)))*exp(-(x_interval-(0.75+v*0.6))^2/(2.0*0.2^2))
    t_f = size(sol,3)
    @test sol[:,1,t_f] ≈ u atol = 0.1;
end

@testset "Test 03: Dt(u(t,x)) ~ -Dx(v(t,x))*u(t,x)-v(t,x)*Dx(u(t,x)) with v(t,x)=1" begin
    # Parameters, variables, and derivatives
    @parameters t x
    @variables v(..) u(..)
    @derivatives Dt'~t
    @derivatives Dx'~x

    # 1D PDE and boundary conditions
    eq  = [ Dt(u(t,x)) ~ -(Dx(v(t,x))*u(t,x)+v(t,x)*Dx(u(t,x))),
            v(t,x) ~ 1.0 ]
    bcs = [u(0,x) ~ (0.5/(0.2*sqrt(2.0*3.1415)))*exp(-(x-0.75)^2/(2.0*0.2^2)),
           u(t,0) ~ 0.0,
           u(t,2) ~ 0.0,
           v(0,x) ~ 1.0,
           v(t,0) ~ 1.0,
           v(t,2) ~ 1.0
           ]

    # Space and time domains
    domains = [t ∈ IntervalDomain(0.0,0.6),
               x ∈ IntervalDomain(0.0,2.0)]

    # PDE system
    pdesys = PDESystem(eq,bcs,domains,[t,x],[u,v])

    # Method of lines discretization
    dx = 2/80
    order = 1
    discretization = MOLFiniteDifference(dx,order)

    # Convert the PDE problem into an ODE problem
    prob = discretize(pdesys,discretization)

    # Solve ODE problem
    using OrdinaryDiffEq
    sol = solve(prob,Euler(),dt=.025,saveat=0.1)

    #Plot and save results
    # using Plots
    # plot(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,1]))
    # plot!(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,2]))
    # plot!(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,3]))
    # plot!(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,4]))
    # plot!(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,5]))
    # plot!(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,6]))
    # savefig("MOL_1D_Linear_Convection_Test03.png")

    # Test
    x_interval = domains[2].domain.lower+dx:dx:domains[2].domain.upper-dx
    u = @. (0.5/(0.2*sqrt(2.0*3.1415)))*exp(-(x_interval-(0.75+1.0*0.6))^2/(2.0*0.2^2))
    t_f = size(sol,3)

    @test sol[:,1,t_f] ≈ u atol = 0.1;
end

@testset "Test 04: Dt(u(t,x)) ~ -Dx(v(t,x))*u(t,x)-v(t,x)*Dx(u(t,x)) with v(t,x)=0.999 + 0.001 * t * x " begin
    # Parameters, variables, and derivatives
    @parameters t x
    @variables v(..) u(..)
    @derivatives Dt'~t
    @derivatives Dx'~x

    # 1D PDE and boundary conditions
    eq  = [ Dt(u(t,x)) ~ -Dx(v(t,x))*u(t,x)-v(t,x)*Dx(u(t,x)),
            v(t,x) ~ 0.999 + 0.001 * t * x ]
    bcs = [u(0,x) ~ (0.5/(0.2*sqrt(2.0*3.1415)))*exp(-(x-0.75)^2/(2.0*0.2^2)),
           u(t,0) ~ 0.0,
           u(t,2) ~ 0.0,
           v(0,x) ~ 0.999,
           v(t,0) ~ 0.999,
           v(t,2) ~ 0.999 + 0.001 * t * 2.0]

    # Space and time domains
    domains = [t ∈ IntervalDomain(0.0,0.6),
               x ∈ IntervalDomain(0.0,2.0)]

    # PDE system
    pdesys = PDESystem(eq,bcs,domains,[t,x],[u,v])

    # Method of lines discretization
    dx = 2/80
    order = 1
    discretization = MOLFiniteDifference(dx,order)

    # Convert the PDE problem into an ODE problem
    prob = discretize(pdesys,discretization)

    # Solve ODE problem
    using OrdinaryDiffEq
    sol = solve(prob,Euler(),dt=.025,saveat=0.1)

    # Plot and save results
    # using Plots
    # plot(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,1]))
    # plot!(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,2]))
    # plot!(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,3]))
    # plot!(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,4]))
    # plot!(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,5]))
    # plot!(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,6]))
    # savefig("MOL_1D_Linear_Convection_Test04.png")

    # Test
    x_interval = domains[2].domain.lower+dx:dx:domains[2].domain.upper-dx
    u = @. (0.5/(0.2*sqrt(2.0*3.1415)))*exp(-(x_interval-(0.75+1.0*0.6))^2/(2.0*0.2^2))
    t_f = size(sol,3)

    @test sol[:,1,t_f] ≈ u atol = 0.1;
end
