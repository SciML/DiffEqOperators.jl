# 1D diffusion problem

# TODO: Add more complex tests.

# Packages and inclusions
using ModelingToolkit,DiffEqOperators,DiffEqBase,LinearAlgebra,Test,OrdinaryDiffEq

# Tests
@testset "Test 00: Dt(u(t,x)) ~ Dxx(u(t,x))" begin
    # Method of Manufactured Solutions
    u_exact = (x,t) -> exp.(-t) * cos.(x)

    # Parameters, variables, and derivatives
    @parameters t x
    @variables u(..)
    @derivatives Dt'~t
    @derivatives Dxx''~x

    # 1D PDE and boundary conditions
    eq  = Dt(u(t,x)) ~ Dxx(u(t,x))
    bcs = [u(0,x) ~ cos(x),
           u(t,0) ~ exp(-t),
           u(t,Float64(pi)) ~ -exp(-t)]

    # Space and time domains
    domains = [t ∈ IntervalDomain(0.0,1.0),
               x ∈ IntervalDomain(0.0,Float64(pi))]

    # PDE system
    pdesys = PDESystem(eq,bcs,domains,[t,x],[u])

    # Method of lines discretization
    dx = 0.1
    order = 2
    discretization = MOLFiniteDifference(dx,order)

    # Convert the PDE problem into an ODE problem
    prob = discretize(pdesys,discretization)

    # Solve ODE problem
    sol = solve(prob,Tsit5(),saveat=0.1)
    x = prob.space[2]
    t = sol.t

    # Plot and save results
    # using Plots
    # plot()
    # for i in 1:4
    #     plot!(x,Array(prob.extrapolation[1](t[i])*sol.u[i]))
    #     scatter!(x, u_exact(x, t[i]))
    # end
    # savefig("MOL_1D_Linear_Diffusion_Test00.png")

    # Test against exact solution
    for i in 1:size(t,1)
        u_approx = Array(prob.extrapolation[1](t[i])*sol.u[i])
        @test u_approx ≈ u_exact(x, t[i]) atol=0.01
    end
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
    sol = solve(prob,Tsit5(),saveat=0.1)

    # Plot and save results
    # x = prob.space[2]
    # t = sol.t
    
    # using Plots
    # plot()
    # for i in 1:4
    #     plot!(x,Array(prob.extrapolation[1](t[i])*sol.u[i]))
    # end
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
    sol = solve(prob,Tsit5(),saveat=0.1)
    
    # Plot and save results
    # x = prob.space[2]
    # t = sol.t
    
    # using Plots
    # plot()
    # for i in 1:4
    #     plot!(x,Array(prob.extrapolation[1](t[i])*sol.u[i]))
    # end
    # savefig("MOL_1D_Linear_Diffusion_Test02.png")

    # Test
    n = size(sol,1)
    t_f = size(sol,3)
    @test sol[:,1,t_f] ≈ zeros(n) atol = 0.1;
end

@testset "Test 03: Dt(u(t,x)) ~ Dxx(u(t,x)), Neumann BCs" begin
    # Method of Manufactured Solutions
    u_exact = (x,t) -> exp.(-t) * sin.(x)

    # Parameters, variables, and derivatives
    @parameters t x
    @variables u(..)
    @derivatives Dt'~t
    @derivatives Dx'~x
    @derivatives Dxx''~x

    # 1D PDE and boundary conditions
    eq  = Dt(u(t,x)) ~ Dxx(u(t,x))
    bcs = [u(0,x) ~ sin(x),
           Dx(u(t,0)) ~ exp(-t),
           Dx(u(t,Float64(pi))) ~ -exp(-t)]

    # Space and time domains
    domains = [t ∈ IntervalDomain(0.0,1.0),
               x ∈ IntervalDomain(0.0,Float64(pi))]

    # PDE system
    pdesys = PDESystem(eq,bcs,domains,[t,x],[u])

    # Method of lines discretization
    dx = 0.1
    order = 2
    discretization = MOLFiniteDifference(dx,order)

    # Convert the PDE problem into an ODE problem
    prob = discretize(pdesys,discretization)

    # Solve ODE problem
    sol = solve(prob,Tsit5(),saveat=0.1)
    x = prob.space[2]
    t = sol.t

    # Plot and save results
    # using Plots
    # plot()
    # for i in 1:4
    #     plot!(x,Array(prob.extrapolation[1](t[i])*sol.u[i]))
    #     scatter!(x, u_exact(x, t[i]))
    # end
    # savefig("MOL_1D_Linear_Diffusion_Test03.png")

    # Test against exact solution
    for i in 1:size(t,1)
        u_approx = Array(prob.extrapolation[1](t[i])*sol.u[i])
        @test u_approx ≈ u_exact(x, t[i]) atol=0.01
    end
end

@testset "Test 04: Dt(u(t,x)) ~ Dxx(u(t,x)), Neumann + Dirichlet BCs" begin
    # Method of Manufactured Solutions
    u_exact = (x,t) -> exp.(-t) * sin.(x)

    # Parameters, variables, and derivatives
    @parameters t x
    @variables u(..)
    @derivatives Dt'~t
    @derivatives Dx'~x
    @derivatives Dxx''~x

    # 1D PDE and boundary conditions
    eq  = Dt(u(t,x)) ~ Dxx(u(t,x))
    bcs = [u(0,x) ~ sin(x),
           u(t,0) ~ 0.0,
           Dx(u(t,Float64(pi))) ~ -exp(-t)]

    # Space and time domains
    domains = [t ∈ IntervalDomain(0.0,1.0),
               x ∈ IntervalDomain(0.0,Float64(pi))]

    # PDE system
    pdesys = PDESystem(eq,bcs,domains,[t,x],[u])

    # Method of lines discretization
    dx = 0.1
    order = 2
    discretization = MOLFiniteDifference(dx,order)

    # Convert the PDE problem into an ODE problem
    prob = discretize(pdesys,discretization)

    # Solve ODE problem
    sol = solve(prob,Tsit5(),saveat=0.1)
    x = prob.space[2]
    t = sol.t

    # Plot and save results
    # using Plots
    # plot()
    # for i in 1:4
    #     plot!(x,Array(prob.extrapolation[1](t[i])*sol.u[i]))
    #     scatter!(x, u_exact(x, t[i]))
    # end
    # savefig("MOL_1D_Linear_Diffusion_Test04.png")

    # Test against exact solution
    for i in 1:size(t,1)
        u_approx = Array(prob.extrapolation[1](t[i])*sol.u[i])
        @test u_approx ≈ u_exact(x, t[i]) atol=0.01
    end
end

@testset "Test 05: Dt(u(t,x)) ~ Dxx(u(t,x)), Robin BCs" begin
    # Method of Manufactured Solutions
    u_exact = (x,t) -> exp.(-t) * sin.(x)

    # Parameters, variables, and derivatives
    @parameters t x
    @variables u(..)
    @derivatives Dt'~t
    @derivatives Dx'~x
    @derivatives Dxx''~x

    # 1D PDE and boundary conditions
    eq  = Dt(u(t,x)) ~ Dxx(u(t,x))
    bcs = [u(0,x) ~ sin(x),
           u(t,-1.0) + 3Dx(u(t,-1.0)) ~ exp(-t) * (sin(-1.0) + 3cos(-1.0)),
           u(t,1.0) + Dx(u(t,1.0)) ~ exp(-t) * (sin(1.0) + cos(1.0))]

    # Space and time domains
    domains = [t ∈ IntervalDomain(0.0,1.0),
               x ∈ IntervalDomain(-1.0,1.0)]

    # PDE system
    pdesys = PDESystem(eq,bcs,domains,[t,x],[u])

    # Method of lines discretization
    dx = 0.1
    order = 2
    discretization = MOLFiniteDifference(dx,order)

    # Convert the PDE problem into an ODE problem
    prob = discretize(pdesys,discretization)

    # Solve ODE problem
    sol = solve(prob,Tsit5(),saveat=0.1)
    x = prob.space[2]
    t = sol.t

    # Plot and save results
    using Plots
    plot()
    for i in 1:4
        plot!(x,Array(prob.extrapolation[1](t[i])*sol.u[i]))
        scatter!(x, u_exact(x, t[i]))
    end
    savefig("MOL_1D_Linear_Diffusion_Test05.png")

    # Test against exact solution
    for i in 1:size(t,1)
        u_approx = Array(prob.extrapolation[1](t[i])*sol.u[i])
        @test u_approx ≈ u_exact(x, t[i]) atol=0.1
    end
end

@testset "Test errors" begin
    # Parameters, variables, and derivatives
    @parameters t x
    @variables u(..) v(..)
    @derivatives Dt'~t
    @derivatives Dx'~x
    @derivatives Dxx''~x

    # 1D PDE and boundary conditions
    eq  = Dt(u(t,x)) ~ Dxx(u(t,x))
    # Space and time domains
    domains = [t ∈ IntervalDomain(0.0,1.0), x ∈ IntervalDomain(0.0,1.0)]

    # Method of lines discretization
    dx = 0.1
    order = 2
    discretization = MOLFiniteDifference(dx,order)

    # Missing boundary condition
    bcs = [u(0,x) ~ 0.5 + sin(2pi*x),
           Dx(u(t,1)) ~ 0.0]    
    pdesys = PDESystem(eq,bcs,domains,[t,x],[u])
    @test_throws BoundaryConditionError discretize(pdesys,discretization)
    
    # Boundary condition not at t=0
    bcs = [u(1,x) ~ 0.5 + sin(2pi*x),
           Dx(u(t,0)) ~ 0.0,
           Dx(u(t,1)) ~ 0.0]
    pdesys = PDESystem(eq,bcs,domains,[t,x],[u])
    @test_throws BoundaryConditionError discretize(pdesys,discretization)

    # Boundary condition not at an edge of the domain
    bcs = [u(0,x) ~ 0.5 + sin(2pi*x),
           Dx(u(t,0)) ~ 0.0,
           Dx(u(t,0.5)) ~ 0.0]    
    pdesys = PDESystem(eq,bcs,domains,[t,x],[u])
    @test_throws BoundaryConditionError discretize(pdesys,discretization)
   
    # Wrong format for Robin BCs
    bcs = [u(0,x) ~ 0.5 + sin(2pi*x),
           Dx(u(t,0)) ~ 0.0,
           u(t,1) * Dx(u(t,1)) ~ 0.0]    
    pdesys = PDESystem(eq,bcs,domains,[t,x],[u])
    @test_throws BoundaryConditionError discretize(pdesys,discretization)
    
    bcs = [u(0,x) ~ 0.5 + sin(2pi*x),
           Dx(u(t,0)) ~ 0.0,
           Dx(u(t,1)) + u(t,1) ~ 0.0]    
    pdesys = PDESystem(eq,bcs,domains,[t,x],[u])
    @test_throws BoundaryConditionError discretize(pdesys,discretization)
    
    bcs = [u(0,x) ~ 0.5 + sin(2pi*x),
           Dx(u(t,0)) ~ 0.0,
           u(t,1) / 2 + Dx(u(t,1)) ~ 0.0]    
    pdesys = PDESystem(eq,bcs,domains,[t,x],[u])
    @test_throws BoundaryConditionError discretize(pdesys,discretization)
    
    bcs = [u(0,x) ~ 0.5 + sin(2pi*x),
           Dx(u(t,0)) ~ 0.0,
           u(t,1) + Dx(u(t,1)) / 2 ~ 0.0]    
    pdesys = PDESystem(eq,bcs,domains,[t,x],[u])
    @test_throws BoundaryConditionError discretize(pdesys,discretization)
    
    # Mismatching arguments
    bcs = [u(0,x) ~ 0.5 + sin(2pi*x),
    Dx(u(t,0)) ~ 0.0,
    u(t,0) + Dx(u(t,1)) ~ 0.0]    
    pdesys = PDESystem(eq,bcs,domains,[t,x],[u])
    @test_throws AssertionError discretize(pdesys,discretization)
    
    # Mismatching variables
    bcs = [u(0,x) ~ 0.5 + sin(2pi*x),
           Dx(u(t,0)) ~ 0.0,
           u(t,1) + Dx(v(t,1)) ~ 0.0]    
    pdesys = PDESystem(eq,bcs,domains,[t,x],[u])
    @test_throws AssertionError discretize(pdesys,discretization)
    
end
