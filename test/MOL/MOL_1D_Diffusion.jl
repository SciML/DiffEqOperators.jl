# 1D diffusion problem

# TODO: Add more complex tests.

# Packages and inclusions
using ModelingToolkit,DiffEqOperators,LinearAlgebra,Test,OrdinaryDiffEq
using ModelingToolkit: Differential

# Tests
@testset "Test 00: Dt(u(t,x)) ~ Dxx(u(t,x))" begin
    # Method of Manufactured Solutions
    u_exact = (x,t) -> exp.(-t) * cos.(x)

    # Parameters, variables, and derivatives
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    # 1D PDE and boundary conditions
    eq  = Dt(u(t,x)) ~ Dxx(u(t,x))
    bcs = [u(0,x) ~ cos(x),
           u(t,0) ~ exp(-t),
           u(t,Float64(π)) ~ -exp(-t)]

    # Space and time domains
    domains = [t ∈ IntervalDomain(0.0,1.0),
               x ∈ IntervalDomain(0.0,Float64(π))]

    # PDE system
    pdesys = PDESystem(eq,bcs,domains,[t,x],[u(t,x)])

    # Method of lines discretization
    dx = range(0.0,Float64(π),length=30)
    order = 2
    discretization = MOLFiniteDifference([x=>dx],t)
    # Explicitly specify order of centered difference
    discretization_centered = MOLFiniteDifference([x=>dx],t;centered_order=order)
    # Higher order centered difference
    discretization_centered_order4 = MOLFiniteDifference([x=>dx],t;centered_order=4)

    # Convert the PDE problem into an ODE problem
    prob = discretize(pdesys,discretization)
    prob_centered = discretize(pdesys,discretization_centered)
    prob_centered_order4 = discretize(pdesys,discretization_centered_order4)

    # Solve ODE problem
    sol = solve(prob,Tsit5(),saveat=0.1)
    sol_centered = solve(prob_centered,Tsit5(),saveat=0.1)
    sol_centered_order4 = solve(prob_centered_order4,Tsit5(),saveat=0.1)

    x = dx[2:end-1]
    t = sol.t

    # Test against exact solution
    for i in 1:length(sol)
        exact = u_exact(x, t[i])
        u_approx = sol.u[i]
        u_approx_centered = sol_centered.u[i]
        u_approx_centered_order4 =sol_centered_order4.u[i]
        @test u_approx ≈ exact atol=0.01
        @test u_approx_centered ≈ exact atol=0.01
        @test u_approx_centered_order4 ≈ exact atol=0.01
    end
end


@testset "Test 01: Dt(u(t,x)) ~ D*Dxx(u(t,x))" begin
    # Parameters, variables, and derivatives
    @parameters t x D
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    # 1D PDE and boundary conditions
    eq  = Dt(u(t,x)) ~ D*Dxx(u(t,x))
    bcs = [u(0,x) ~ -x*(x-1)*sin(x),
           u(t,0) ~ 0.0,
           u(t,1) ~ 0.0]

    # Space and time domains
    domains = [t ∈ IntervalDomain(0.0,1.0),
               x ∈ IntervalDomain(0.0,1.0)]

    # PDE system
    pdesys = PDESystem(eq,bcs,domains,[t,x],[u(t,x)],[D=>10.0])

    # Method of lines discretization
    dx = 0.1
    order = 2
    discretization = MOLFiniteDifference([x=>dx],t)

    # Convert the PDE problem into an ODE problem
    prob = discretize(pdesys,discretization)

    # Solve ODE problem
    sol = solve(prob,Tsit5(),saveat=0.1)

    # Test
    n = size(sol,1)
    t_f = size(sol,3)
    @test sol[end] ≈ zeros(n) atol = 0.001;
end

# @test_set "Test 02: Dt(u(t,x)) ~ Dx(D(t,x))*Dx(u(t,x))+D(t,x)*Dxx(u(t,x))" begin
@test_broken begin
    # Parameters, variables, and derivatives
    @parameters t x
    @variables u(..) D(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Differential(x)^2

    D = (t,x) -> 0.999 + 0.001 * t * x
    DxD = expand_derivatives(Dx(D(t,x)))

    # 1D PDE and boundary conditions

    eq  = [ Dt(u(t,x)) ~ DxD*Dx(u(t,x))+D(t,x)*Dxx(u(t,x)),]

    bcs = [u(0,x) ~ -x*(x-1)*sin(x),
           u(t,0) ~ 0.0,
           u(t,1) ~ 0.0]

    # Space and time domains
    domains = [t ∈ IntervalDomain(0.0,1.0),
               x ∈ IntervalDomain(0.0,1.0)]

    # PDE system
    pdesys = PDESystem(eq,bcs,domains,[t,x],[u(t,x)])

    # Method of lines discretization
    dx = 0.1
    order = 2
    discretization = MOLFiniteDifference([x=>dx],t)

    # Convert the PDE problem into an ODE problem
    prob = discretize(pdesys,discretization)

    # Solve ODE problem
    sol = solve(prob,Tsit5(),saveat=0.1)

    # Test
    n = size(sol,1)
    t_f = size(sol,3)
    @test_broken sol[:,1,t_f] ≈ zeros(n) atol=0.01;
end

@testset "Test 03: Dt(u(t,x)) ~ Dxx(u(t,x)), Neumann BCs" begin
    # Method of Manufactured Solutions
    u_exact = (x,t) -> exp.(-t) * sin.(x)

    # Parameters, variables, and derivatives
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Differential(x)^2

    # 1D PDE and boundary conditions
    eq  = Dt(u(t,x)) ~ Dxx(u(t,x))
    bcs = [u(0,x) ~ sin(x),
           Dx(u(t,0)) ~ exp(-t),
           Dx(u(t,Float64(pi))) ~ -exp(-t)]

    # Space and time domains
    domains = [t ∈ IntervalDomain(0.0,1.0),
               x ∈ IntervalDomain(0.0,Float64(pi))]

    # PDE system
    pdesys = PDESystem(eq,bcs,domains,[t,x],[u(t,x)])

    # Method of lines discretization
    dx = range(0.0,Float64(π),length=30)
    order = 2
    discretization = MOLFiniteDifference([x=>dx],t)

    # Convert the PDE problem into an ODE problem
    prob = discretize(pdesys,discretization)

    # Solve ODE problem
    sol = solve(prob,Tsit5(),saveat=0.1)

    x = dx[2:end-1]
    t = sol.t

    # Test against exact solution
    for i in 1:length(sol)
        exact = u_exact(x, t[i])
        u_approx = sol.u[i]
        @test u_approx ≈ exact atol=0.01
    end
end

@testset "Test 04: Dt(u(t,x)) ~ Dxx(u(t,x)), Neumann + Dirichlet BCs" begin
    # Method of Manufactured Solutions
    u_exact = (x,t) -> exp.(-t) * sin.(x)

    # Parameters, variables, and derivatives
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Differential(x)^2

    # 1D PDE and boundary conditions
    eq  = Dt(u(t,x)) ~ Dxx(u(t,x))
    bcs = [u(0,x) ~ sin(x),
           u(t,0) ~ 0.0,
           Dx(u(t,Float64(pi))) ~ -exp(-t)]

    # Space and time domains
    domains = [t ∈ IntervalDomain(0.0,1.0),
               x ∈ IntervalDomain(0.0,Float64(pi))]

    # PDE system
    pdesys = PDESystem(eq,bcs,domains,[t,x],[u(t,x)])

    # Method of lines discretization
    dx = range(0.0,Float64(π),length=30)
    order = 2
    discretization = MOLFiniteDifference([x=>dx],t)

    # Convert the PDE problem into an ODE problem
    prob = discretize(pdesys,discretization)

    # Solve ODE problem
    sol = solve(prob,Tsit5(),saveat=0.1)
    x = dx[2:end-1]
    t = sol.t

    # Test against exact solution
    for i in 1:length(sol)
        exact = u_exact(x, t[i])
        u_approx = sol.u[i]
        @test u_approx ≈ exact atol=0.01
    end
end

@testset "Test 05: Dt(u(t,x)) ~ Dxx(u(t,x)), Robin BCs" begin
    # Method of Manufactured Solutions
    u_exact = (x,t) -> exp.(-t) * sin.(x)

    # Parameters, variables, and derivatives
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Differential(x)^2

    # 1D PDE and boundary conditions
    eq  = Dt(u(t,x)) ~ Dxx(u(t,x))
    bcs = [u(0,x) ~ sin(x),
           u(t,-1.0) + 3Dx(u(t,-1.0)) ~ exp(-t) * (sin(-1.0) + 3cos(-1.0)),
           4u(t,1.0) + Dx(u(t,1.0)) ~ exp(-t) * (4sin(1.0) + cos(1.0))]

    # Space and time domains
    domains = [t ∈ IntervalDomain(0.0,1.0),
               x ∈ IntervalDomain(-1.0,1.0)]

    # PDE system
    pdesys = PDESystem(eq,bcs,domains,[t,x],[u(t,x)])

    # Method of lines discretization
    dx = 0.01
    order = 2
    discretization = MOLFiniteDifference([x=>dx],t)

    # Convert the PDE problem into an ODE problem
    prob = discretize(pdesys,discretization)

    # Solve ODE problem
    sol = solve(prob,Tsit5(),saveat=0.1)
    x = (-1:dx:1)[2:end-1]
    t = sol.t

    # Test against exact solution
    for i in 1:length(sol)
        exact = u_exact(x, t[i])
        u_approx = sol.u[i]
        @test u_approx ≈ exact atol=0.1
    end
end


@testset "Test 06: Dt(u(t,x)) ~ Dxx(u(t,x)), time-dependent Robin BCs" begin
    # Method of Manufactured Solutions
    u_exact = (x,t) -> exp.(-t) * sin.(x)

    # Parameters, variables, and derivatives
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Differential(x)^2

    # 1D PDE and boundary conditions
    eq  = Dt(u(t,x)) ~ Dxx(u(t,x))
    bcs = [u(0,x) ~ sin(x),
          t^2 * u(t,-1.0) + 3Dx(u(t,-1.0)) ~ exp(-t) * (t^2 * sin(-1.0) + 3cos(-1.0)),
          4u(t,1.0) + t * Dx(u(t,1.0)) ~ exp(-t) * (4sin(1.0) + t * cos(1.0))]

    # Space and time domains
    domains = [t ∈ IntervalDomain(0.0,1.0),
               x ∈ IntervalDomain(-1.0,1.0)]

    # PDE system
    pdesys = PDESystem(eq,bcs,domains,[t,x],[u(t,x)])

    # Method of lines discretization
    dx = 0.01
    order = 2
    discretization = MOLFiniteDifference([x=>dx],t)

    # Convert the PDE problem into an ODE problem
    prob = discretize(pdesys,discretization)

    # Solve ODE problem
    sol = solve(prob,Rodas4(),reltol=1e-6,saveat=0.1)

    x = (-1:dx:1)
    t = sol.t

    # Test against exact solution
    for i in 1:length(sol)
       exact = u_exact(x, t[i])
       # Due to structural simplification
       # [u2 -> u(n-1), u(1), u(n)]
       # Will be fixed by sol[u]
       u_approx = [sol.u[i][end-1];sol.u[i][1:end-2];sol.u[i][end]]
       @test u_approx ≈ exact atol=0.05
    end
end

@testset "Test 07: Dt(u(t,r)) ~ 1/r^2 * Dr(r^2 * Dr(u(t,r))) (Spherical Laplacian)" begin
    # Method of Manufactured Solutions
    # general solution of the spherical Laplacian equation
    # satisfies Dr(u(t,0)) = 0
    u_exact = (r,t) -> exp.(-t) * sin.(r) ./ r

    # Parameters, variables, and derivatives
    @parameters t r
    @variables u(..)
    Dt = Differential(t)
    Dr = Differential(r)

    # 1D PDE and boundary conditions

    eq  = Dt(u(t,r)) ~ 1/r^2 * Dr(r^2 * Dr(u(t,r)))
    bcs = [u(0,r) ~ sin(r)/r,
           Dr(u(t,0)) ~ 0,
           u(t,1) ~ exp(-t) * sin(1)]
        #    Dr(u(t,1)) ~ -exp(-t) * sin(1)]

    # Space and time domains
    domains = [t ∈ IntervalDomain(0.0,1.0),
               r ∈ AxisymmetricSphereDomain(0.0,1.0)]

    # PDE system
    pdesys = PDESystem(eq,bcs,domains,[t,r],[u(t,r)])

    # Method of lines discretization
    dr = 0.1
    order = 2
    discretization = MOLFiniteDifference([r=>dr],t)
    prob = discretize(pdesys,discretization)

    # Solve ODE problem
    sol = solve(prob,Tsit5(),saveat=0.1)

    r = (0:dr:1)[2:end-1]
    t = sol.t

    # Test against exact solution
    for i in 1:length(sol)
        exact = u_exact(r, t[i])
        u_approx = sol.u[i]
        @test u_approx ≈ exact atol=0.01
    end
end

@testset "Test 10: linear diffusion, two variables, mixed BCs" begin
    # Method of Manufactured Solutions
    u_exact = (x,t) -> exp.(-t) * cos.(x)
    v_exact = (x,t) -> exp.(-t) * sin.(x)

    # Parameters, variables, and derivatives
    @parameters t x
    @variables u(..) v(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Dx^2

    # 1D PDE and boundary conditions
    eqs = [Dt(u(t,x)) ~ Dxx(u(t,x)),
           Dt(v(t,x)) ~ Dxx(v(t,x))]
    bcs = [u(0,x) ~ cos(x),
           v(0,x) ~ sin(x),
           u(t,0) ~ exp(-t),
           Dx(u(t,1)) ~ -exp(-t) * sin(1),
           Dx(v(t,0)) ~ exp(-t),
           v(t,1) ~ exp(-t) * sin(1)]

    # Space and time domains
    domains = [t ∈ IntervalDomain(0.0,1.0),
               x ∈ IntervalDomain(0.0,1.0)]

    # PDE system
    pdesys = PDESystem(eqs,bcs,domains,[t,x],[u(t,x),v(t,x)])

    # Method of lines discretization
    l = 100
    dx = range(0.0,1.0,length=l)
    order = 2
    discretization = MOLFiniteDifference([x=>dx],t)

    # Convert the PDE problem into an ODE problem
    prob = discretize(pdesys,discretization)

    # Solve ODE problem
    sol = solve(prob,Tsit5(),saveat=0.1)

    x_sol = dx[2:end-1]
    t_sol = sol.t

    # Test against exact solution
    for i in 1:length(sol)
        @test u_exact(x_sol, t_sol[i]) ≈ sol.u[i][1:l-2] atol=0.01
        @test v_exact(x_sol, t_sol[i]) ≈ sol.u[i][l-1:end] atol=0.01
    end
end
