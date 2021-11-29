# 1D diffusion problem

# Packages and inclusions
using ModelingToolkit,DiffEqOperators,LinearAlgebra,Test,OrdinaryDiffEq, DomainSets
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
    domains = [t ∈ Interval(0.0,1.0),
               x ∈ Interval(0.0,Float64(π))]

    # PDE system
    @named pdesys = PDESystem(eq,bcs,domains,[t,x],[u(t,x)])

    # Method of lines discretization
    dx = range(0.0,Float64(π),length=30)
    order = 2
    discretization = MOLFiniteDifference([x=>dx],t)
    discretization_edge = MOLFiniteDifference([x=>dx],t;grid_align=edge_align)
    # Explicitly specify order of centered difference
    discretization_centered = MOLFiniteDifference([x=>dx],t;centered_order=order)
    # Higher order centered difference
    discretization_centered_order4 = MOLFiniteDifference([x=>dx],t;centered_order=4)

    for disc in [discretization, discretization_edge, discretization_centered, discretization_centered_order4]
        # Convert the PDE problem into an ODE problem
        prob = discretize(pdesys,disc)

        # Solve ODE problem
        sol = solve(prob,Tsit5(),saveat=0.1)

        if disc.grid_align == center_align
            x = dx[2:end-1]
        else
            x = (dx[1:end-1]+dx[2:end])/2
        end
        t = sol.t

        # Test against exact solution
        for i in 1:length(sol)
            exact = u_exact(x, t[i])
            u_approx = sol.u[i]
            @test all(isapprox.(u_approx, exact, atol=0.01))
        end
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
    domains = [t ∈ Interval(0.0,1.0),
               x ∈ Interval(0.0,1.0)]

    # PDE system
    @named pdesys = PDESystem(eq,bcs,domains,[t,x],[u(t,x)],[D=>10.0])

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
    domains = [t ∈ Interval(0.0,1.0),
               x ∈ Interval(0.0,1.0)]

    # PDE system
    @named pdesys = PDESystem(eq,bcs,domains,[t,x],[u(t,x)])

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

@testset "Test 03: Dt(u(t,x)) ~ Dxx(u(t,x)), homogeneous Neumann BCs" begin
    # Method of Manufactured Solutions
    u_exact = (x,t) -> exp.(-t) * cos.(x)

    # Parameters, variables, and derivatives
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Differential(x)^2

    # 1D PDE and boundary conditions
    eq  = Dt(u(t,x)) ~ Dxx(u(t,x))
    bcs = [u(0,x) ~ cos(x),
           Dx(u(t,0)) ~ 0,
           Dx(u(t,Float64(pi))) ~ 0]

    # Space and time domains
    domains = [t ∈ Interval(0.0,1.0),
               x ∈ Interval(0.0,Float64(pi))]

    # PDE system
    @named pdesys = PDESystem(eq,bcs,domains,[t,x],[u(t,x)])

    # Method of lines discretization
    dx = range(0.0,Float64(π),length=300)
    order = 2
    discretization = MOLFiniteDifference([x=>dx],t)
    discretization_edge = MOLFiniteDifference([x=>dx],t;grid_align=center_align)

    # Convert the PDE problem into an ODE problem
    for disc in [discretization, discretization_edge]
        prob = discretize(pdesys,disc)

        # Solve ODE problem
        sol = solve(prob,Tsit5(),saveat=0.1)

        if disc.grid_align == center_align
            x_sol = dx[2:end-1]
        else
            x_sol = (dx[1:end-1]+dx[2:end])/2
        end
        t_sol = sol.t

        # Test against exact solution
        for i in 1:length(sol)
            exact = u_exact(x_sol, t_sol[i])
            u_approx = sol.u[i]
            @test all(isapprox.(u_approx, exact, atol=0.01))
            @test sum(u_approx) ≈ 0 atol=1e-10
        end
    end
end

@testset "Test 03a: Dt(u(t,x)) ~ Dxx(u(t,x)), Neumann BCs" begin
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
    domains = [t ∈ Interval(0.0,1.0),
               x ∈ Interval(0.0,Float64(pi))]

    # PDE system
    @named pdesys = PDESystem(eq,bcs,domains,[t,x],[u(t,x)])

    # Method of lines discretization
    dx = range(0.0,Float64(π),length=30)
    order = 2
    discretization = MOLFiniteDifference([x=>dx],t)
    discretization_edge = MOLFiniteDifference([x=>dx],t;grid_align=edge_align)

    # Convert the PDE problem into an ODE problem
    for disc ∈ [discretization, discretization_edge]
        prob = discretize(pdesys,disc)

        # Solve ODE problem
        sol = solve(prob,Tsit5(),saveat=0.1)

        if disc.grid_align == center_align
            x = dx[2:end-1]
        else
            x = (dx[1:end-1]+dx[2:end])/2
        end
        t = sol.t

        # Test against exact solution
        # exact integral based on Neumann BCs
        integral_u_exact = t -> sum(sol.u[1] * dx[2]) + 2 * (exp(-t) - 1)
        for i in 1:length(sol)
            exact = u_exact(x, t[i])
            u_approx = sol.u[i]
            @test all(isapprox.(u_approx, exact, atol=0.01))
            # test mass conservation
            integral_u_approx = sum(u_approx * dx[2])
            @test integral_u_exact(t[i]) ≈ integral_u_approx atol=1e-13
        end
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
    domains = [t ∈ Interval(0.0,1.0),
               x ∈ Interval(0.0,Float64(pi))]

    # PDE system
    @named pdesys = PDESystem(eq,bcs,domains,[t,x],[u(t,x)])

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
        @test all(isapprox.(u_approx, exact, atol=0.01))
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
    domains = [t ∈ Interval(0.0,1.0),
               x ∈ Interval(-1.0,1.0)]

    # PDE system
    @named pdesys = PDESystem(eq,bcs,domains,[t,x],[u(t,x)])

    # Method of lines discretization
    dx = 0.01
    order = 2
    discretization = MOLFiniteDifference([x=>dx],t)
    discretization_edge = MOLFiniteDifference([x=>dx],t;grid_align=edge_align)

    for disc ∈ [discretization, discretization_edge]
        # Convert the PDE problem into an ODE problem
        prob = discretize(pdesys,disc)

        # Solve ODE problem
        sol = solve(prob,Tsit5(),saveat=0.1)
        x = (-1:dx:1)
        if disc.grid_align == center_align
            x = x[2:end-1]
        else
            x = (x[1:end-1].+x[2:end])/2
        end
        t = sol.t

        # Test against exact solution
        for i in 1:length(sol)
            exact = u_exact(x, t[i])
            u_approx = sol.u[i]
            @test all(isapprox.(u_approx, exact, atol=0.1))
        end
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
    domains = [t ∈ Interval(0.0,1.0),
               x ∈ Interval(-1.0,1.0)]

    # PDE system
    @named pdesys = PDESystem(eq,bcs,domains,[t,x],[u(t,x)])

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
       @test all(isapprox.(u_approx, exact, atol=0.01))
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
    domains = [t ∈ Interval(0.0,1.0),
               r ∈ Interval(0.0,1.0)]

    # PDE system
    @named pdesys = PDESystem(eq,bcs,domains,[t,r],[u(t,r)])

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
        @test all(isapprox.(u_approx, exact, atol=0.01))
    end
end

@testset "Test 08: Dt(u(t,r)) ~ 4/r^2 * Dr(r^2 * Dr(u(t,r))) (Spherical Laplacian)" begin
    # Method of Manufactured Solutions
    # general solution of the spherical Laplacian equation
    # satisfies Dr(u(t,0)) = 0
    u_exact = (r,t) -> exp.(-4t) * sin.(r) ./ r

    # Parameters, variables, and derivatives
    @parameters t r
    @variables u(..)
    Dt = Differential(t)
    Dr = Differential(r)

    # 1D PDE and boundary conditions

    eq  = Dt(u(t,r)) ~ 4/r^2 * Dr(r^2 * Dr(u(t,r)))
    bcs = [u(0,r) ~ sin(r)/r,
           Dr(u(t,0)) ~ 0,
           u(t,1) ~ exp(-4t) * sin(1)]

    # Space and time domains
    domains = [t ∈ Interval(0.0,1.0),
               r ∈ Interval(0.0,1.0)]

    # PDE system
    @named pdesys = PDESystem(eq,bcs,domains,[t,r],[u(t,r)])

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
        @test all(isapprox.(u_approx, exact, atol=0.01))
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
    domains = [t ∈ Interval(0.0,1.0),
               x ∈ Interval(0.0,1.0)]

    # PDE system
    @named pdesys = PDESystem(eqs,bcs,domains,[t,x],[u(t,x),v(t,x)])

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
        @test all(isapprox.(u_exact(x_sol, t_sol[i]), sol.u[i][1:l-2], atol=0.01))
        @test all(isapprox.(v_exact(x_sol, t_sol[i]), sol.u[i][l-1:end], atol=0.01))
    end
end

@testset "Test 11: linear diffusion, two variables, mixed BCs, with parameters" begin
    @parameters t x
    @parameters Dn, Dp
    @variables u(..) v(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Differential(x)^2

    eqs  = [Dt(u(t,x)) ~ Dn * Dxx(u(t,x)) + u(t,x)*v(t,x),
            Dt(v(t,x)) ~ Dp * Dxx(v(t,x)) - u(t,x)*v(t,x)]
    bcs = [u(0,x) ~ sin(pi*x/2),
        v(0,x) ~ sin(pi*x/2),
        u(t,0) ~ 0.0, Dx(u(t,1)) ~ 0.0,
        v(t,0) ~ 0.0, Dx(v(t,1)) ~ 0.0]

    domains = [t ∈ Interval(0.0,1.0),
            x ∈ Interval(0.0,1.0)]

    @named pdesys = PDESystem(eqs,bcs,domains,[t,x],[u(t,x),v(t,x)],[Dn=>0.5, Dp=>2])
    discretization = MOLFiniteDifference([x=>0.1],t)
    prob = discretize(pdesys,discretization)
    @test prob.p == [0.5,2]
    # Make sure it can be solved
    sol = solve(prob,Tsit5())
end

@testset "Test 12: linear diffusion, two variables, mixed BCs, different independent variables" begin
    # Method of Manufactured Solutions
    u_exact = (x,t) -> exp.(-t) * cos.(x)
    v_exact = (y,t) -> exp.(-t) * sin.(y)

    # Parameters, variables, and derivatives
    @parameters t x y
    @variables u(..) v(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Dx^2
    Dy = Differential(y)
    Dyy = Dy^2

    # 1D PDE and boundary conditions
    eqs = [Dt(u(t,x)) ~ Dxx(u(t,x)),
           Dt(v(t,y)) ~ Dyy(v(t,y))]
    bcs = [u(0,x) ~ cos(x),
           v(0,y) ~ sin(y),
           u(t,0) ~ exp(-t),
           Dx(u(t,1)) ~ -exp(-t) * sin(1),
           Dy(v(t,0)) ~ exp(-t),
           v(t,2) ~ exp(-t) * sin(2)]

    # Space and time domains
    domains = [t ∈ Interval(0.0,1.0),
               x ∈ Interval(0.0,1.0),
               y ∈ Interval(0.0,2.0)]

    # PDE system
    @named pdesys = PDESystem(eqs,bcs,domains,[t,x,y],[u(t,x),v(t,y)])

    # Method of lines discretization
    l = 100
    dx = range(0.0,1.0,length=l)
    dy = range(0.0,2.0,length=l)
    order = 2
    discretization = MOLFiniteDifference([x=>dx,y=>dy],t)

    # Convert the PDE problem into an ODE problem
    prob = discretize(pdesys,discretization)

    # Solve ODE problem
    sol = solve(prob,Tsit5(),saveat=0.1)

    x_sol = dx[2:end-1]
    y_sol = dy[2:end-1]
    t_sol = sol.t

    # Test against exact solution
    for i in 1:length(sol)
        @test all(isapprox.(u_exact(x_sol, t_sol[i]), sol.u[i][1:l-2], atol=0.01))
        @test all(isapprox.(v_exact(y_sol, t_sol[i]), sol.u[i][l-1:end], atol=0.01))
    end
end

@testset "Test 13: one linear diffusion with mixed BCs, one ODE" begin
    # Method of Manufactured Solutions
    u_exact = (x,t) -> exp.(-t) * sin.(x)
    v_exact = (t) -> exp.(-t)

    @parameters t x
    @variables u(..) v(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Dx^2

    # 1D PDE and boundary conditions
    eqs = [Dt(u(t,x)) ~ Dxx(u(t,x)),
           Dt(v(t)) ~ -v(t)]
    bcs = [u(0,x) ~ sin(x),
           v(0) ~ 1,
           u(t,0) ~ 0,
           Dx(u(t,1)) ~ exp(-t) * cos(1)]

    # Space and time domains
    domains = [t ∈ Interval(0.0,1.0),
               x ∈ Interval(0.0,1.0)]

    # PDE system
    @named pdesys = PDESystem(eqs,bcs,domains,[t,x],[u(t,x),v(t)])

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
        @test all(isapprox.(u_exact(x_sol, t_sol[i]), sol.u[i][1:length(x_sol)], atol=0.01))
        @test v_exact(t_sol[i]) ≈ sol.u[i][end] atol=0.01
    end
end

@testset "Test error 01: Test Invalid Centered Order" begin
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
    domains = [t ∈ Interval(0.0,1.0),
               x ∈ Interval(0.0,Float64(π))]

    # PDE system
    @named pdesys = PDESystem(eq,bcs,domains,[t,x],[u(t,x)])

    # Method of lines discretization
    dx = range(0.0,Float64(π),length=30)

    # Explicitly specify and invalid order of centered difference
    for order in 1:6
        discretization = MOLFiniteDifference([x=>dx],t;centered_order=order)
        if order % 2 != 0
            @test_throws ArgumentError discretize(pdesys,discretization)
        else
            discretize(pdesys,discretization)
        end
    end
end
