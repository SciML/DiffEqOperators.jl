# 2D diffusion problem

# Packages and inclusions
using ModelingToolkit,DiffEqOperators,LinearAlgebra,Test,OrdinaryDiffEq,Sundials
using ModelingToolkit: Differential

# Tests
@testset "Test 00: Dt(u(t,x,y)) ~ Dxx(u(t,x,y)) + Dyy(u(t,x,y))" begin

    # Variables, parameters, and derivatives
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

    # Analytic solution
    analytic_sol_func(t,x,y) = exp(x+y)*cos(x+y+4t)
    
    # Equation
    eq  = Dt(u(t,x,y)) ~ Dxx(u(t,x,y)) + Dyy(u(t,x,y))

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

    # Space and time domains
    pdesys = PDESystem([eq],bcs,domains,[t,x,y],[u(t,x,y)])

    dx = 0.1; dy = 0.2
    for order in [2,4,6]
        # Method of lines discretization
        discretization = MOLFiniteDifference([x=>dx,y=>dy],t;centered_order=order)
        prob = ModelingToolkit.discretize(pdesys,discretization)
        
        # Solution of the ODE system
        sol = solve(prob,Tsit5())

        # Test against exact solution
        # TODO: do this properly when sol[u] with reshape etc works
        @test sol.u[1][1] ≈ analytic_sol_func(sol.t[1],0.1,0.2)
        @test sol.u[1][2] ≈ analytic_sol_func(sol.t[1],0.2,0.2)

    end
end

@testset "Test 01: Dt(u(t,x,y)) ~ Dx( a(x,y,u) * Dx(u(t,x,y))) + Dy( a(x,y,u) * Dy(u(t,x,y)))" begin

    # Variables, parameters, and derivatives
    @parameters t x y
    @variables u(..)
    Dx = Differential(x)
    Dy = Differential(y)
    Dt = Differential(t)
    t_min= 0.
    t_max = 2.0
    x_min = 0.
    x_max = 2.
    y_min = 0.
    y_max = 2.

    # Analytic solution
    analytic_sol_func(t,x,y) = exp(x+y)*cos(x+y+4t)
    
    # Equation
    eq = Dt(u(t,x,y)) ~ Dx( (u(t,x,y)^2 / exp(x+y)^2 + sin(x+y+4t)^2)^0.5 * Dx(u(t,x,y))) +
                        Dy( (u(t,x,y)^2 / exp(x+y)^2 + sin(x+y+4t)^2)^0.5 * Dy(u(t,x,y)))

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

    # Space and time domains
    pdesys = PDESystem([eq],bcs,domains,[t,x,y],[u(t,x,y)])

    dx = 0.1; dy = 0.2
    for order in [2,4,6]
        # Method of lines discretization
        discretization = MOLFiniteDifference([x=>dx,y=>dy],t;centered_order=order)
        prob = ModelingToolkit.discretize(pdesys,discretization)
        
        # Solution of the ODE system
        sol = solve(prob,Rosenbrock23())

        # Test against exact solution
        # TODO: do this properly when sol[u] with reshape etc works
        @test sol.u[1][1] ≈ analytic_sol_func(sol.t[1],0.1,0.2)
        @test sol.u[1][2] ≈ analytic_sol_func(sol.t[1],0.2,0.2)

    end
end

