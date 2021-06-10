# 2D linear convection-diffusion problem

# Packages and inclusions
using ModelingToolkit, DiffEqOperators, LinearAlgebra, Test, OrdinaryDiffEq
using ModelingToolkit: Interval, infimum, supremum


# Tests
@testset "Test 00: Dt(u(t, x, y)) ~  0.5 Dxx(u(t, x, y)) + 0.5 Dyy(u(t, x, y))
          -5.0 Dx(u(t, x, y)) - 5.0 Dy(u(t, x, y))" begin

    # Parameters, variables, and derivatives
    @parameters t x y
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Differential(x)^2
    Dy = Differential(y)
    Dyy = Differential(y)^2
    t_min= 0.0
    t_max = 0.12
    x_min = 0.0
    x_max = 1.0
    y_min = 0.0
    y_max = 1.0
    
    # Analytic solution
    # https://nbuckman.scripts.mit.edu:444/homepage/wp-content/uploads/2016/03/Convection-Diffusion-Paper-PDF.pdf
    dx = dy = 0.05 # this solution requires calculating an integral and an infinite serie
    A(m, n) = 4.0 * sum([exp(-5 * (x + y)) * sin(π * x) * sin(π * y) * sin(m * π * x) *
                         sin(n * π * y) * dx * dy for x = dx/2.0:dx:1.0, y = dy/2.0:dy:1.0])
    analytic_sol_func(t, x, y) = sum([A(m, n) * exp((-0.5 * m^2 * π^2 - 0.5 * n^2 * π^2 - 25.0) * t) *
                                      exp(5.0 * (x + y)) * sin(m * π * x) * sin(n * π * y)
                                      for m = 1:30, n = 1:30])

    # 1D PDE and boundary conditions
    eq  = Dt(u(t, x, y)) ~  0.5 * Dxx(u(t, x, y)) + 0.5 * Dyy(u(t, x, y))
                           -5.0 * Dx(u(t, x, y)) - 5.0 * Dy(u(t, x, y))
    bcs = [u(t_min, x, y) ~ sin(π * x) * sin(π * y),
           u(t, x_min, y) ~ 0.0,
           u(t, x_max, y) ~ 0.0,
           u(t, x, y_min) ~ 0.0,
           u(t, x, y_max) ~ 0.0]

    # Space and time domains
    domains = [t ∈ Interval(t_min,t_max),
               x ∈ Interval(x_min,x_max),
               y ∈ Interval(y_min,y_max)]

    # PDE system
    pdesys = PDESystem(eq, bcs, domains, [t, x, y], [u(t, x, y)])

    # Method of lines discretization
    dx = dy = 0.0217
    discretization = MOLFiniteDifference([x => dx, y => dy], t)
    prob = discretize(pdesys, discretization)

    # Solve ODE problem
    sol = solve(prob, Tsit5())
    
    # Test against exact solution
    dx = dy = 0.05
    Nx = floor(Int64, (x_max - x_min) / dx) + 1
    Ny = floor(Int64, (y_max - y_min) / dy) + 1
    @variables u[1:Nx,1:Ny](t)
    sol′ = reshape([sol[u[(i-1)*Ny+j]][end] for i in 1:Nx for j in 1:Ny],(Nx,Ny))
    asf = reshape([analytic_sol_func(t_max,x,y)
                   for y = y_min:dy:y_max for x = x_min:dx:x_max],(Nx,Ny))
    m = max(asf...)
    @test asf / m ≈ sol′ / m  atol=0.1 

    #Plot
#    using Plots
#    heatmap(sol′)
#    savefig("MOL_2D_Linear_Convection_Diffusion_Test00.png")
#    heatmap(asf)
#    savefig("MOL_2D_Linear_Convection_Diffusion_analytic_sol_Test00.png")

end

