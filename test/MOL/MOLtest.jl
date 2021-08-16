using ModelingToolkit, DiffEqOperators, LinearAlgebra, OrdinaryDiffEq
using ModelingToolkit: operation, istree, arguments
using DomainSets

# Define some variables
@parameters t x
@variables u(..) v(..)
Dt = Differential(t)
Dx = Differential(x)
Dxx = Differential(x)^2
eqs  = [Dt(u(t,x)) ~ Dxx(u(t,x)), 
        Dt(v(t,x)) ~ Dxx(v(t,x))]
bcs = [u(0,x) ~ - x * (x-1) * sin(x),
       v(0,x) ~ - x * (x-1) * sin(x),
       u(t,0) ~ 0.0, u(t,1) ~ 0.0,
       v(t,0) ~ 0.0, v(t,1) ~ 0.0]

domains = [t ∈ Interval(0.0,1.0),
           x ∈ Interval(0.0,1.0)]

@named pdesys = PDESystem(eqs,bcs,domains,[t,x],[u(t,x),v(t,x)])
discretization = MOLFiniteDifference([x=>0.1],t;grid_align=edge_align)
prob = discretize(pdesys,discretization) # This gives an ODEProblem since it's time-dependent
sol = solve(prob,Tsit5())

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

# 3D PDE
eq  = Dt(u(t,x,y)) ~ Dxx(u(t,x,y)) + Dyy(u(t,x,y))

analytic_sol_func(t,x,y) = exp(x+y)*cos(x+y+4t)
# Initial and boundary conditions
bcs = [u(t_min,x,y) ~ analytic_sol_func(t_min,x,y),
       u(t,x_min,y) ~ analytic_sol_func(t,x_min,y),
       u(t,x_max,y) ~ analytic_sol_func(t,x_max,y),
       u(t,x,y_min) ~ analytic_sol_func(t,x,y_min),
       u(t,x,y_max) ~ analytic_sol_func(t,x,y_max)]

# Space and time domains
domains = [t ∈ Interval(t_min,t_max),
           x ∈ Interval(x_min,x_max),
           y ∈ Interval(y_min,y_max)]
@named pdesys = PDESystem([eq],bcs,domains,[t,x,y],[u(t,x,y)])

# Method of lines discretization
dx = 0.1; dy = 0.2
discretization = MOLFiniteDifference([x=>dx,y=>dy],t)
prob = ModelingToolkit.discretize(pdesys,discretization)
sol = solve(prob,Tsit5())

# Diffusion in a sphere
@parameters t r
@variables u(..)
Dt = Differential(t)
Dr = Differential(r)
Drr = Dr^2
eq  = Dt(u(t,r)) ~ 0.6 * (1/r^2 * Dr(r^2 * Dr(u(t,r))))
bcs = [u(0,r) ~ - r * (r-1) * sin(r),
       Dr(u(t,0)) ~ 0.0, u(t,1) ~ sin(1)]

domains = [t ∈ Interval(0.0,1.0),
           r ∈ Interval(0.0,1.0)]

@named pdesys = PDESystem(eq,bcs,domains,[t,r],[u(t,r)])
discretization = MOLFiniteDifference([r=>0.1],t)
prob = discretize(pdesys,discretization) # This gives an ODEProblem since it's time-dependent
sol = solve(prob,Tsit5())
