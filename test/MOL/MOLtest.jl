using ModelingToolkit, DiffEqOperators, LinearAlgebra, OrdinaryDiffEq

# Define some variables
@parameters t x
@variables u(..) v(..)
Dt = Differential(t)
Dxx = Differential(x)^2
eqs  = [Dt(u(t,x)) ~ Dxx(u(t,x)), 
        Dt(v(t,x)) ~ Dxx(v(t,x))]
bcs = [u(0,x) ~ - x * (x-1) * sin(x),
       v(0,x) ~ - x * (x-1) * sin(x),
       u(t,0) ~ 0.0, u(t,1) ~ 0.0,
       v(t,0) ~ 0.0, v(t,1) ~ 0.0]

domains = [t ∈ IntervalDomain(0.0,1.0),
           x ∈ IntervalDomain(0.0,1.0)]

pdesys = PDESystem(eqs,bcs,domains,[t,x],[u(t,x),v(t,x)])
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
domains = [t ∈ IntervalDomain(t_min,t_max),
           x ∈ IntervalDomain(x_min,x_max),
           y ∈ IntervalDomain(y_min,y_max)]
pdesys = PDESystem([eq],bcs,domains,[t,x,y],[u(t,x,y)])

# Method of lines discretization
dx = 0.1; dy = 0.2
discretization = MOLFiniteDifference([x=>dx,y=>dy],t)
prob = ModelingToolkit.discretize(pdesys,discretization)
sol = solve(prob,Tsit5())

using ModelingToolkit
using IfElse

# MTK model
# ('negative electrode', 'separator', 'positive electrode') -> x
@parameters t x
# 'Electrolyte concentration ' -> c_e
# 'Electrolyte potential' -> phi_e
@variables c_e(..) phi_e(..)
Dt = Differential(t)
Dx = Differential(x)

# 'Electrolyte concentration ' equation

function concatenation(n, s, p)
   # A concatenation in the electrolyte domain
  f= (x) -> IfElse.ifelse(
      x < 0.4444444444444445, n, IfElse.ifelse(
         x < 0.5555555555555556, s, p
      )
   )
end

cache_5101060308695467050(x) = concatenation(2.25, 0.0, -2.25)(x)
@register cache_5101060308695467050(x)
cache_8771569224475106856 = (Dx(Dx(c_e(t, x)))) + cache_5101060308695467050(x)

# 'Electrolyte potential' equation
cache_5101060308695467050(x) = concatenation(2.25, 0.0, -2.25)(x)
@register cache_5101060308695467050(x)
# cache_4483180157687090897 = (Dx(1 / c_e(t, x) * Dx(c_e(t, x))) - Dx(Dx(phi_e(t, x)))) - cache_5101060308695467050(x)
cache_4483180157687090897 = Dx(1/c_e(t,x) * Dx(c_e(t,x))) - Dx(Dx(phi_e(t, x))) - cache_5101060308695467050(x)


eqs = [
   Dt(c_e(t, x)) ~ cache_8771569224475106856,
   0 ~ cache_4483180157687090897,
]

ics_bcs = [
   # initial conditions
   c_e(0, x) ~ 1.0,
   phi_e(0, x) ~ 0.0,
   # boundary conditions
   Dx(c_e(t, 0.0)) ~ 0.0,
   Dx(c_e(t, 1.0)) ~ 0.0,
   phi_e(t, 0.0) ~ 0.0,
   Dx(phi_e(t, 1.0)) ~ 0.0,
]

t_domain = IntervalDomain(0.0, 3600.0)
x_domain = IntervalDomain(0.0, 1.0)

domains = [
   t in t_domain,
   x in x_domain,
]
ind_vars = [t, x]
dep_vars = [c_e(t,x), phi_e(t,x)]

reduced_c_phi_pde_system = PDESystem(eqs, ics_bcs, domains, ind_vars, dep_vars)


# Finite difference solution
using DiffEqOperators
using OrdinaryDiffEq

ln = 4/9
ls = 1/9
dx = vcat(range(0,ln,length=11)[1:end-1], range(ln,ln+ls,length=11)[1:end-1], range(ln+ls,1,length=11))
discretization = MOLFiniteDifference([x=>dx],t; grid_align=edge_align)

prob = discretize(reduced_c_phi_pde_system,discretization) # This gives an ODEProblem since it's time-dependent
sol = solve(prob,Rodas4())
# using BenchmarkTools
# @btime solve(prob,KenCarp47();saveat=t_pb)

@variables c_e[1:length(dx)](..) phi_e[1:length(dx)](..)
sol[phi_e[end](t)]