# 1D linear convection problem

# Packages and inclusions
using ModelingToolkit,DiffEqOperators,DiffEqBase,LinearAlgebra,Test

# Parameters, variables, and derivatives
@parameters t x
@variables u(..)
@derivatives Dt'~t
@derivatives Dx'~x

# 1D PDE and boundary conditions
eq  = Dt(u(t,x)) ~ 1.1 * Dx(u(t,x))
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

# Test
x_interval = domains[2].domain.lower+dx:dx:domains[2].domain.upper-dx
u = @. (0.5/(0.2*sqrt(2.0*3.1415)))*exp(-(x_interval-(0.75+1.1*0.6))^2/(2.0*0.2^2))
t_f = size(sol)[2]
@test sol[t_f] ≈ u atol = 0.1;

# Plot and save results
#using Plots
#plot(prob.space,Array(prob.extrapolation*sol[1]))
#plot!(prob.space,Array(prob.extrapolation*sol[2]))
#plot!(prob.space,Array(prob.extrapolation*sol[3]))
#plot!(prob.space,Array(prob.extrapolation*sol[4]))
#plot!(prob.space,Array(prob.extrapolation*sol[5]))
#plot!(prob.space,Array(prob.extrapolation*sol[6]))
#savefig("MOL_1D_Linear_Convection.png")
