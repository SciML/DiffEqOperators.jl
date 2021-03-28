using ModelingToolkit, DiffEqOperators, LinearAlgebra

# Define some variables
@parameters t x
@variables u(..)
Dt = Differential(t)
Dxx = Differential(x)^2
eq  = Dt(u(t,x)) ~ Dxx(u(t,x))
bcs = [u(0,x) ~ - x * (x-1) * sin(x),
       u(t,0) ~ 0.0, u(t,1) ~ 0.0]

domains = [t ∈ IntervalDomain(0.0,1.0),
           x ∈ IntervalDomain(0.0,1.0)]

pdesys = PDESystem(eq,bcs,domains,[t,x],[u(t,x)])
discretization = MOLFiniteDifference([x=>0.1],t)
prob = discretize(pdesys,discretization) # This gives an ODEProblem since it's time-dependent

using OrdinaryDiffEq
sol = solve(prob,Tsit5())
