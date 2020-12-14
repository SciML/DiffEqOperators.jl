# Examples

## Heat equation

```julia
using ModelingToolkit, DiffEqOperators

# Parameters, variables, and derivatives
@parameters t x
@variables u(..)
@derivatives Dt'~t
@derivatives Dxx''~x

# 1D PDE and boundary conditions
eq  = Dt(u(t,x)) ~ Dxx(u(t,x))
bcs = [u(0,x) ~ -x*(x-1)*sin(x),
        u(t,0) ~ 0.0,
        u(t,1) ~ 0.0]

# Space and time domains
domains = [t ∈ IntervalDomain(0.0,1.0),
            x ∈ IntervalDomain(0.0,1.0)]

# PDE system
pdesys = PDESystem(eq,bcs,domains,[t,x],[u])

# Method of lines discretization
dx = 0.1
order = 2
discretization = MOLFiniteDifference(dx,order)

# Convert the PDE problem into an ODE problem
prob = discretize(pdesys,discretization)

# Solve ODE problem
using OrdinaryDiffEq
sol = solve(prob,Tsit5(),saveat=0.1)
```