# Examples

## Heat equation

### Dirichlet boundary conditions

```julia
using ModelingToolkit, DiffEqOperators
# Method of Manufactured Solutions: exact solution
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
        u(t,1) ~ exp(-t) * cos(1)]

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
sol = solve(prob,Tsit5(),saveat=0.2)

# Plot results and compare with exact solution
x = prob.space[2]
t = sol.t

using Plots
plt = plot()
for i in 1:length(t)
    plot!(x,Array(prob.extrapolation[1](t[i])*sol.u[i]),label="Numerical, t=$(t[i])")
    scatter!(x, u_exact(x, t[i]),label="Exact, t=$(t[i])")
end
display(plt)
```
### Neumann boundary conditions

```julia
using ModelingToolkit, DiffEqOperators
# Method of Manufactured Solutions: exact solution
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
        Dx(u(t,0)) ~ 0.0,
        Dx(u(t,1)) ~ -exp(-t) * sin(1)]

# Space and time domains
domains = [t ∈ IntervalDomain(0.0,1.0),
        x ∈ IntervalDomain(0.0,1.0)]

# PDE system
pdesys = PDESystem(eq,bcs,domains,[t,x],[u])

# Method of lines discretization
# Need a small dx here for accuracy
dx = 0.01
order = 2
discretization = MOLFiniteDifference(dx,order)

# Convert the PDE problem into an ODE problem
prob = discretize(pdesys,discretization)

# Solve ODE problem
using OrdinaryDiffEq
sol = solve(prob,Tsit5(),saveat=0.2)

# Plot results and compare with exact solution
x = prob.space[2]
t = sol.t

using Plots
plt = plot()
for i in 1:length(t)
    plot!(x,Array(prob.extrapolation[1](t[i])*sol.u[i]),label="Numerical, t=$(t[i])")
    scatter!(x, u_exact(x, t[i]),label="Exact, t=$(t[i])")
end
display(plt)
```

### Robin boundary conditions

```julia
using ModelingToolkit, DiffEqOperators 
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
        u(t,1.0) + Dx(u(t,1.0)) ~ exp(-t) * (sin(1.0) + cos(1.0))]

# Space and time domains
domains = [t ∈ IntervalDomain(0.0,1.0),
        x ∈ IntervalDomain(-1.0,1.0)]

# PDE system
pdesys = PDESystem(eq,bcs,domains,[t,x],[u])

# Method of lines discretization
# Need a small dx here for accuracy
dx = 0.05
order = 2
discretization = MOLFiniteDifference(dx,order)

# Convert the PDE problem into an ODE problem
prob = discretize(pdesys,discretization)

# Solve ODE problem
using OrdinaryDiffEq
sol = solve(prob,Tsit5(),saveat=0.2)

# Plot results and compare with exact solution
x = prob.space[2]
t = sol.t

using Plots
plt = plot()
for i in 1:length(t)
    plot!(x,Array(prob.extrapolation[1](t[i])*sol.u[i]),label="Numerical, t=$(t[i])")
    scatter!(x, u_exact(x, t[i]),label="Exact, t=$(t[i])")
end
display(plt)
```
