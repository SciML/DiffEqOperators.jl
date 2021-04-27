# Solving the Heat Equation

#### Note: This uses a currently unreleased interface that is still a work in progress. Use at your own risk!

In this tutorial we will use the symbolic interface to solve the heat equation.

### Dirichlet boundary conditions

```julia
using OrdinaryDiffEq, ModelingToolkit, DiffEqOperators
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
pdesys = PDESystem(eq,bcs,domains,[t,x],[u(t,x)])

# Method of lines discretization
dx = 0.1
order = 2
discretization = MOLFiniteDifference([x=>dx],t)

# Convert the PDE problem into an ODE problem
prob = discretize(pdesys,discretization)

# Solve ODE problem
using OrdinaryDiffEq
sol = solve(prob,Tsit5(),saveat=0.2)

# Plot results and compare with exact solution
x = (0:dx:1)[2:end-1]
t = sol.t

using Plots
plt = plot()

for i in 1:length(t)
    plot!(x,sol.u[i],label="Numerical, t=$(t[i])")
    scatter!(x, u_exact(x, t[i]),label="Exact, t=$(t[i])")
end
display(plt)
savefig("plot.png")
```
### Neumann boundary conditions

```julia
using OrdinaryDiffEq, ModelingToolkit, DiffEqOperators
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
pdesys = PDESystem(eq,bcs,domains,[t,x],[u(t,x)])

# Method of lines discretization
# Need a small dx here for accuracy
dx = 0.01
order = 2
discretization = MOLFiniteDifference([x=>dx],t)

# Convert the PDE problem into an ODE problem
prob = discretize(pdesys,discretization)

# Solve ODE problem
using OrdinaryDiffEq
sol = solve(prob,Tsit5(),saveat=0.2)

# Plot results and compare with exact solution
x = (0:dx:1)[2:end-1]
t = sol.t

using Plots
plt = plot()

for i in 1:length(t)
    plot!(x,sol.u[i],label="Numerical, t=$(t[i])",lw=12)
    scatter!(x, u_exact(x, t[i]),label="Exact, t=$(t[i])")
end
display(plt)
savefig("plot.png")
```

### Robin boundary conditions

```julia
using OrdinaryDiffEq, ModelingToolkit, DiffEqOperators
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
pdesys = PDESystem(eq,bcs,domains,[t,x],[u(t,x)])

# Method of lines discretization
# Need a small dx here for accuracy
dx = 0.05
order = 2
discretization = MOLFiniteDifference([x=>dx],t)

# Convert the PDE problem into an ODE problem
prob = discretize(pdesys,discretization)

# Solve ODE problem
using OrdinaryDiffEq
sol = solve(prob,Tsit5(),saveat=0.2)

# Plot results and compare with exact solution
x = (0:dx:1)[2:end-1]
t = sol.t

using Plots
plt = plot()

for i in 1:length(t)
    plot!(x,sol.u[i],label="Numerical, t=$(t[i])")
    scatter!(x, u_exact(x, t[i]),label="Exact, t=$(t[i])")
end
display(plt)
savefig("plot.png")
```

### Adding parameters

We can also build up more complicated systems with multiple dependent variables and parameters as follows

```julia
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

domains = [t ∈ IntervalDomain(0.0,1.0),
           x ∈ IntervalDomain(0.0,1.0)]

pdesys = PDESystem(eqs,bcs,domains,[t,x],[u(t,x),v(t,x)],[Dn=>0.5, Dp=>2])
discretization = MOLFiniteDifference([x=>0.1],t)
prob = discretize(pdesys,discretization) # This gives an ODEProblem since it's time-dependent
sol = solve(prob,Tsit5())

x = (0:0.1:1)[2:end-1]
t = sol.t

using Plots
anim = @animate for i in 1:length(t)
       p1 = plot(x,sol.u[i][1:9],label="u, t=$(t[i])";legend=false,xlabel="x",ylabel="u",ylim=[0,1])
       p2 = plot(x,sol.u[i][10:end],label="v, t=$(t[i])";legend=false,xlabel="x",ylabel="v",ylim=[0,1])
       plot(p1,p2)
end
gif(anim, "plot.gif",fps=30)
```