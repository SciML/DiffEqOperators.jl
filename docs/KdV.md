# Solving the Heat Equation using DiffEqOperators

In this tutorial we will try to solve the famous **KdV equation** which describes the motion of waves on shallow water surfaces.
The equation is commonly written as 

        $$\frac{\partial u}{\partial t} + \frac{{\partial}^3 u}{\partial t^3} - 6*u*\frac{\partial u}{\partial t} = 0$.

Lets consider the cosine wave as the initial waveform and evolve it using the equation with a [periodic boundary condition](https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.15.240). This example is taken from [here](https://en.wikipedia.org/wiki/Korteweg%E2%80%93de_Vries_equation).

    using DiffEqOperators, DifferentialEquations, Plots
    x = collect(0 : 1/99 : 2);
    u0 = cos.(π*x);
    du3 = zeros(u0); # a container array
    function KdV(t, u, du)
        C(t,u,du3)
        A(t, u, du)
        copy!(du, -u.*du .- 0.022^2.*du3)
    end

Now defining our DiffEqOperators
```
    A = DerivativeOperator{Float64}(1,2,1/99,199,:periodic,:periodic);
    C = DerivativeOperator{Float64}(3,2,1/99,199,:periodic,:periodic);
```

Now call the ODE solver as follows:-

    prob1 = ODEProblem(KdV, u0, (0.,5.));
    sol1 = solve(prob1, dense=false, tstops=0:0.01:10);

and plot the solutions to see the waveform evolve with time.
**Note:** The waveform being solved for here is non-directional unlike the many waves you might see like traveling solitons. In that case you might need to use the Upwind operators.