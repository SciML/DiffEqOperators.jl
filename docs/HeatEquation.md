# Solving the Heat Equation using DiffEqOperators

In this tutorial we will solve the famous heat equation using the explicit discretization on a 2D `space x time` grid. The heat equation is:-
        
$$\frac{\partial u}{\partial t} - \frac{{\partial}^2 u}{\partial x^2} = 0$$

For this example we consider a Dirichlet boundary condition with the initial distribution being parabolic. Since we have fixed the value at boundaries (in this case equal), after a long time we expect the 1D rod to be heated in a linear manner.

        julia> using DiffEqOperators, DifferentialEquations, Plots
        julia> x = collect(-pi : 2pi/511 : pi);
        julia> u0 = -(x - 0.5).^2 + 1/12;
        julia> A = DerivativeOperator{Float64}(2,2,2pi/511,512,:Dirichlet,:Dirichlet;BC=(u0[1],u0[end]));

Now solving equation as an ODE we have:-
    
        julia> prob1 = ODEProblem(A, u0, (0.,10.));
        julia> sol1 = solve(prob1, dense=false, tstops=0:0.01:10);
        # try to plot the solution at different time points using
        julia> plot(x, [sol1(i) for i in 0:1:10])

**Note:** Many a times the solver may inform you that the solution is unstable. This problem is usually handled by changing the solver algorithm. Your best bet might be the `CVODE_BDE()` algorithm from the OrdinaryDiffEq.jl suite.