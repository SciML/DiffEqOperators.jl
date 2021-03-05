using Test, DiffEqOperators
using OrdinaryDiffEq

@testset "Parabolic Heat Equation with Dirichlet BCs" begin
    x = collect(-pi : 2pi/511 : pi)
    u0 = @. -(x - 0.5).^2 + 1/12
    @. u_analytic(x)  = -(x .- 0.5).^2 + 1/12
    A = CenteredDifference(2,2,2π/511,512);
    bc = DirichletBC(u_analytic(-pi - 2pi/511),u_analytic(pi + 2pi/511))
    step(u,p,t) = A*bc*u
    heat_eqn = ODEProblem(step, u0, (0.,10.))
    soln = solve(heat_eqn,Tsit5(),dense=false,tstops=0:0.01:10)

    for t in 0:0.1:10
        @test soln(t)[1] ≈ u0[1] rtol=0.05
        @test soln(t)[end] ≈ u0[end] rtol=0.05
    end
end

@testset "Parabolic Heat Equation with Neumann BCs" begin
    N = 512
    dx = 2π/(N-1)
    x = collect(-pi : dx : pi)
    u0 = @. -(x - 0.5)^2 + 1/12
    B = CenteredDifference(1,2,dx,N)
    deriv_start, deriv_end = (B*u0)[1], (B*u0)[end]

    A = CenteredDifference(2,2,dx,N)
    bc = NeumannBC((deriv_start,deriv_end),dx,1)

    step(u,p,t) = A*bc*u
    heat_eqn = ODEProblem(step, u0, (0.,10.))
    soln = solve(heat_eqn,Tsit5(),dense=false,tstops=0:0.01:10)

    first_order_coeffs_start = [-11/6, 3.0, -3/2, 1/3] * (1/dx)
    first_order_coeffs_end = -reverse([-11/6, 3.0, -3/2, 1/3] * (1/dx))

    for t in 0:0.1:10
        @test sum(first_order_coeffs_start .* soln(t)[1:4]) ≈ deriv_start atol=1e-1
        @test sum(first_order_coeffs_end .* soln(t)[end-3:end]) ≈ deriv_end atol=1e-1
    end
end

@testset "Parabolic Heat Equation with Robin BCs" begin
    N = 512
    dx = 2π/(N-1)
    x = collect(-pi : dx : pi)
    u0 = @. -(x - 0.5)^2 + 1/12
    B = CenteredDifference(1,2,dx,N);
    deriv_start, deriv_end = (B*u0)[1], (B*u0)[end]
    params = [1.0,0.5]

    left_RBC = params[1]*u0[1] - params[2]*deriv_start
    right_RBC = params[1]*u0[end] + params[2]*deriv_end

    A = CenteredDifference(2,2,dx,N);
    bc = RobinBC((params[1],-params[2],left_RBC), (params[1],params[2],right_RBC),dx,1);
    step(u,p,t) = A*bc*u
    heat_eqn = ODEProblem(step, u0, (0.,10.));
    soln = solve(heat_eqn,Tsit5(),dense=false,tstops=0:0.01:10);

    first_order_coeffs_start = [-11/6, 3.0, -3/2, 1/3] * (1/dx)
    first_order_coeffs_end = -reverse([-11/6, 3.0, -3/2, 1/3] * (1/dx))
    val = []

    for t in 0.2:0.1:9.8
        @test params[1]*soln(t)[1] - params[2]*sum(first_order_coeffs_start .* soln(t)[1:4]) ≈ left_RBC atol=1e-1
        # append!(val,params[1]*soln(t)[1] + -params[2]*sum(first_order_coeffs_start .* soln(t)[1:4]) - left_RBC)
        @test params[1]*soln(t)[end] + params[2]*sum(first_order_coeffs_end .* soln(t)[end-3:end]) ≈ right_RBC atol=1e-1
    end
end
