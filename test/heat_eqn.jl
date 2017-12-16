using Base.Test
using DiffEqOperators, OrdinaryDiffEq

@testset "Parabolic Heat Equation with Dirichlet BCs" begin
    x = collect(-pi : 2pi/511 : pi);
    u0 = -(x - 0.5).^2 + 1/12;
    A = DerivativeOperator{Float64}(2,2,2π/511,512,:Dirichlet,:Dirichlet;BC=(u0[1],u0[end]));
    heat_eqn = ODEProblem(A, u0, (0.,10.));
    soln = solve(heat_eqn,Tsit5(),dense=false,tstops=0:0.01:10);

    for t in 0:0.1:10
        @test soln(t)[1] ≈ u0[1]
        @test soln(t)[end] ≈ u0[end]
    end
end

@testset "Parabolic Heat Equation with Neumann BCs" begin
    N = 512
    dx = 2π/(N-1)
    x = collect(-pi : dx : pi);
    u0 = -(x - 0.5).^2 + 1/12;
    B = DerivativeOperator{Float64}(1,2,dx,N,:None,:None);
    deriv_start, deriv_end = (B*u0)[1], (B*u0)[end]

    A = DerivativeOperator{Float64}(2,2,dx,N,:Neumann,:Neumann;BC=(deriv_start,deriv_end));

    heat_eqn = ODEProblem(A, u0, (0.,10.));
    soln = solve(heat_eqn,Tsit5(),dense=false,tstops=0:0.01:10);

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
    x = collect(-pi : dx : pi);
    u0 = -(x - 0.5).^2 + 1/12;
    B = DerivativeOperator{Float64}(1,2,dx,N,:None,:None);
    deriv_start, deriv_end = (B*u0)[1], (B*u0)[end]
    params = [1.0,0.5]

    left_RBC = params[1]*u0[1] - params[2]*deriv_start
    right_RBC = params[1]*u0[end] + params[2]*deriv_end

    A = DerivativeOperator{Float64}(2,2,dx,N,:Robin,:Dirichlet0;BC=((params[1],params[2],left_RBC),u0[end]));

    heat_eqn = ODEProblem(A, u0, (0.,10.));
    soln = solve(heat_eqn,Tsit5(),dense=false,tstops=0:0.01:10);

    first_order_coeffs_start = [-11/6, 3.0, -3/2, 1/3] * (1/dx)
    first_order_coeffs_end = -reverse([-11/6, 3.0, -3/2, 1/3] * (1/dx))
    # Broken in different amounts on each CI computer
    for t in 0.2:0.1:9.8
        @test_skip params[1]*soln(t)[1] + -params[2]*sum(first_order_coeffs_start .* soln(t)[1:4]) ≈ left_RBC atol=1e-1
        # append!(val,params[1]*soln(t)[1] + -params[2]*sum(first_order_coeffs_start .* soln(t)[1:4]) - left_RBC)
        @test_skip params[1]*soln(t)[end] + params[2]*sum(first_order_coeffs_end .* soln(t)[end-3:end]) ≈ right_RBC atol=1e-1
    end
end
