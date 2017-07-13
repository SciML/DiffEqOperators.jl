using Base.Test
using FactCheck

context("Parabolic Heat Equation with Dirichlet BCs:")do
    x = collect(-pi : 2pi/511 : pi);
    u0 = -(x - 0.5).^2 + 1/12;
    A = LinearOperator{Float64}(2,2,2π/511,512,:Dirichlet,:Dirichlet;bndry_fn=(u0[1],u0[end]));
    heat_eqn = ODEProblem(A, u0, (0.,10.));
    soln = solve(heat_eqn,Rosenbrock23(),dense=false,tstops=0:0.01:10);

    for t in 0:0.1:10
        @test soln(t)[1] ≈ u0[1]
        @test soln(t)[end] ≈ u0[end]
    end
end

context("Parabolic Heat Equation with Neumann BCs:")do
    x = collect(-pi : 2pi/511 : pi);
    u0 = -(x - 0.5).^2 + 1/12;
    B = LinearOperator{Float64}(1,2,2π/511,512,:None,:None);
    deriv_start, deriv_end = (B*u0)[1], (B*u0)[end]

    A = LinearOperator{Float64}(2,2,2π/511,512,:Neumann,:Neumann;bndry_fn=(deriv_start,deriv_end));

    heat_eqn = ODEProblem(A, u0, (0.,10.));
    soln = solve(heat_eqn,dense=false,tstops=0:0.01:10);

    first_order_coeffs_start = [-11/6, 3.0, -3/2, 1/3] * (511/2π)
    first_order_coeffs_end = -reverse([-11/6, 3.0, -3/2, 1/3] * (511/2π))

    for t in 0:0.1:10
        @test sum(first_order_coeffs_start .* soln(t)[1:4]) ≈ deriv_start atol=1e-1
        @test sum(first_order_coeffs_end .* soln(t)[end-3:end]) ≈ deriv_end atol=1e-1
    end
end