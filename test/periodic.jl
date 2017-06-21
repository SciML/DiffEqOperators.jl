using Base.Test
using FactCheck
using SpecialMatrices

context("Periodic Boundary")do
    N = 100
    d_order = 2
    approx_order = 2
    x = collect(1:1.0:N).^2
    A = LinearOperator{Float64}(2,2,N,:periodic,:periodic)
    boundary_points = A.boundary_point_count

    res = A*x
    @test res[boundary_points + 1: N - boundary_points] == 2.0*ones(N - 2*boundary_points);

    N = 1000
    d_order = 4
    approx_order = 10
    y = collect(1:1.0:N).^4 - 2*collect(1:1.0:N).^3 + collect(1:1.0:N).^2;

    A = LinearOperator{Float64}(d_order,approx_order,N,:periodic,:periodic)
    boundary_points = A.boundary_point_count

    res = A*y
    @test res[boundary_points + 1: N - boundary_points] ≈ 24.0*ones(N - 2*boundary_points) atol=10.0^-1; # Float64 is less stable

    A = LinearOperator{BigFloat}(d_order,approx_order,N,:periodic,:periodic)
    y = convert(Array{BigFloat, 1}, y)
    res = A*y;
    @test res[boundary_points + 1: N - boundary_points] ≈ 24.0*ones(N - 2*boundary_points) atol=10.0^-approx_order;
end
