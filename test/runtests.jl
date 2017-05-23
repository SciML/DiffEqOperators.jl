using PDEOperators
using Base.Test

L = 100
d_order = 2
approx_order = 2
x = collect(1:1.0:L).^2
A = LinearOperator{Float64}(2,2)
boundary_points = A.boundary_point_count

res = A*x
@time @test res[boundary_points + 1: L - boundary_points] == 2.0*ones(L - 2*boundary_points)

L = 1000
d_order = 4
approx_order = 10
y = collect(1:1.0:L).^4 - 2*collect(1:1.0:L).^3 + collect(1:1.0:L).^2;

A = LinearOperator{Float64}(d_order,approx_order)
boundary_points = A.boundary_point_count

res = A*y
@test_approx_eq_eps res[boundary_points + 1: L - boundary_points] 24.0*ones(L - 2*boundary_points) 10.0^-1 # Float64 is less stable

A = LinearOperator{BigFloat}(d_order,approx_order)
res = A*y
@test_approx_eq_eps res[boundary_points + 1: L - boundary_points] 24.0*ones(L - 2*boundary_points) 10.0^-approx_order

y = convert(Array{Rational, 1}, y)
res = A*y
@test_approx_eq_eps res[boundary_points + 1: L - boundary_points] 24.0*ones(L - 2*boundary_points) 10.0^-approx_order
