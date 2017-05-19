using PDEOperators
using Base.Test

L = 100
x = collect(1:1.0:L).^2
temp = LinearOperator{Float64}(2,2)
boundary_points = temp.boundary_point_count

res = operate(temp, x)
@time @test res[boundary_points + 1: L - boundary_points] == 2.0*ones(L - 2*boundary_points)

operate!(res, temp, x)
@time @test res[boundary_points + 1: L - boundary_points] == 2.0*ones(L - 2*boundary_points)

res = temp*x
@time @test res[boundary_points + 1: L - boundary_points] == 2.0*ones(L - 2*boundary_points)
