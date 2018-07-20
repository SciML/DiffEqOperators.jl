using Test

@testset "Neumann0 Boundary" begin
    N = 100
    d_order = 2
    approx_order = 2
    x = collect(1:1.0:N).^2
    A = DerivativeOperator{Float64}(2,2,1.0,N,:Neumann0,:Neumann0)
    boundary_points = A.boundary_point_count
    res = A*x
    @test res[boundary_points[1] + 1: N - boundary_points[2]] ≈ 2.0*ones(N - sum(boundary_points)) atol=10.0^approx_order
    FD = DerivativeOperator{Float64}(1,2,1.0,N,:Neumann0,:Neumann0)
    first_deriv = FD*res
    @test first_deriv[1] ≈ 0.0 atol=10.0^-1 ## Derivative at edges in Neumann 0 is 0
    @test first_deriv[end] ≈ 0.0 atol=10.0^-1 ## Derivative at edges in Neumann 0 is 0

    N = 1000
    d_order = 4
    approx_order = 10
    y = collect(1:1.0:N).^4 - 2*collect(1:1.0:N).^3 + collect(1:1.0:N).^2;
    A = DerivativeOperator{Float64}(d_order,approx_order,1.0,N,:Neumann0,:Neumann0)
    boundary_points = A.boundary_point_count
    res = A*y
    @test res[boundary_points[1] + 1: N - boundary_points[2]] ≈ 24.0*ones(N - sum(boundary_points)) atol=10.0^-1; # Float64 is less stable
    FD = DerivativeOperator{Float64}(1,2,1.0,N,:Neumann0,:Neumann0)
    first_deriv = FD*res
    @test first_deriv[1] ≈ 0.0 atol=10.0^-1 ## Derivative at edges in Neumann 0 is 0
    @test first_deriv[end] ≈ 0.0 atol=10.0^-1 ## Derivative at edges in Neumann 0 is 0

    A = DerivativeOperator{BigFloat}(d_order,approx_order,one(BigFloat),N,:Neumann0,:Neumann0)
    y = convert(Array{BigFloat, 1}, y)
    res = A*y
    @test res[boundary_points[1] + 1: N - boundary_points[2]] ≈ 24.0*ones(N - sum(boundary_points)) atol=10.0^-approx_order;
    FD = DerivativeOperator{BigFloat}(1,2,one(BigFloat),N,:Neumann0,:Neumann0)
    first_deriv = FD*res
    @test first_deriv[1] ≈ 0.0 atol=10.0^-1 ## Derivative at edges in Neumann 0 is 0
    @test first_deriv[end] ≈ 0.0 atol=10.0^-1 ## Derivative at edges in Neumann 0 is 0
end

@testset "General Neumann Boundary Condition" begin
    N = 100
    h_inv = 1/(N-1)
    d_order = 2
    approx_order = 2
    x = 0:h_inv:π
    A = DerivativeOperator{Float64}(2,2,h_inv,N,:Neumann,:Neumann;BC=(1,-1))
    boundary_points = A.boundary_point_count
    res = A*sin.(x)
    @test res ≈ cos.(x) atol=10.0^approx_order
    FD = DerivativeOperator{Float64}(1,2,h_inv,N,:None,:None)
    first_deriv_res = FD*res
    @test first_deriv_res[1] ≈ -1.0 atol=10.0^-1 ## Derivative at edges in Neumann 0 is 0
    @test first_deriv_res[end] ≈ +1.0 atol=10.0^-1 ## Derivative at edges in Neumann 0 is 0


    N = 1000
    h_inv = 1/(N-1)
    d_order = 2
    approx_order = 2
    x = 0:h_inv:1
    A = DerivativeOperator{Float64}(d_order,approx_order,h_inv,N,:Neumann,:Neumann;BC=(1,-1))
    boundary_points = A.boundary_point_count
    res = A*(x.*(1.0.-x))
    @test res ≈ -2*ones(size(x)) atol=10.0^-approx_order

    # A = DerivativeOperator{BigFloat}(d_order,approx_order,N,:Neumann0,:Neumann0)
    # y = convert(Array{BigFloat, 1}, y)
    # res = A*y
    # @test res[boundary_points + 1: N - boundary_points] ≈ 24.0*ones(N - 2*boundary_points) atol=10.0^-approx_order;
    # FD = DerivativeOperator{BigFloat}(1,2,N,:Neumann0,:Neumann0)
    # first_deriv = FD*res
    # @test first_deriv[1] ≈ 0.0 atol=10.0^-1 ## Derivative at edges in Neumann 0 is 0
    # @test first_deriv[end] ≈ 0.0 atol=10.0^-1 ## Derivative at edges in Neumann 0 is 0

end
