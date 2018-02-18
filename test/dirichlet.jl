using Base.Test

@testset "Dirichlet0 Boundary Conditions" begin
    N = 100
    h_inv = 1/(N-1)
    d_order = 2
    approx_order = 2
    x = collect(1:h_inv:N).^2
    A = DerivativeOperator{Float64}(2,2,1/99,length(x),:Dirichlet0,:Dirichlet0,BC=(0.0,0.0))
    boundary_points = A.boundary_point_count
    res = A*x
    @test res[boundary_points[1] + 1: N - boundary_points[2]] ≈ 2.0*ones(N - sum(boundary_points));

    N = 1000
    d_order = 4
    approx_order = 10
    y = collect(1:1.0:N).^4 - 2*collect(1:1.0:N).^3 + collect(1:1.0:N).^2;
    A = DerivativeOperator{Float64}(d_order,approx_order,1.0,N,:Dirichlet0,:Dirichlet0,BC=(0.0,0.0))
    boundary_points = A.boundary_point_count
    res = A*y
    @test res[boundary_points[1] + 1: N - boundary_points[2]] ≈ 24.0*ones(N - sum(boundary_points)) atol=10.0^-1; # Float64 is less stable

    A = DerivativeOperator{BigFloat}(d_order,approx_order,one(BigFloat),N,:Dirichlet0,:Dirichlet0,BC=(0.0,0.0))
    y = convert(Array{BigFloat, 1}, y)
    res = A*y
    @test res[boundary_points[1] + 1: N - boundary_points[2]] ≈ 24.0*ones(N - sum(boundary_points)) atol=10.0^-approx_order;
end


@testset "Dirichlet0 Boundary Conditions FD == DO?" begin

    #test if generic operator with constant spacing is equivalent to DerivativeOperator:
    N = 100
    h_inv = 1/(N-1)
    d_order = 2
    approx_order = 2
    x = collect(1:h_inv:N).^2
    A = DerivativeOperator{Float64}(2,2,1/99,length(x),:Dirichlet0,:Dirichlet0,BC=(0.0,0.0))
    Ag = FiniteDifference{Float64}(2,2,1/99*ones(length(x)-1),length(x),:Dirichlet0,:Dirichlet0,BC=(0.0,0.0))

    @test full(A) ≈ full(Ag)
    # boundary_points = A.boundary_point_count
    # res = A*x
    # resg = Ag*x
    #
    # @test res[boundary_points[1] + 1: N - boundary_points[2]] ≈ resg[boundary_points[1] + 1: N - boundary_points[2]];

    N = 1000
    d_order = 4
    approx_order = 10
    y = collect(1:1.0:N).^4 - 2*collect(1:1.0:N).^3 + collect(1:1.0:N).^2;
    A = DerivativeOperator{Float64}(d_order,approx_order,1.0,N,:Dirichlet0,:Dirichlet0,BC=(0.0,0.0))
    Ag = FiniteDifference{Float64}(d_order,approx_order,ones(N-1),N,:Dirichlet0,:Dirichlet0,BC=(0.0,0.0))

    @test full(A) ≈ full(Ag)
    # boundary_points = A.boundary_point_count
    # res = A*y
    # resg = Ag*x
    # @test res[boundary_points[1] + 1: N - boundary_points[2]] ≈ resg[boundary_points[1] + 1: N - boundary_points[2]] atol=10.0^-1; # Float64 is less stable

    A = DerivativeOperator{BigFloat}(d_order,approx_order,one(BigFloat),N,:Dirichlet0,:Dirichlet0,BC=(0.0,0.0))
    Ag = FiniteDifference{BigFloat}(d_order,approx_order,ones(BigFloat,N-1),N,:Dirichlet0,:Dirichlet0,BC=(0.0,0.0))

    @test full(A) ≈ full(Ag)
    # y = convert(Array{BigFloat, 1}, y)
    # res = A*y
    # resg = Ag*y
    #
    # @test res[boundary_points[1] + 1: N - boundary_points[2]] ≈ resg[boundary_points[1] + 1: N - boundary_points[2]] atol=10.0^-approx_order;

end

@testset "Dirichlet0 Boundary Conditions irregular grid" begin
    # tests as for DerivativeOperator, but with arbitrary grid (increasing step size):
    N = 1000
    d_order = 2
    approx_order = 2
    x = (1.:1.:N)*2.#erf.(linspace(-2,2,N))*N/2;
    dx = diff(x)
    y = x.^2

    A = FiniteDifference{Float64}(2,2,dx,length(x),:Dirichlet0,:Dirichlet0,BC=(0.0,0.0))
    boundary_points = A.boundary_point_count
    res = A*y
    @test res[boundary_points[1] + 1: N - boundary_points[2]] ≈ 2.0*ones(N - sum(boundary_points));

    # N = 1000
    d_order = 4
    approx_order = 10
    # dx = rand(N)
    # x = [0.0; cumsum(dx)]
    y = x.^4 - 2*x.^3 + x.^2;
    A = FiniteDifference{Float64}(d_order,approx_order,dx,N,:Dirichlet0,:Dirichlet0,BC=(0.0,0.0))
    boundary_points = A.boundary_point_count
    res = A*y
    @test res[boundary_points[1] + 1: N - boundary_points[2]] ≈ 24.0*ones(N - sum(boundary_points)) atol=10.0^-1; # Float64 is less stable

    A = FiniteDifference{BigFloat}(d_order,approx_order,BigFloat.(dx),N,:Dirichlet0,:Dirichlet0,BC=(0.0,0.0))
    y = convert(Array{BigFloat, 1}, y)
    res = A*y
    @test res[boundary_points[1] + 1: N - boundary_points[2]] ≈ 24.0*ones(N - sum(boundary_points)) atol=10.0^-approx_order;



end
