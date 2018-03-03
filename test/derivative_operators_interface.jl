# tests for full and sparse function
@testset "Full and Sparse functions:" begin
    N = 100
    d_order = 2
    approx_order = 2
    x = collect(1:1.0:N).^2

    A = DerivativeOperator{Float64}(d_order,approx_order,1.0,N,:Dirichlet0,:Dirichlet0)
    mat = full(A)
    sp_mat = sparse(A)
    @test mat == sp_mat;
    @test full(A, 10) == -Strang(10); # Strang Matrix is defined with the center term +ve
    @test full(A, N) == -Strang(N); # Strang Matrix is defined with the center term +ve
    @test full(A) == sp_mat
    @test norm(A, Inf) == norm(mat, Inf)

    # testing correctness
    N = 1000
    d_order = 4
    approx_order = 10
    y = collect(1:1.0:N).^4 - 2*collect(1:1.0:N).^3 + collect(1:1.0:N).^2;
    y = convert(Array{BigFloat, 1}, y)

    A = DerivativeOperator{BigFloat}(d_order,approx_order,one(BigFloat),N,:Dirichlet0,:Dirichlet0)
    boundary_points = A.boundary_point_count
    mat = full(A, N)
    sp_mat = sparse(A)
    @test mat == sp_mat;

    res = A*y
    @test res[boundary_points[1] + 1: N - boundary_points[2]] ≈ 24.0*ones(N - sum(boundary_points)) atol=10.0^-approx_order;
    @test A*y ≈ mat*y atol=10.0^-approx_order;
    @test A*y ≈ sp_mat*y atol=10.0^-approx_order;
    @test sp_mat*y ≈ mat*y atol=10.0^-approx_order;
end

@testset "Indexing tests" begin
    N = 1000
    d_order = 4
    approx_order = 10

    A = DerivativeOperator{Float64}(d_order,approx_order,1.0,N,:Dirichlet0,:Dirichlet0)
    @test A[1,1] ≈ 13.717407 atol=1e-4
    @test A[:,1] == (full(A))[:,1]
    @test A[10,20] == 0

    for i in 1:N
        @test A[i,i] == A.stencil_coefs[div(A.stencil_length, 2) + 1]
    end

    # Indexing Tests
    N = 1000
    d_order = 2
    approx_order = 2

    A = DerivativeOperator{Float64}(d_order,approx_order,1.0,N,:Dirichlet0,:Dirichlet0)
    M = full(A)

    @test A[1,1] == -2.0
    @test A[1:4,1] == M[1:4,1]
    @test A[5,2:10] == M[5,2:10]
    @test A[60:100,500:600] == M[60:100,500:600]
end

@testset begin "Operations on matrices"
    N = 51
    M = 101
    d_order = 2
    approx_order = 2

    xarr = linspace(0,1,N)
    yarr = linspace(0,1,M)
    dx = xarr[2]-xarr[1]
    dy = yarr[2]-yarr[1]
    F = [x^2+y for x = xarr, y = yarr]

    A = DerivativeOperator{Float64}(d_order,approx_order,dx,length(xarr),:None,:None)
    B = DerivativeOperator{Float64}(d_order,approx_order,dy,length(yarr),:None,:None)

    @test A*F ≈ 2*ones(N,M) atol=1e-2
    @test F*B ≈ 8*ones(N,M) atol=1e-2
    @test A*F*B ≈ zeros(N,M) atol=1e-2

    G = [x^2+y^2 for x = xarr, y = yarr]

    @test A*G ≈ 2*ones(N,M) atol=1e-2
    @test G*B ≈ 8*ones(N,M) atol=1e-2
    @test A*G*B ≈ zeros(N,M) atol=1e-2
end
