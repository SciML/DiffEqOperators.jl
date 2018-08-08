using Test

@testset "None BC DerivativeOperator" begin
    N = 10
    d_order = 1
    approx_order = 2
    A = DerivativeOperator{Float64}(d_order,approx_order,1.0,N,:None,:None)
    Amat = convert(Array,A)

    @test Amat[1,1:3] ≈ [-1.5, 2.0, -0.5]
    for row = 2:N-1
        @test Amat[row,row-1:row+1] ≈ [-0.5, 0.0, 0.5]
    end
    @test Amat[end,end-2:end] ≈ [0.5, -2.0, 1.5]

    d_order = 2
    A = DerivativeOperator{Float64}(d_order,approx_order,1.0,N,:None,:None)
    Amat = convert(Array,A)

    @test Amat[1,1:4] ≈ [2.0, -5.0, 4.0, -1.0]
    for row = 2:N-1
        @test Amat[row,row-1:row+1] ≈ [1.0, -2.0, 1.0]
    end
    @test Amat[end,end-3:end] ≈ [-1.0, 4.0, -5.0, 2.0]

    N = 10
    d_order = 1
    approx_order = 3
    A = DerivativeOperator{Float64}(d_order,approx_order,1.0,N,:None,:None)
    Amat = convert(Array,A)

    @test Amat[1,1:4] ≈ [-11/6, 3, -3/2, 1/3]
    for row = 3:N-2
        @test Amat[row,row-1:row+1] ≈ [-1/2, 0, 1/2]
    end
    @test Amat[end,end-3:end] ≈ [-1/3, 3/2, -3, 11/6]

end

@testset "None BC FiniteDifference Operator" begin
    N = 10
    d_order = 1
    approx_order = 2
    A = FiniteDifference{Float64}(d_order,approx_order,ones(N-1),N,:None,:None)
    Amat = convert(Array,A)

    @test Amat[1,1:3] ≈ [-1.5, 2.0, -0.5]
    for row = 2:N-1
        @test Amat[row,row-1:row+1] ≈ [-0.5, 0.0, 0.5]
    end
    @test Amat[end,end-2:end] ≈ [0.5, -2.0, 1.5]

    d_order = 2
    A = FiniteDifference{Float64}(d_order,approx_order,ones(N-1),N,:None,:None)
    Amat = convert(Array,A)

    @test Amat[1,1:4] ≈ [2.0, -5.0, 4.0, -1.0]
    for row = 2:N-1
        @test Amat[row,row-1:row+1] ≈ [1.0, -2.0, 1.0]
    end
    @test Amat[end,end-3:end] ≈ [-1.0, 4.0, -5.0, 2.0]

    N = 10
    d_order = 1
    approx_order = 3
    A = FiniteDifference{Float64}(d_order,approx_order,ones(N-1),N,:None,:None)
    Amat = convert(Array,A)

    @test Amat[1,1:4] ≈ [-11/6, 3, -3/2, 1/3]
    for row = 3:N-2
        @test Amat[row,row-1:row+1] ≈ [-1/2, 0, 1/2]
    end
    @test Amat[end,end-3:end] ≈ [-1/3, 3/2, -3, 11/6]

end


@testset "FiniteDifference Operator sin(x)==d/dx^4 sin(x)" begin
    N = 10
    n = 1000
    x = erf.(range(-2,stop=2,length=n))*2π
    x .+=abs(x[1])
    dx = diff(x)

    A = DiffEqOperators.FiniteDifference{Float64}(4,4,dx,length(x),:None,:None)
    boundary_points = A.boundary_point_count

    y = @. sin(x)+2 #avoid 0.0
    res = A*y
    @test res[boundary_points[1] .+ 1: N - boundary_points[2]] .+ 2.0 ≈ y[boundary_points[1] .+ 1: N - boundary_points[2]] atol=10.0^-1;

end
