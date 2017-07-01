using FactCheck
using Base.Test

context("One-sided derivatives at boundaries: ") do
    N = 10
    d_order = 1
    approx_order = 2
    A = LinearOperator{Float64}(d_order,approx_order,1.0,N,:None,:None)
    Amat = full(A)

    @test isapprox(Amat[1,1:3], [-1.5, 2.0, -0.5])
    for row = 2:N-1
        @test isapprox(Amat[row,row-1:row+1], [-0.5, 0.0, 0.5])
    end
    @test isapprox(Amat[end,end-2:end], [0.5, -2.0, 1.5])

    d_order = 2
    A = LinearOperator{Float64}(d_order,approx_order,1.0,N,:None,:None)
    Amat = full(A)

    @test isapprox(Amat[1,1:4], [2.0, -5.0, 4.0, -1.0])
    for row = 2:N-1
        @test isapprox(Amat[row,row-1:row+1], [1.0, -2.0, 1.0])
    end
    @test isapprox(Amat[end,end-3:end], [-1.0, 4.0, -5.0, 2.0])

    N = 10
    d_order = 1
    approx_order = 3
    A = LinearOperator{Float64}(d_order,approx_order,1.0,N,:None,:None)
    Amat = full(A)

    @test isapprox(Amat[1,1:4], [-11/6, 3, -3/2, 1/3])
    for row = 3:N-2
        @test isapprox(Amat[row,row-1:row+1], [-1/2, 0, 1/2])
    end
    @test isapprox(Amat[end,end-3:end], [-1/3, 3/2, -3, 11/6])
end
