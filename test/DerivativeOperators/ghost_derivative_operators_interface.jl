using DiffEqOperators, Test

@testset "Auxiliary functions for GhostDerivativeOperator" begin
    # check length and size functions
    L = CenteredDifference(2, 2, 1.0, 10)
    Q1 = RobinBC((-1.2, 0.3, 0.5), (0.2, 0.5, -30.5), 1.0)
    Q2 = PeriodicBC(Float64)

    A = L * Q1
    B = L * Q2

    @test ndims(A) == 2
    @test size(A) == (10, 10)
    @test size(A, 1) == 10
    @test length(A) == 100

    @test ndims(B) == 2
    @test size(B) == (10, 10)
    @test size(B, 1) == 10
    @test length(B) == 100
end
