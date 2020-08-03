using DiffEqOperators, Test

@testset "Auxiliary functions for GhostDerivativeOperator" begin
    # check length and size functions
    L = CenteredDifference(2, 2, 1., 10)
    Q = RobinBC((-1.2, .3, .5), (.2, .5, -30.5), 1.)
    A = L * Q

    @test ndims(A) == 2
    @test size(A) == (10, 10)
    @test size(A, 1) == 10
    @test length(A) == 100
end
