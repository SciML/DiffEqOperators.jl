@testset "Generic Derivative Operator Interface" begin
    xgrid = 0.0:1.0:3.0
    L = UniformDerivativeStencil(xgrid,2,2)
    BC = (:Dirichlet0, :Neumann0)
    LB = GenericDerivativeOperator(L, BC)
    x = [1.0, 2.0]
    xbar = [0.0, 1.0, 2.0, 2.0]
    y = [xbar[1] + xbar[3] - 2xbar[2], xbar[2] + xbar[4] - 2xbar[3]]
    @test LB * x ≈ y

    Lmat = [1. -2. 1. 0.;
            0. 1. -2. 1.]
    Qmat = [0. 0.; 1. 0.; 0. 1.; 0. 1.]
    @test Matrix(LB) ≈ Lmat * Qmat
end
