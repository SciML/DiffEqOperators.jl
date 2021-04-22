using Test, LinearAlgebra
using DiffEqOperators

@testset "utility functions" begin
    @test DiffEqOperators.unit_indices(2) == (CartesianIndex(1,0), CartesianIndex(0,1))
    @test DiffEqOperators.add_dims(zeros(2,2), ndims(zeros(2,2)) + 2) == [6. 6.; 0. 0.; 0. 0.]
    @test DiffEqOperators.perpindex(collect(1:5), 3) == [1, 2, 4, 5]
    @test DiffEqOperators.perpsize(zeros(2,2,3,2), 3) == (2, 2, 2)
end

@testset "count differentials 1D" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)

    Dx = Differential(x)
    eq  = Dt(u(t,x)) ~ -Dx(u(t,x))
    @test DiffEqOperators.count_differentials(eq.rhs, x.val) == 1
    @test DiffEqOperators.count_differentials(eq.rhs, t.val) == 0
    @test DiffEqOperators.count_differentials(eq.lhs, t.val) == 1
    @test DiffEqOperators.count_differentials(eq.lhs, x.val) == 0

    Dxx = Differential(x)^2
    eq  = Dt(u(t,x)) ~ Dxx(u(t,x))
    @test DiffEqOperators.count_differentials(eq.rhs, x.val) == 2
    @test DiffEqOperators.count_differentials(eq.rhs, t.val) == 0
    @test DiffEqOperators.count_differentials(eq.lhs, t.val) == 1
    @test DiffEqOperators.count_differentials(eq.lhs, x.val) == 0

    Dxxxx = Differential(x)^4
    eq  = Dt(u(t,x)) ~ -Dxxxx(u(t,x))
    @test DiffEqOperators.count_differentials(eq.rhs, x.val) == 4
    @test DiffEqOperators.count_differentials(eq.rhs, t.val) == 0
    @test DiffEqOperators.count_differentials(eq.lhs, t.val) == 1
    @test DiffEqOperators.count_differentials(eq.lhs, x.val) == 0
end

@testset "count differentials 2D" begin
    @parameters t x y
    @variables u(..)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2
    Dt = Differential(t)

    eq  = Dt(u(t,x,y)) ~ Dxx(u(t,x,y)) + Dyy(u(t,x,y))
    @test DiffEqOperators.count_differentials(eq.rhs, x.val) == 2
    @test DiffEqOperators.count_differentials(eq.rhs, y.val) == 2
    @test DiffEqOperators.count_differentials(eq.rhs, t.val) == 0
    @test DiffEqOperators.count_differentials(eq.lhs, t.val) == 1
    @test DiffEqOperators.count_differentials(eq.lhs, x.val) == 0
    @test DiffEqOperators.count_differentials(eq.lhs, y.val) == 0
end
