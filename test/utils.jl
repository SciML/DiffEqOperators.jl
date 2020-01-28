using Test, LinearAlgebra
using DiffEqOperators

@testset "utility functions" begin
    @test DiffEqOperators.unit_indices(2) == (CartesianIndex(1,0), CartesianIndex(0,1))
    @test DiffEqOperators.add_dims(zeros(2,2), ndims(zeros(2,2)) + 2) == [6. 6.; 0. 0.; 0. 0.]
    @test DiffEqOperators.perpindex(collect(1:5), 3) == [1, 2, 4, 5]
    @test DiffEqOperators.perpsize(zeros(2,2,3,2), 3) == (2, 2, 2)
end
