using Test, LinearAlgebra
using DiffEqOperators

@testset "utility functions" begin
    @test unit_indices(2) == (CartesianIndex(1,0), CartesianIndex(0,1))
    @test add_dims(ndims(zeros(2,2)) + 2, zeros(2,2)) == [4. 4.; 0. 0.; 0. 0.]
    @test perpindex(collect(1:5), 3) == [1, 2, 4, 5]
    @test perpsize(zeros(2,2,3,2), 3) == 2 * ones(3)
end
