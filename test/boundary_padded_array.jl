using LinearAlgebra, DiffEqOperators, Random, Test
################################################################################
# Test BoundaryPaddedArray up to 5 dimensions
################################################################################

for dimensionality in 2:5
    for dim in 1:dimensionality
        sizes = rand(10:20, dimensionality)
        A = rand(sizes...)
        lower = selectdim(A, dim, 1)
        upper = selectdim(A, dim, size(A)[dim])

        Apad = DiffEqOperators.BoundaryPaddedArray{Float64, dim, dimensionality, dimensionality-1, typeof(A), typeof(lower)}(lower, upper, selectdim(A, dim, 2:(size(A)[dim]-1)))

        @test A == Array(Apad) #test Concretization of BoundaryPaddedMatrix

        for I in CartesianIndeces(A)  #test getindex for all indicies of Apad
            @test A[I] == Apad[I]
        end
    end
end
