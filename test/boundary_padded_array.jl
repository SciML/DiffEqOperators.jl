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

        for I in CartesianIndices(A)  #test getindex for all indicies of Apad
            @test A[I] == Apad[I]
        end
    end
end

################################################################################
# Test ComposedBoundaryPaddedMatrix
################################################################################

n = 10
m = 21
A = rand(n,m)
A[1,1] = A[end,1] = A[1,end] = A[end,end] = 0.0

lower = Vector[A[1,2:(end-1)], A[2:(end-1),1]]
upper = Vector[A[end,2:(end-1)], A[2:(end-1),end]]

Apad = ComposedBoundaryPaddedMatrix{Float64, typeof(A), typeof(lower[1])}(lower, upper, A[2:(end-1), 2:(end-1)])

@test A == Array(Apad) #test Concretization of BoundaryPaddedMatrix

for i in 1:n, j in 1:m #test getindex for all indicies of Apad
    @test A[i,j] == Apad[i,j]
end

################################################################################
# Test ComposedBoundaryPaddedTensor{3}
################################################################################

n = 10
m = 12
o = 7
A = rand(n,m,o)
A[1,1,:] = A[end,1, :] = A[1,end, :] = A[end,end, :] = 0.0
A[1,:,1] = A[end, :, 1] = A[1,:,end] = A[end,:,end] = 0.0
A[:,1,1] = A[:,end,1] = A[:,1,end] = A[:,end,end] = 0.0

lower = Matrix[A[1,2:(end-1), 2:(end-1)], A[2:(end-1),1, 2:(end-1)] A[2:(end-1), 2:(end-1), 1]]
upper = Matrix[A[end, 2:(end-1), 2:(end-1)], A[2:(end-1), end, 2:(end-1)] A[2:(end-1), 2:(end-1), end]]

Apad = BoundaryPadded3Tensor{Float64, typeof(A), typeof(lower[1])}(lower, upper, A[2:(end-1), 2:(end-1), 2:(end-1)])

@test A == Array(Apad) #test Concretization of BoundaryPaddedMatrix

for I in CartesianIndices(A) #test getindex for all indicies of Apad
    @test A[I] == Apad[I]
end
