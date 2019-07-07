using LinearAlgebra, DiffEqOperators, Random, Test
################################################################################
# Test BoundaryPaddedMatrix
################################################################################

n = 100
m = 120
A = rand(n,m)
A[1,1] = A[end,1] = A[1,end] = A[end,end] = 0.0

lower = Vector[A[1,2:(end-1)], A[2:(end-1),1]]
upper = Vector[A[end,2:(end-1)], A[2:(end-1),end]]

Apad = BoundaryPaddedMatrix{Float64, typeof(A), typeof(lower[1])}(lower, upper, A[2:(end-1), 2:(end-1)])

@test A == Array(Apad) #test Concretization of BoundaryPaddedMatrix

for i in 1:n, j in 1:m #test getindex for all indicies of Apad
    @test A[i,j] == Apad[i,j]
end

################################################################################
# Test BoundaryPaddedTensor{3}
################################################################################

n = 100
m = 120
o = 78
A = rand(n,m,o)
A[1,1,:] = A[end,1, :] = A[1,end, :] = A[end,end, :] = 0.0
A[1,:,1] = A[end, :, 1] = A[1,:,end] = A[end,:,end] = 0.0
A[:,1,1] = A[:,end,1] = A[:,1,end] = A[:,end,end] = 0.0

lower = Matrix[A[1,2:(end-1), 2:(end-1)], A[2:(end-1),1, 2:(end-1)] A[2:(end-1), 2:(end-1), 1]]
upper = Matrix[A[end, 2:(end-1), 2:(end-1)], A[2:(end-1), end, 2:(end-1)] A[2:(end-1), 2:(end-1), end]]

Apad = BoundaryPadded3Tensor{Float64, typeof(A), typeof(lower[1])}(lower, upper, A[2:(end-1), 2:(end-1), 2:(end-1)])

@test A == Array(Apad) #test Concretization of BoundaryPaddedMatrix

for i in 1:n, j in 1:m, k in 1:o #test getindex for all indicies of Apad
    @test A[i,j,k] == Apad[i,j,k]
end
