using LinearAlgebra, DiffEqOperators, Random, Test
################################################################################
# Test BoundaryPaddedMatrix
################################################################################

n = 100
m = 120
A = rand(n,m)


lower = A[1,:]
upper = A[end,:]

Apad = BoundaryPaddedMatrix{Float64,1, typeof(A), typeof(lower)}(lower, upper, A[2:(end-1), :])

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
S = size(A)
for dim in 1:3
    lower = selectdim(A, dim, 1)
    upper = selectdim(A, dim, size(A)[dim])

    Apad = BoundaryPadded3Tensor{Float64, dim, typeof(A), typeof(lower)}(lower, upper, selectdim(A, dim, 2:(size(A)[dim]-1)))

    @test A == Array(Apad) #test Concretization of BoundaryPaddedMatrix
    @test_broken A == Apad[:,:,:]
    @test_broken A == Apad[1:S[1], 1:S[2], 1:S[3]]
    for i in 1:n, j in 1:m, k in 1:o #test getindex for all indicies of Apad
        @test A[i,j,k] == Apad[i,j,k]
    end
end
################################################################################
# Test BoundaryPaddedArray to 5D just for fun
################################################################################


n = 20
m = 26
o = 8
p = 12
q = 14
A = rand(n,m,o,p,q)
for dim in 1:5
    lower = selectdim(A, dim, 1)
    upper = selectdim(A, dim, size(A)[dim])

    Apad = BoundaryPaddedArray{Float64, dim, 5, 4, typeof(A), typeof(lower)}(lower, upper, selectdim(A, dim, 2:(size(A)[dim]-1)))

    @test A == Array(Apad) #test Concretization of BoundaryPaddedMatrix

    for i in 1:n, j in 1:m, k in 1:o, l in 1:p, f in 1:q  #test getindex for all indicies of Apad
        @test A[i,j,k,l,f] == Apad[i,j,k,l,f]
    end
end
