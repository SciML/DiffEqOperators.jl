using LinearAlgebra, DiffEqOperators, Random, Test
################################################################################
# Test 2d extension
################################################################################

#Create Array
n = 100
m = 120
A = rand(n,m)

#Create atomic BC
q1 = RobinBC([1.0, 2.0, 3.0], [0.0, -1.0, 2.0], [0.1, 0.1], 4.0)
q2 = PeriodicBC{Float64}()

BCx = vcat(fill(q1, div(m,2)), fill(q2, div(m,2)))  #The size of BCx has to be all size components *except* for x
BCy = vcat(fill(q1, div(n,2)), fill(q2, div(n,2)))

Q = MultiDimBC(BCx, BCy)

Aextended = Q*A

@test size(Aextended) == size(A).+2
for i in 2:(n-1), j in 2:(m-1)
    @test [Aextended[i, k] for k in 1:(m+2)] == Array(BCy[i-1]*A[i-1,:])
    @test [Aextended[k, j] for k in 1:(n+2)] == Array(BCx[j-1]*A[:, j-1])
end
