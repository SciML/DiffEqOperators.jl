using LinearAlgebra, SparseArrays, DiffEqOperators, Random, Test
################################################################################
# Test 2d extension
################################################################################

#Create Array
n = 8
m = 15
A = rand(n,m)

#Create atomic BC
q1 = RobinBC((1.0, 2.0, 3.0), (0.0, -1.0, 2.0), 0.1, 4.0)
q2 = PeriodicBC{Float64}()

BCx = vcat(fill(q1, div(m,2)), fill(q2, m-div(m,2)))  #The size of BCx has to be all size components *except* for x
BCy = vcat(fill(q1, div(n,2)), fill(q2, n-div(n,2)))


Qx = MultiDimBC(BCx, 1)
Qy = MultiDimBC(BCy, 2)

Ax = Qx*A
Ay = Qy*A

@test size(Ax)[1] == size(A)[1]+2
@test size(Ay)[2] == size(A)[2]+2

for j in 1:m
    @test Ax[:, j]  == Array(BCx[j]*A[:, j])
end
for i in 1:n
    @test Ay[i,:] == Array(BCy[i]*A[i,:])
end


################################################################################
# Test 3d extension
################################################################################

#Create Array
n = 8
m = 11
o = 12
A = rand(n,m, o)

#Create atomic BC
q1 = RobinBC((1.0, 2.0, 3.0), (0.0, -1.0, 2.0), 0.1, 4.0)
q2 = PeriodicBC{Float64}()

BCx = vcat(fill(q1, (div(m,2), o)), fill(q2, (m-div(m,2), o)))  #The size of BCx has to be all size components *except* for x
BCy = vcat(fill(q1, (div(n,2), o)), fill(q2, (n-div(n,2), o)))
BCz = fill(Dirichlet0BC(Float64), (n,m))

Qx = MultiDimBC(BCx, 1)
Qy = MultiDimBC(BCy, 2)
Qz = MultiDimBC(Dirichlet0BC(Float64), size(A), 3) #Test the other constructor

Ax = Qx*A
Ay = Qy*A
Az = Qz*A

Q = compose(Qx,Qy,Qz)
QL, Qb = Array(Q, size(A))
QLs, Qbs = sparse(Q, size(A))

A_conc = QL*reshape(A, prod(size(A))) .+Qb
A_conc_sp = QLs*reshape(A,prod(size(A))) .+Qbs

#test BC concretization
A_arr = Array(Q*A)
@test reshape(A_arr, prod(size(A_arr))) ≈ A_conc_sp ≈ A_conc

@test size(Ax)[1] == size(A)[1]+2
@test size(Ay)[2] == size(A)[2]+2
@test size(Az)[3] == size(A)[3]+2
for j in 1:m, k in 1:o
    @test Ax[:, j, k] == Array(BCx[j, k]*A[:, j, k])
end
for i in 1:n, k in 1:o
    @test Ay[i, :, k] == Array(BCy[i, k]*A[i, :, k])
end
for i in 1:n, j in 1:m
    @test Az[i, j, :] == Array(BCz[i, j]*A[i, j, :])
end

#test compositions to higher dimension
for N in 2:7
    sizes = rand(4:7, N)
    A = rand(sizes...)

    Q1_N = RobinBC(Tuple(rand(3)), Tuple(rand(3)), fill(0.1, N), 4.0, size(A))

    Q = compose(Q1_N...)

    A1_N = Q1_N.*fill(A, N)

    A_arr = Array(Q*A)

    @test A_arr == Array(compose(A1_N...))
end
