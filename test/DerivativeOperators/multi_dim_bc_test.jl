using LinearAlgebra, SparseArrays, DiffEqOperators, Random, Test
################################################################################
# Test 2d extension
################################################################################

s = x, y = (-1.9:0.1:1.9, -1.9:0.1:1.9)                 # domain of unpadded function
dx = dy = x[2] - x[1]
paraboloid(x::T, y::T) where T = 2*(x^2+y^2) - 4        # declare an elliptic paraboloid function
u0 = [paraboloid(X, Y) for X in x, Y in y]

q1 = MultiDimBC{1}(2,6,dx)                              # Padding along x
q2 = MultiDimBC{2}(2,6,dy)                              # Padding along y
Q = compose(q1,q2)                                      # composition ofMultiDimBC operators
u1 = Q*u0

s = x, y = (-2.0:0.1:2.0, -2.0:0.1:2.0)
u_pad = [paraboloid(X, Y) for X in x, Y in y]           # padded along x & y of 

for I in CartesianIndices(u_pad)
	@test u1[I] ≈ u_pad[I] atol = 0.05
end

dx = dy = 0.1*ones(40)                                  # non-uniform grid
q1 = MultiDimBC{1}(2,6,dx)
q2 = MultiDimBC{2}(2,6,dy)
Q = compose(q1,q2)
u1 = Q*u0

for I in CartesianIndices(u_pad)
	@test u1[I] ≈ u_pad[I] atol = 0.05
end


################################################################################
# Test 3d extension
################################################################################

#Create hyperboloid Array
s = x, y, z = (-1.9:0.1:1.9, -1.9:0.1:1.9, -1.9:0.1:1.9)
dx = dy = dz = x[2] - x[1]
hyperboloid(x::T, y::T, z::T) where T = 4*x^2+ 9*y^2 - z^2
u0 = [hyperboloid(X, Y, Z) for X in x, Y in y, Z in z]

# Create MultiDimBCs & compose
q1 = MultiDimBC{1}(3,6,dx)
q2 = MultiDimBC{2}(3,6,dy)
q3 = MultiDimBC{3}(3,6,dz)
Q = compose(q1,q2,q3)
u1 = Q*u0

# Padded analytical solution
s = x, y, z = (-2.0:0.1:2.0, -2.0:0.1:2.0, -2.0:0.1:2.0)
u_pad = [hyperboloid(X, Y, Z) for X in x, Y in y, Z in z]

# Testing
for I in CartesianIndices(u_pad)
	@test u1[I] ≈ u_pad[I] atol = 0.2
end

# Test for non-uniform grids
dx = dy = dz = 0.1*ones(40)
q1 = MultiDimBC{1}(3,6,dx)
q2 = MultiDimBC{2}(3,6,dy)
q3 = MultiDimBC{3}(3,6,dz)
Q = compose(q1,q2,q3)
u1 = Q*u0

for I in CartesianIndices(u_pad)
	@test u1[I] ≈ u_pad[I] atol = 0.2
end

# test BC concretization
# A_arr = Array(Q*A)
# @test reshape(A_arr, prod(size(A_arr))) ≈ A_conc_sp ≈ A_conc

# @test size(Ax)[1] == size(A)[1]+2
# @test size(Ay)[2] == size(A)[2]+2
# @test size(Az)[3] == size(A)[3]+2
# for j in 1:m, k in 1:o
#     @test Ax[:, j, k] == Array(BCx[j, k]*A[:, j, k])
# end
# for i in 1:n, k in 1:o
#     @test Ay[i, :, k] == Array(BCy[i, k]*A[i, :, k])
# end
# for i in 1:n, j in 1:m
#     @test Az[i, j, :] == Array(BCz[i, j]*A[i, j, :])
# end

# #test compositions to higher dimension
# for N in 2:6
#     sizes = rand(4:7, N)
#     local A = rand(sizes...)

#     Q1_N = RobinBC(Tuple(rand(3)), Tuple(rand(3)), fill(0.1, N), 4.0, size(A))

#     local Q = compose(Q1_N...)

#     A1_N = Q1_N.*fill(A, N)

#     local A_arr = Array(Q*A)
#     Q_l, Q_b = sparse(Q, size(A))

#     @test A_arr ≈ Array(compose(A1_N...))
#     @test A_arr ≈ reshape(Q_l*reshape(A, length(A)) .+ Q_b, size(A_arr)) #Test concretization
# end
