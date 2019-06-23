using LinearAlgebra, DiffEqOperators, Random, Test, BandedMatrices, SparseArrays

function fourth_deriv_approx_stencil(N)
    A = zeros(N,N+2)
    A[1,1:8] = [3.5 -56/3 42.5 -54.0 251/6 -20.0 5.5 -2/3]
    A[2,1:8] = [2/3 -11/6 0.0 31/6 -22/3 4.5 -4/3 1/6]
    A[N-1,N-5:end] = reverse([2/3 -11/6 0.0 31/6 -22/3 4.5 -4/3 1/6], dims=2)
    A[N,N-5:end] = reverse([3.5 -56/3 42.5 -54.0 251/6 -20.0 5.5 -2/3], dims=2)
    for i in 3:N-2
        A[i,i-2:i+4] = [-1/6 2.0 -13/2 28/3 -13/2 2.0 -1/6]
    end
    return A
end

# Generate random parameters
al = rand()
bl = rand()
cl = rand()
dx_l = rand()
ar = rand()
br = rand()
cr = rand()
dx_r = rand()

Q = RobinBC(al, bl, cl, dx_l, ar, br, cr, dx_r)
N = 20
L = CenteredDifference(4,4, 1.0, N)
L2 = CenteredDifference(2,4, 1.0, N)

function coeff_func(du,u,p,t)
  du .= u
end

cL = coeff_func*L
coeffs = rand(N)
DiffEqOperators.update_coefficients!(cL,coeffs,nothing,0.0)

@test cL.coefficients == coeffs

# Test GhostDerivativeOperator constructor by *
u = rand(N)
A = L*Q
# Test for consistency of GhostDerivativeOperator*u with L*(Q*u)
@test A*u ≈ L*(Q*u)


# Test concretization
@test Array(A)[1] ≈ (Array(L)*Array(Q,N)[1], Array(L)*Array(Q,N)[2])[1]
@test Array(A)[2] ≈ (Array(L)*Array(Q,N)[1], Array(L)*Array(Q,N)[2])[2]
@test SparseMatrixCSC(A)[1] ≈ (SparseMatrixCSC(L)*SparseMatrixCSC(Q,N)[1], SparseMatrixCSC(L)*SparseMatrixCSC(Q,N)[2])[1]
@test SparseMatrixCSC(A)[2] ≈ (SparseMatrixCSC(L)*SparseMatrixCSC(Q,N)[1], SparseMatrixCSC(L)*SparseMatrixCSC(Q,N)[2])[2]
@test sparse(A)[1] ≈ (sparse(L)*sparse(Q,N)[1], sparse(L)*sparse(Q,N)[2])[1]
@test sparse(A)[2] ≈ (sparse(L)*sparse(Q,N)[1], sparse(L)*sparse(Q,N)[2])[2]
# BandedMatrix not implemeted for boundary operator
@test_broken BandedMatrix(A)[1] ≈ (BandedMatrix(L)*BandedMatrix(Q,N)[1], BandedMatrix(L)*BandedMatrix(Q,N)[2])[1]
@test_broken BandedMatrix(A)[2] ≈ (BandedMatrix(L)*BandedMatrix(Q,N)[1], BandedMatrix(L)*BandedMatrix(Q,N)[2])[2]

@test Array(A)[1]*u + Array(A)[2] ≈ L*(Q*u) ≈ A*u
@test sparse(A)[1]*u + sparse(A)[2] ≈ L*(Q*u) ≈ A*u

u = rand(22)
@test (L + L2) * u ≈ convert(AbstractMatrix,L + L2) * u ≈ (BandedMatrix(L) + BandedMatrix(L2)) * u

# Test \
dx = 0.0001
x = 0.0001:dx:0.01
N = length(x)
u = sin.(x)

L = CenteredDifference(4, 4, dx, N)
Q = RobinBC(1.0, 0.0, sin(0), dx, 1.0, 0.0, sin(0.01+dx), dx)
A = L*Q

correct_L = fourth_deriv_approx_stencil(N) / dx^4
correct_QL = [transpose(zeros(N)); Diagonal(ones(N)); transpose(zeros(N))]
correct_Qb = [zeros(N+1); sin(0.01+dx)]

correct_x = (correct_L*correct_QL) \ (u - correct_L*correct_Qb)
x = A \ u

@test x ≈ correct_x
