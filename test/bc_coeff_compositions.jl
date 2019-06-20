using LinearAlgebra, DiffEqOperators, Random, Test, BandedMatrices, SparseArrays

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
