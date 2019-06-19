using LinearAlgebra, DiffEqOperators, Random, Test, BandedMatrices

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
@test A*u ≈ L*(Q*u)

# Test concretization
@test Array(A)[1] ≈ (Array(L)*Array(Q,N)[1], Array(L)*Array(Q,N)[2])[1]
@test Array(A)[2] ≈ (Array(L)*Array(Q,N)[1], Array(L)*Array(Q,N)[2])[2]
@test Array(A)[1]*u + Array(A)[2] ≈ L*(Q*u)

u = rand(22)
@test (L + L2) * u ≈ convert(AbstractMatrix,L + L2) * u ≈ (BandedMatrix(L) + BandedMatrix(L2)) * u
