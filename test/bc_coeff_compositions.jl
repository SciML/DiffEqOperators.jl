using LinearAlgebra, DiffEqOperators, Random, Test, BandedMatrices, SparseArrays

# 4th derivative, 4th order
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

function second_deriv_fourth_approx_stencil(N)
    A = zeros(N,N+2)
    A[1,1:6] = [5/6 -15/12 -1/3 7/6 -6/12 5/60]
    A[N,N-3:end] = [1/12 -6/12 14/12 -4/12 -15/12 10/12]
    for i in 2:N-1
        A[i,i-1:i+3] = [-1/12 4/3 -5/2 4/3 -1/12]
    end
    return A
end

function second_derivative_stencil(N)
  A = zeros(N,N+2)
  for i in 1:N, j in 1:N+2
      (j-i==0 || j-i==2) && (A[i,j]=1)
      j-i==1 && (A[i,j]=-2)
  end
  A
end

@testset "Test Constructor, Multiplication, and Concretization" begin
    # Generate random parameters
    al = rand()
    bl = rand()
    cl = rand()

    ar = rand()
    br = rand()
    cr = rand()
    dx = rand()

    Q = RobinBC((al, bl, cl), (ar, br, cr), dx)
    N = 20
    L = CenteredDifference(4, 4, dx, N)
    L2 = CenteredDifference(2,4, dx, N)
    L1 = UpwindDifference(1, 1, dx, N, 1.)

    function coeff_func(du,u,p,t)
        du .= u
    end

    cL = coeff_func*L
    coeffs = rand(N)
    DiffEqOperators.update_coefficients!(cL,coeffs,nothing,0.0)
    cA = cL*Q
    DiffEqOperators.update_coefficients!(cA,coeffs,nothing,0.0)
    @test cL.coefficients == coeffs == cA.L.coefficients

    cL1 = coeff_func*L1
    coeffs = rand(N)
    DiffEqOperators.update_coefficients!(cL1,coeffs,nothing,0.0)
    cA1 = cL1*Q
    DiffEqOperators.update_coefficients!(cA1,coeffs,nothing,0.0)
    @test cL1.coefficients == coeffs == cA1.L.coefficients

    A = L*Q
    c2A = coeff_func*A
    DiffEqOperators.update_coefficients!(c2A,coeffs,nothing,0.0)
    @test coeffs == c2A.L.coefficients


    # Test GhostDerivativeOperator constructor by *
    u = rand(N)
    A = L*Q
    A1 = L1*Q

    # Test for consistency of GhostDerivativeOperator*u with L*(Q*u)
    @test A == L * Q
    @test A*u ≈ L*(Q*u)
    @test A1 == L1 * Q
    @test A1*u ≈ L1*(Q*u)

    # Test for consistency of c*GhostDerivativeOperator*u with alternative methods
    c = 2.1
    @test c * A == (c * L) * Q == c * (L * Q)
    @test c * A * u ≈ (c * L) * (Q * u) ≈ c * (L * Q) * u
    @test c * A1 == (c * L1) * Q == c * (L1 * Q)
    @test c * A1 * u ≈ (c * L1) * (Q * u) ≈ c * (L1 * Q) * u

    # check A + B, where A and B are GhostDerivativeOperators
    B = c * A
    B1 = c * A1
    @test (A + B) * u == (A + c * A) * u == (B + A) * u == (c * A + A) * u
    @test (A + B) * u == A * u + B * u
    @test (A1 + B1) * u == (A1 + c * A1) * u == (B1 + A1) * u == (c * A1 + A1) * u
    @test (A1 + B1) * u == A1 * u + B1 * u

    # Check (L + L) * Q works
    LLQ = (L + L) * Q
    LLQ1 = (L1 + L1) * Q
    @test LLQ * u == A * u + A * u == (A + A) * u
    @test LLQ1 * u == A1 * u + A1 * u == (A1 + A1) * u

    # Test for consistency of c*GhostDerivativeOperator*u when c is a vector
    c = rand(N)
    L = CenteredDifference(4, 4, dx, N)
    L1 = UpwindDifference(1, 1, 1., N, 1.)
    A1 = L1 * Q
    cA = c * A
    cL = c * L
    cLQ = (c * L) * Q
    cA1 = c * A1
    cL1 = c * L1
    cLQ1 = (c * L1) * Q
    @test cA.L == cL
    @test cA.L.coefficients == cL.coefficients == cLQ.L.coefficients
    @test c * A == (c * L) * Q == c * (L * Q)
    @test c * A * u ≈ (c * L) * (Q * u) ≈ c * (L * Q) * u
    @test cA1.L.coefficients == cL1.coefficients == cLQ1.L.coefficients
    @test c * A1 == (c * L1) * Q == c * (L1 * Q)
    @test c * A1 * u ≈ (c * L1) * (Q * u) ≈ c * (L1 * Q) * u

    # check A + B, where A and B are GhostDerivativeOperators
    B = c * A
    B1 = c * A1
    # @test (A + B) == A + c * A == B + A == c * A + A # uncomment if we want to implement and test for equality of linear combinations of operators
    @test (A + B) * u == (A + c * A) * u == (B + A) * u == (c * A + A) * u
    @test (A + B) * u == A * u + B * u
    # @test (A1 + B1) == A + c * A1 == B1 + A1 == c * A1 + A1
    @test (A1 + B1) * u == (A1 + c * A1) * u == (B1 + A1) * u == (c * A1 + A1) * u
    @test (A1 + B1) * u == A1 * u + B1 * u

    # Test for consistency of GhostDerivativeOperator*M with L*(Q*M)

    M = rand(N,10)
    Qx = MultiDimBC{1}(Q, size(M))
    Am = L*Qx
    LQM = zeros(N,10)
    for i in 1:10
        mul!(view(LQM,:,i), L, Q*M[:,i])
    end
    ghost_LQM = Am*M
    @test ghost_LQM ≈ LQM

    u = rand(N + 2)
    @test (L + L2) * u ≈ convert(AbstractMatrix,L + L2) * u ≈ (BandedMatrix(L) + BandedMatrix(L2)) * u

    # Test concretization
    @test Array(A)[1] ≈ (Array(L)*Array(Q,N)[1], Array(L)*Array(Q,N)[2])[1]
    @test Array(A)[2] ≈ (Array(L)*Array(Q,N)[1], Array(L)*Array(Q,N)[2])[2]
    @test SparseMatrixCSC(A)[1] ≈ (SparseMatrixCSC(L)*SparseMatrixCSC(Q,N)[1], SparseMatrixCSC(L)*SparseMatrixCSC(Q,N)[2])[1]
    @test SparseMatrixCSC(A)[2] ≈ (SparseMatrixCSC(L)*SparseMatrixCSC(Q,N)[1], SparseMatrixCSC(L)*SparseMatrixCSC(Q,N)[2])[2]
    @test sparse(A)[1] ≈ (sparse(L)*sparse(Q,N)[1], sparse(L)*sparse(Q,N)[2])[1]
    @test sparse(A)[2] ≈ (sparse(L)*sparse(Q,N)[1], sparse(L)*sparse(Q,N)[2])[2]
    # BandedMatrix not implemented for boundary operator
    @test_broken BandedMatrix(A)[1] ≈ (BandedMatrix(L)*BandedMatrix(Q,N)[1], BandedMatrix(L)*BandedMatrix(Q,N)[2])[1]
    @test_broken BandedMatrix(A)[2] ≈ (BandedMatrix(L)*BandedMatrix(Q,N)[1], BandedMatrix(L)*BandedMatrix(Q,N)[2])[2]

    # Test that concretization works with multiplication
    u = rand(N)
    @test Array(A)[1]*u + Array(A)[2] ≈ L*(Q*u) ≈ A*u
    @test sparse(A)[1]*u + sparse(A)[2] ≈ L*(Q*u) ≈ A*u
end

@testset "Test Constructor, Multiplication, and Concretization (Non-uniform grid)" begin
    # Generate random parameters
    al = rand()
    bl = rand()
    cl = rand()

    ar = rand()
    br = rand()
    cr = rand()
    N = 20
    dx = rand(N + 2)

    Q = RobinBC((al, bl, cl), (ar, br, cr), dx)
    L = UpwindDifference(1, 1, dx, N + 1, 1.)
    L2 = CenteredDifference(2, 2, dx, N)
    L4 = CenteredDifference(4, 4, dx, N - 2)

    function coeff_func(du,u,p,t)
        du .= u
    end

    cL = coeff_func*L
    coeffs = rand(N + 1)
    DiffEqOperators.update_coefficients!(cL,coeffs,nothing,0.0)
    cA = cL*Q
    DiffEqOperators.update_coefficients!(cA,coeffs,nothing,0.0)
    @test cL.coefficients == coeffs == cA.L.coefficients

    cL2 = coeff_func*L2
    coeffs = rand(N)
    DiffEqOperators.update_coefficients!(cL2,coeffs,nothing,0.0)
    cA2 = cL2*Q
    DiffEqOperators.update_coefficients!(cA2,coeffs,nothing,0.0)
    @test cL2.coefficients == coeffs == cA2.L.coefficients

    cL4 = coeff_func*L4
    coeffs = rand(N - 2)
    DiffEqOperators.update_coefficients!(cL4,coeffs,nothing,0.0)
    cA4 = cL4*Q
    DiffEqOperators.update_coefficients!(cA4,coeffs,nothing,0.0)
    @test cL4.coefficients == coeffs == cA4.L.coefficients

    # Test GhostDerivativeOperator constructor by *
    u = rand(N + 1)
    u2 = rand(N)
    u4 = rand(N - 2)
    A = L*Q
    A2 = L2*Q
    A4 = L4*Q

    # Test for consistency of GhostDerivativeOperator*u with L*(Q*u)
    @test A == L * Q
    @test A*u ≈ L*(Q*u)
    @test A2 == L2 * Q
    @test A2*u2 ≈ L2*(Q*u2)
    @test A4 == L4 * Q
    @test A4*u4 ≈ L4*(Q*u4)

    # Test for consistency of c*GhostDerivativeOperator*u with alternative methods
    c = 2.1
    cA = c * A
    cL = c * L
    @test c * A == (c * L) * Q == c * (L * Q)
    @test c * A * u ≈ (c * L) * (Q * u) ≈ c * (L * Q) * u
    @test c * A2 == (c * L2) * Q == c * (L2 * Q)
    @test c * A2 * u2 ≈ (c * L2) * (Q * u2) ≈ c * (L2 * Q) * u2
    @test c * A4 == (c * L4) * Q == c * (L4 * Q)
    @test c * A4 * u4 ≈ (c * L4) * (Q * u4) ≈ c * (L4 * Q) * u4

    # check A + B, where A and B are GhostDerivativeOperators
    B = c * A
    B2 = c * A2
    B4 = c * A4
    @test (A + B) * u == (A + c * A) * u == (B + A) * u == (c * A + A) * u
    @test (A + B) * u == A * u + B * u
    @test (A2 + B2) * u2 == (A2 + c * A2) * u2 == (B2 + A2) * u2 == (c * A2 + A2) * u2
    @test (A2 + B2) * u2 == A2 * u2 + B2 * u2
    @test (A4 + B4) * u4 == (A4 + c * A4) * u4 == (B4 + A4) * u4 == (c * A4 + A4) * u4
    @test (A4 + B4) * u4 == A4 * u4 + B4 * u4

    # Check (L + L) * Q works
    LLQ = (L + L) * Q
    LLQ2 = (L2 + L2) * Q
    LLQ4 = (L4 + L4) * Q
    @test LLQ * u == A * u + A * u == (A + A) * u
    @test LLQ2 * u2 == A2 * u2 + A2 * u2 == (A2 + A2) * u2 # this comparison fails, even though the operators seem alike
    @test LLQ4 * u4 == A4 * u4 + A4 * u4 == (A4 + A4) * u4

    # Test for consistency of c*GhostDerivativeOperator*u when c is a vector
    c = rand(N + 1)
    L1 = L
    A1 = L1 * Q
    cA = c * A
    cL = c * L
    cLQ = (c * L) * Q
    cA1 = c * A1
    cL1 = c * L1
    cLQ1 = (c * L1) * Q
    @test cA.L == cL
    @test cA.L.coefficients == cL.coefficients == cLQ.L.coefficients
    @test c * A == (c * L) * Q == c * (L * Q)
    @test c * A * u ≈ (c * L) * (Q * u) ≈ c * (L * Q) * u
    @test cA1.L.coefficients == cL1.coefficients == cLQ1.L.coefficients
    @test c * A1 == (c * L1) * Q == c * (L1 * Q)
    @test c * A1 * u ≈ (c * L1) * (Q * u) ≈ c * (L1 * Q) * u

    # check A + B, where A and B are GhostDerivativeOperators
    B = c * A
    # @test (A + B) == A + c * A == B + A == c * A + A # uncomment if implement equality of combinations of linear operators
    @test (A + B) * u == (A + c * A) * u == (B + A) * u == (c * A + A) * u
    @test (A + B) * u == A * u + B * u

    # Test for consistency of GhostDerivativeOperator*M with L*(Q*M)
    M = rand(N+1,10)
    LQM = zeros(N+1,10)
    for i in 1:10
        mul!(view(LQM,:,i), L, Q*M[:,i])
    end
    ghost_LQM = A*M
    @test ghost_LQM ≈ LQM

    # Test concretization for UpwindDifference
    @test Array(A)[1] ≈ (Array(L)*Array(Q,N + 1)[1], Array(L)*Array(Q,N + 1)[2])[1]
    @test Array(A)[2] ≈ (Array(L)*Array(Q,N + 1)[1], Array(L)*Array(Q,N + 1)[2])[2]
    @test SparseMatrixCSC(A)[1] ≈ (SparseMatrixCSC(L)*SparseMatrixCSC(Q,N + 1)[1], SparseMatrixCSC(L)*SparseMatrixCSC(Q,N + 1)[2])[1]
    @test SparseMatrixCSC(A)[2] ≈ (SparseMatrixCSC(L)*SparseMatrixCSC(Q,N + 1)[1], SparseMatrixCSC(L)*SparseMatrixCSC(Q,N + 1)[2])[2]
    @test sparse(A)[1] ≈ (sparse(L)*sparse(Q,N + 1)[1], sparse(L)*sparse(Q,N + 1)[2])[1]
    @test sparse(A)[2] ≈ (sparse(L)*sparse(Q,N + 1)[1], sparse(L)*sparse(Q,N + 1)[2])[2]
    # BandedMatrix not implemeted for boundary operator
    @test_broken BandedMatrix(A)[1] ≈ (BandedMatrix(L)*BandedMatrix(Q,N + 1)[1], BandedMatrix(L)*BandedMatrix(Q,N + 1)[2])[1]
    @test_broken BandedMatrix(A)[2] ≈ (BandedMatrix(L)*BandedMatrix(Q,N + 1)[1], BandedMatrix(L)*BandedMatrix(Q,N + 1)[2])[2]

    # Test concretization for CenteredDifference
    A2 = L2 * Q
    @test Array(A2)[1] ≈ (Array(L2)*Array(Q,N)[1], Array(L2)*Array(Q,N)[2])[1]
    @test Array(A2)[2] ≈ (Array(L2)*Array(Q,N)[1], Array(L2)*Array(Q,N)[2])[2]
    @test SparseMatrixCSC(A2)[1] ≈ (SparseMatrixCSC(L2)*SparseMatrixCSC(Q,N)[1], SparseMatrixCSC(L2)*SparseMatrixCSC(Q,N)[2])[1]
    @test SparseMatrixCSC(A2)[2] ≈ (SparseMatrixCSC(L2)*SparseMatrixCSC(Q,N)[1], SparseMatrixCSC(L2)*SparseMatrixCSC(Q,N)[2])[2]
    @test sparse(A2)[1] ≈ (sparse(L2)*sparse(Q,N)[1], sparse(L2)*sparse(Q,N)[2])[1]
    @test sparse(A2)[2] ≈ (sparse(L2)*sparse(Q,N)[1], sparse(L2)*sparse(Q,N)[2])[2]
    # BandedMatrix not implemeted for boundary operator
    @test_broken BandedMatrix(A2)[1] ≈ (BandedMatrix(L2)*BandedMatrix(Q,N)[1], BandedMatrix(L2)*BandedMatrix(Q,N)[2])[1]
    @test_broken BandedMatrix(A2)[2] ≈ (BandedMatrix(L2)*BandedMatrix(Q,N)[1], BandedMatrix(L2)*BandedMatrix(Q,N)[2])[2]

    # Test that concretization works with multiplication, UpwindDifference
    u = rand(N + 1)
    @test Array(A)[1]*u + Array(A)[2] ≈ L*(Q*u) ≈ A*u
    @test sparse(A)[1]*u + sparse(A)[2] ≈ L*(Q*u) ≈ A*u

    # Test that concretization works with multiplication, CenteredDifference
    u = rand(N)
    @test Array(A2)[1]*u + Array(A2)[2] ≈ L2*(Q*u) ≈ A2*u
    @test sparse(A2)[1]*u + sparse(A2)[2] ≈ L2*(Q*u) ≈ A2*u
end

@testset "Test Left Division L2 (second order)" begin
    # Test \ Homogenous BC
    # f(x) = -x^2 + x
    # f''(x) = -2.0
    # f'(0) = f'(1) = 0
    f(x) = -x^2 + x
    f2(x) = -2.0

    dx = 0.01
    x = dx:dx:1.0-dx
    N = length(x)

    L = CenteredDifference(2, 2, dx, N)
    Q = RobinBC((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), dx)
    A = L*Q

    analytic_L = second_derivative_stencil(N) ./ dx^2
    analytic_QL = [transpose(zeros(N)); Diagonal(ones(N)); transpose(zeros(N))]
    analytic_AL = analytic_L*analytic_QL

    analytic_QM = zeros((length(x)+2)*2, length(x)*2)
    interior = CartesianIndices(Tuple([2:length(x)+1, 1:2]))
    I1 = CartesianIndex(1,0)
    for I in interior
        i = DiffEqOperators.cartesian_to_linear(I, (length(x)+2, 2)) #helper function, see utils.jl
        j = DiffEqOperators.cartesian_to_linear(I-I1, (length(x), 2))
        analytic_QM[i,j] = 1.0
    end
    analytic_Am = kron(Diagonal(ones(2)), analytic_L)*analytic_QM

    # No affine component to the this system
    analytic_f = analytic_AL \ f2.(x)
    ghost_f = A \ f2.(x)

    # Check that A\f2.(x) is consistent with analytic_AL \ f2.(x)
    @test analytic_f ≈ ghost_f

    # Additionally test that A\f2.(x) ≈ f.(x)
    @test f.(x) ≈ ghost_f ≈ analytic_f

    # Check that left division with matrices works
    M = [f2.(x) f2.(x)]
    Qx = MultiDimBC{1}(Q, size(M))

    Am = L*Qx
    ghost_fM = Am \ M
    s = size(M)
    analytic_fM = analytic_Am \ reshape(M, prod(s))
    @test ghost_fM ≈ reshape(analytic_fM, s)

    fM_temp = zeros(N,2)

    # Test \ Inhomogenous BC
    # f(x) = -x^2 + x + 4.0
    # f''(x) = -2.0
    # f'(0) = f'(1) = 0
    f(x) = -x^2 + x + 4.0
    f2(x) = -2.0

    dx = 0.01
    x = dx:dx:1.0-dx
    N = length(x)

    L = CenteredDifference(2, 2, dx, N)
    Q = RobinBC((1.0, 0.0, 4.0), (1.0, 0.0, 4.0), dx)
    A = L*Q

    analytic_L = second_derivative_stencil(N) ./ dx^2
    analytic_QL = [transpose(zeros(N)); Diagonal(ones(N)); transpose(zeros(N))]
    analytic_Qb = [4.0; zeros(N); 4.0]
    analytic_AL = analytic_L*analytic_QL
    analytic_Ab = analytic_L*analytic_Qb
    analytic_Lm = kron(Diagonal(ones(3)), analytic_L)

    analytic_f = analytic_AL \ (f2.(x) - analytic_Ab)
    ghost_f = A \ f2.(x)

    # Check that A\f2.(x) is consistent with analytic_AL \ f2.(x)
    @test analytic_f ≈ ghost_f

    # Additionally test that A\f2.(x) ≈ f.(x)
    @test f.(x) ≈ ghost_f ≈ analytic_f

    # Check \ for Matrix
    M2 = [f2.(x) 2.0*f2.(x) 10.0*f2.(x)]

    s = size(M2)
    Qx = MultiDimBC{1}(Q, size(M2))

    analytic_QLm, analytic_Qbm = Array(Qx, s)

    analytic_ALm = analytic_Lm*analytic_QLm
    analytic_Abm = analytic_Lm*analytic_Qbm

    Am = L*Qx
    analytic_M = analytic_ALm \ (reshape(M2 , prod(s)).- analytic_Abm)
    ghost_M = Am \ M2
    @test reshape(analytic_M, s) ≈ ghost_M

    # Additionally test that A\M2 ≈ [f, 2.0(f-4.0)+4.0, 10.0(f-4.0)+4.0]
    M = [f.(x) 2.0*(f.(x) .- 4.0).+4.0 10.0*(f.(x) .- 4.0).+4.0]
    @test M ≈ reshape(analytic_M, s) ≈ ghost_M
end

@testset "Test Left Division L4 (fourth order)" begin

    # Test \ homogenous and inhomogenous BC
    dx = 0.01
    x = 0.01:dx:0.2
    N = length(x)
    u = sin.(x)

    L = CenteredDifference(4, 4, dx, N)
    Q = RobinBC((1.0, 0.0, 0.0), (1.0, 0.0, sin(0.2+dx)), dx)
    A = L*Q

    analytic_L = fourth_deriv_approx_stencil(N) ./ dx^4
    analytic_QL = [transpose(zeros(N)); Diagonal(ones(N)); transpose(zeros(N))]
    analytic_AL = analytic_L*analytic_QL
    analytic_Qb = [zeros(N+1); sin(0.2+dx)]
    analytic_Ab = analytic_L*analytic_Qb


    analytic_QM = zeros((length(x)+2)*3, length(x)*3)
    interior = CartesianIndices(Tuple([2:length(x)+1, 1:3]))
    I1 = CartesianIndex(1,0)
    for I in interior
        i = DiffEqOperators.cartesian_to_linear(I, (length(x)+2, 3)) #helper function, see utils.jl
        j = DiffEqOperators.cartesian_to_linear(I-I1, (length(x), 3))
        analytic_QM[i,j] = 1.0
    end
    analytic_Am = kron(Diagonal(ones(3)), analytic_L)*analytic_QM

    analytic_u = analytic_AL \ Vector(u .- analytic_Ab)
    ghost_u = A \ u

    As_l, As_b = sparse(A, length(u))

    @test As_l ≈ analytic_AL #This is true
    @test As_b ≈  analytic_Ab #This is true
    @test Vector(u .- analytic_Ab) ≈ Vector(u .- As_b) #this is true
    u_translated = u .- analytic_Ab |> Vector
    # TODO validate analytic_AL as having reasonable condition (what is reasonable condition?)
    @test norm(analytic_AL \ u_translated - As_l \ u_translated) / norm(analytic_AL \ u_translated) < cond(analytic_AL)*eps()

    sp_u = As_l\Vector(u .- As_b)
    # Check that A\u.(x) is consistent with analytic_AL \ u.(x)
    @test ghost_u ≈ sp_u
    # TODO make cond function in ghost_derivative_operator?
    # however, note: ArgumentError: 2-norm condition number is not implemented for sparse matrices, try cond(Array(A), 2) instead
    # so the actual function would probably just propagate that error
    _cond(A::GhostDerivativeOperator, u) = cond(sparse(A,length(u))[1] |> Array)
    @test norm(analytic_u - ghost_u) / norm(ghost_u) < _cond(A,u) * eps() #

    M2 = [u 2.0*u 10.0*u]
    s = size(M2)
    Qx = MultiDimBC{1}(Q, size(M2))
    Am = L*Qx
    #Somehow the operator is singular
    @test_broken analytic_M = analytic_Am \ (reshape(M2, prod(s)) .- repeat(analytic_Ab, 3))
    @test_broken ghost_M = Am \ M2
    @test_broken reshape(analytic_M, s) ≈ ghost_M

end

@testset "Test Operator and BC combinations" begin
    N = 40
    x = range(-pi, stop = pi, length=N)
    Δx = x[2]-x[1]
    u₀=1.0
    Γ=1.0
    Dx=u₀*CenteredDifference{1}(1,4,Δx,N)
    Dxx=Γ*CenteredDifference{1}(2,4,Δx,N)
    Q=NeumannBC((2x[1]+2, 2x[end]+2), Δx, 6)

    A = Dx*Q + Dxx*Q

    y = A*(x.^2)
    analytic_y = 2x.+2

    @test_broken y ≈ analytic_y
end
