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
    L = CenteredDifference(4,4, 1.0, N)
    L2 = CenteredDifference(2,4, 1.0, N)

    function coeff_func(du,u,p,t)
        du .= u
    end

    cL = coeff_func*L
    coeffs = rand(N)
    DiffEqOperators.update_coefficients!(cL,coeffs,nothing,0.0)
    cA = cL*Q
    DiffEqOperators.update_coefficients!(cA,coeffs,nothing,0.0)
    @test cL.coefficients == coeffs == cA.L.coefficients

    A = L*Q
    c2A = coeff_func*A
    DiffEqOperators.update_coefficients!(c2A,coeffs,nothing,0.0)
    @test coeffs == c2A.L.coefficients


    # Test GhostDerivativeOperator constructor by *
    u = rand(N)
    A = L*Q
    # Test for consistency of GhostDerivativeOperator*u with L*(Q*u)
    @test A*u ≈ L*(Q*u)

    # Test for consistency of GhostDerivativeOperator*M with L*(Q*M)

    M = rand(N,10)
    Qx = MultiDimBC(Q, size(M),1)
    Am = L*Qx
    LQM = zeros(N,10)
    for i in 1:10
        mul!(view(LQM,:,i), L, Q*M[:,i])
    end
    ghost_LQM = Am*M
    @test ghost_LQM ≈ LQM

    u = rand(22)
    @test (L + L2) * u ≈ convert(AbstractMatrix,L + L2) * u ≈ (BandedMatrix(L) + BandedMatrix(L2)) * u

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

    # Test that concretization works with multiplication
    u = rand(20)
    @test Array(A)[1]*u + Array(A)[2] ≈ L*(Q*u) ≈ A*u
    @test sparse(A)[1]*u + sparse(A)[2] ≈ L*(Q*u) ≈ A*u
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
        i = c2l(I, (length(x)+2, 2)) #helper function, see utils.jl
        j = c2l(I-I1, (length(x), 2))
        analytic_QM[i,j] = 1.0
    end
    analytic_Am = kron(Diagonal(ones(2)), analytic_L)*analytic_QM

    @show N
    @show size(L)
    @show size(Array(Q,N)[1])
    @show size(Array(Q,N)[2])
    @show size(L*Array(Q,N)[2])
    # No affine component to the this system
    analytic_f = analytic_AL \ f2.(x)
    ghost_f = A \ f2.(x)

    # Check that A\f2.(x) is consistent with analytic_AL \ f2.(x)
    @test analytic_f ≈ ghost_f

    # Additionally test that A\f2.(x) ≈ f.(x)
    @test f.(x) ≈ ghost_f ≈ analytic_f

    # Check ldiv!
    f_temp = zeros(N)
    ldiv!(f_temp, A, f2.(x))
    @test f_temp ≈ ghost_f ≈ analytic_f


    # Check that left division with matrices works
    M = [f2.(x) f2.(x)]
    Qx = MultiDimBC(Q, size(M),1)

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

    # Check ldiv!
    f_temp = zeros(N)
    ldiv!(f_temp, A, f2.(x))
    @test f_temp ≈ ghost_f ≈ analytic_f

    # Check \ for Matrix
    M2 = [f2.(x) 2.0*f2.(x) 10.0*f2.(x)]

    s = size(M2)
    Qx = MultiDimBC(Q, size(M2), 1)

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
        i = c2l(I, (length(x)+2, 3)) #helper function, see utils.jl
        j = c2l(I-I1, (length(x), 3))
        analytic_QM[i,j] = 1.0
    end
    analytic_Am = kron(Diagonal(ones(3)), analytic_L)*analytic_QM



    analytic_u = analytic_AL \ (u - analytic_Ab)
    ghost_u = A \ u

    # Check that A\u.(x) is consistent with analytic_AL \ u.(x)
    @test analytic_u ≈ ghost_u

    # Check ldiv!
    u_temp = zeros(N)
    ldiv!(u_temp, A, u)
    @test u_temp ≈ ghost_u ≈ analytic_u



    M2 = [u 2.0*u 10.0*u]
    s = size(M2)
    Qx = MultiDimBC(Q, size(M2), 1)
    Am = L*Qx
    #Somehow the operator is singular
    @test_broken analytic_M = analytic_Am \ (reshape(M2, prod(s)) .-repeat(analytic_Ab, 3))
    @test_broken ghost_M = Am \ M2
    @test_broken reshape(analytic_M, s) ≈ ghost_M

end

@testset "Test Operator and BC combinations" begin
    N = 40
    x = range(-pi, stop = pi, length=N)
    Δx = x[2]-x[1]
    u₀=1.0
    Γ=1.0
    Dx=u₀*CenteredDifference{1}(1,2,Δx,N)
    Dxx=Γ*CenteredDifference{1}(2,2,Δx,N)
    Q=PeriodicBC(Float64)

    A = Dx*Q + Dxx*Q
    y = A*(x.^2)

    analytic_y = 2x+2

    @test y ≈ analytic_y
end
