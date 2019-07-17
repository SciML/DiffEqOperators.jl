using SparseArrays, DiffEqOperators, LinearAlgebra, Random,
      Test, BandedMatrices, FillArrays

function second_derivative_stencil(N)
  A = zeros(N,N+2)
  A[1,1:4] = [-0.0, -1.0, 2.0, -1.0]
  for i in 2:N-1, j in 1:N+2
      (j-i==0 || j-i==2) && (A[i,j]=1)
      j-i==1 && (A[i,j]=-2)
  end
  A[end,end-3:end] = [-1.0, 2.0, -1.0, 0.0]
  A
end

# Analytic solutions to higher order operators.
# Do not modify unless you are completely certain of the changes.
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
    A[1,1:6] = [5/6 -1.25 -1/3 7/6 -0.5 5/60]
    A[N,N-3:end] = reverse([5/6 -1.25 -1/3 7/6 -0.5 5/60], dims=2)
    for i in 2:N-1
        A[i,i-1:i+3] = [-1/12 4/3 -5/2 4/3 -1/12]
    end
    return A
end

function convert_by_multiplication(::Type{Array}, A::AbstractDerivativeOperator{T}, N::Int=A.dimension) where T
    @assert N >= A.stencil_length # stencil must be able to fit in the matrix
    mat = zeros(T, (N, N+2))
    v = zeros(T, N+2)
    for i=1:N+2
        v[i] = one(T)
        #=
            calculating the effect on a unit vector to get the matrix of transformation
            to get the vector in the new vector space.
        =#
        mul!(view(mat,:,i), A, v)
        v[i] = zero(T)
    end
    return mat
end

# Tests the corrrectness of stencils, along with concretization.
# Do not modify the following test-set unless you are completely certain of your changes.
@testset "Correctness of Stencils" begin
    N = 20
    L1 = UpwindDifference(1,2, 1.0, N, t->1.0)
    correct = [-3/2, 2.0, -1/2]
    @test L1.stencil_coefs ≈ correct

    L1 = UpwindDifference(1,3, 1.0, N, t->1.0)
    correct = [-2/6, -3/6, 6/6, -1/6]
    @test L1.stencil_coefs ≈ correct
end

@testset "Taking derivatives" begin
    N = 20
    x = 0:1/(N-1):1
    y = 2x.^2 .- 3x .+2

    # y = x.^2

    # Dirichlet BC with fixed end points
    Q = RobinBC(1.0, 0.0, y[1], 1.0, 1.0, 0.0, y[end], 1.0)
    U = UpwindDifference(1,2, 1.0, N, t->1.0)
    A = CenteredDifference(1,2, 1.0, N)
    res1 = U*Q*y
    res2 = A*Q*y
    @test res1[3:end-3] ≈ res2[1:end-5] atol=10.0^(-2) # shifted due to upwind operators

    y = 3x.^3 .- 4x.^2 .+ 2x .+ 1
    Q = RobinBC(1.0, 0.0, y[1], 1.0, 1.0, 0.0, y[end], 1.0)
    U = UpwindDifference(1,2, 1.0, N, t->1.0)
    A = CenteredDifference(1,2, 1.0, N)
    res1 = U*Q*y
    res2 = A*Q*y
    @test res1[3:end-3] ≈ res2[1:end-5] atol=10.0^(-2) # shifted due to upwind operators
end

# tests for full and sparse function.... BROKEN!
@testset "Full and Sparse functions:" begin
    N = 10
    d_order = 2
    approx_order = 2
    correct = second_derivative_stencil(N)
    A = UpwindDifference(d_order,approx_order,1.0,N,t->1.0)

    @test convert_by_multiplication(Array,A,N) == correct
    @test Array(A) == second_derivative_stencil(N)
    @testbroken sparse(A) == second_derivative_stencil(N)
    @test BandedMatrix(A) == second_derivative_stencil(N)
    @test opnorm(A, Inf) == opnorm(correct, Inf)

    # testing higher derivative and approximation concretization
    N = 20
    d_order = 4
    approx_order = 4
    A = UpwindDifference(d_order,approx_order,1.0,N)
    correct = convert_by_multiplication(Array,A,N)

    @test Array(A) ≈ correct
    @testbroken sparse(A) ≈ correct
    @test BandedMatrix(A) ≈ correct

    # N = 26
    # d_order = 8
    # approx_order = 8
    # A = UpwindDifference(d_order,approx_order,1.0,N)
    # correct = convert_by_multiplication(Array,A,N)

    # @test Array(A) ≈ correct
    # @test sparse(A) ≈ correct
    # @test BandedMatrix(A) ≈ correct

    # # testing correctness of multiplication
    # N = 1000
    # d_order = 4
    # approx_order = 10
    # y = collect(1:1.0:N+2).^4 - 2*collect(1:1.0:N+2).^3 + collect(1:1.0:N+2).^2;
    # y = convert(Array{BigFloat, 1}, y)

    # A = UpwindDifference(d_order,approx_order,one(BigFloat),N)
    # correct = convert_by_multiplication(Array,A,N)
    # @test Array(A) ≈ correct
    # @test sparse(A) ≈ correct
    # @test BandedMatrix(A) ≈ correct
    # @test A*y ≈ Array(A)*y
end

@testset "Indexing tests" begin
    N = 1000
    d_order = 4
    approx_order = 10

    A = UpwindDifference(d_order,approx_order,1.0,N,t->1.0)
    @test A[1,1] == Array(A)[1,1]
    @test A[10,20] == 0.0

    correct = Array(A)
    for i in 1:N
        @test A[i,i] ≈ correct[i,i] atol=10^-9
    end

    # Indexing Tests
    N = 1000
    d_order = 2
    approx_order = 2

    A = UpwindDifference(d_order,approx_order,1.0,N,t->1.0)
    M = Array(A,N)
    @test A[1,1] == M[1,1]
    @test A[1:4,1] == M[1:4,1]
    @test A[5,2:10] == M[5,2:10]
    @test A[60:100,500:600] == M[60:100,500:600]
end

@testset begin "Operations on matrices"
    N = 51
    M = 101
    d_order = 2
    approx_order = 2

    xarr = range(0,stop=1,length=N)
    yarr = range(0,stop=1,length=M)
    dx = xarr[2]-xarr[1]
    dy = yarr[2]-yarr[1]
    F = [x^2+y for x = xarr, y = yarr]

    A = UpwindDifference(d_order,approx_order,dx,length(xarr)-2,t->1.0)
    B = UpwindDifference(d_order,approx_order,dy,length(yarr),t->1.0)


    @test A*F ≈ 2*ones(N-2,M)
    F*B
    A*F*B

    G = [x^2+y^2 for x = xarr, y = yarr]

    @test A*G ≈ 2*ones(N-2,M) atol=1e-2
    G*B
    A*G*B
end

@testset "Linear combinations of operators" begin
    N = 10
    Random.seed!(0); LA = DiffEqArrayOperator(rand(N,N+2))
    LD = UpwindDifference(2,2,1.0,N,t->1.0)
    L = 1.1*LA - 2.2*LD + 3.3*Eye(N,N+2)
    # Builds convert(L) the brute-force way
    fullL = zeros(N,N+2)
    v = zeros(N+2)
    for i = 1:N+2
        v[i] = 1.0
        fullL[:,i] = L*v
        v[i] = 0.0
    end
    @test convert(AbstractMatrix,L) ≈ fullL
    for p in [1,2,Inf]
        @test opnorm(L,p) ≈ opnorm(fullL,p) atol=0.1
    end
end
