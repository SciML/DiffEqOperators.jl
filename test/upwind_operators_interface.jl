using SparseArrays, DiffEqOperators, LinearAlgebra, Random,
      Test, BandedMatrices, FillArrays

function second_derivative_stencil(N)
  A = zeros(N,N+2)
  A[1,1:5] = [0.916667,   -1.66667,    0.5,   0.333333,  -0.0833333]
  A[2,1:5] = [-0.0833333,   1.33333,   -2.5,   1.33333,   -0.0833333]
  A[3,1:5] = [-0.0833333,   0.333333,   0.5,  -1.66667,    0.916667]
  A[4,1:5] = [0.916667,   -4.66667,    9.5,  -8.66667,    2.91667]
  for i in 5:N-4, j in 5:N-2
      (j-i==0 || j-i==2) && (A[i,j]=1)
      j-i==1 && (A[i,j]=-2)
  end
  A[end,end-4:end] = reverse(A[1,1:5])
  A[end-1,end-4:end] = reverse(A[2,1:5])
  A[end-2,end-4:end] = reverse(A[3,1:5])
  A[end-3,end-4:end] = reverse(A[4,1:5])

  return A
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
enda

@testset "Taking derivatives" begin
    N = 20
    x = 0:1/(N-1):1
    y = 2x.^2 .- 3x .+2
    y_ = y[2:end-1]
    # y = x.^2

    # Dirichlet BC with fixed end points
    Q = RobinBC([1.0, 0.0, y[1]], [1.0, 0.0, y[end]], 1.0)
    U = UpwindDifference(1,2, 1.0, N-2, t->1.0)
    A = CenteredDifference(1,2, 1.0, N-2)
    D1 = CenteredDifference(1,2, 1.0, N-4) # For testing whether the array is constant

    res1 = U*Q*y_
    res2 = A*Q*y_
    @test res1[3:end-2] ≈ res2[1:end-4] # shifted due to upwind operators
    # It is shifted by a constant value so its first derivative has to be 0
    @test D1*(res1[3:end-2] - res2[3:end-2]) ≈ zeros(12) atol=10.0^(-6)

    y = 3x.^3 .- 4x.^2 .+ 2x .+ 1
    y_ = y[2:end-1]
    Q = RobinBC([1.0, 0.0, y[1]], [1.0, 0.0, y[end]], 1.0)
    U = UpwindDifference(2,2, 1.0, N-2, t->1.0)
    A = CenteredDifference(2,2, 1.0, N-2)
    res1 = U*Q*y_
    res2 = A*Q*y_
    @test res1 ≈ res2 # shifted due to upwind operators

    # CAN ADD MORE TESTS
end

# tests for full and sparse function.... BROKEN!
@testset "Full and Sparse functions:" begin
    N = 10
    d_order = 2
    approx_order = 2
    correct = second_derivative_stencil(N)
    A = UpwindDifference(d_order,approx_order,1.0,N,t->1.0)

    @test convert_by_multiplication(Array,A,N) ≈ correct atol=10.0^(-4)
    @test Array(A) ≈ second_derivative_stencil(N) atol=10.0^(-4)
    @test sparse(A) ≈ second_derivative_stencil(N) atol=10.0^(-4)
    @test BandedMatrix(A) ≈ second_derivative_stencil(N) atol=10.0^(-4)
    @test_broken opnorm(A, Inf) ≈ opnorm(correct, Inf) atol=10.0^(-4)

    # testing higher derivative and approximation concretization
    N = 20
    d_order = 4
    approx_order = 4
    A = UpwindDifference(d_order,approx_order,1.0,N,t->1.0)
    correct = convert_by_multiplication(Array,A,N)

    @test Array(A) ≈ correct
    @test sparse(A) ≈ correct
    @test BandedMatrix(A) ≈ correct

    N = 100
    d_order = 8
    approx_order = 8
    A = UpwindDifference(d_order,approx_order,1.0,N,t->1.0)
    correct = convert_by_multiplication(Array,A,N)

    @test Array(A) ≈ correct
    @test sparse(A) ≈ correct
    @test BandedMatrix(A) ≈ correct

    # testing correctness of multiplication
    N = 1000
    d_order = 4
    approx_order = 10
    y = collect(1:1.0:N+2).^4 - 2*collect(1:1.0:N+2).^3 + collect(1:1.0:N+2).^2;
    y = convert(Array{BigFloat, 1}, y)

    A = UpwindDifference(d_order,approx_order,one(BigFloat),N,t->1.0)
    correct = convert_by_multiplication(Array,A,N)
    @test Array(A) ≈ correct
    @test sparse(correct) ≈ correct
    @test BandedMatrix(A) ≈ correct
    @test A*y ≈ Array(A)*y
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

    d_order = 4
    approx_order = 10

    A = UpwindDifference(d_order,approx_order,1.0,N,t->1.0)
    M = Array(A,N)
    @test A[1,1] == M[1,1]
    @test A[1:4,1] == M[1:4,1]
    @test A[5,2:10] == M[5,2:10]
    @test A[524,:] == M[524,:]
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
