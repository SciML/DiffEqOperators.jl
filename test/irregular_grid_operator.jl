using SparseArrays, DiffEqOperators, LinearAlgebra, Random,
      Test, BandedMatrices, FillArrays

function second_derivative_stencil(N)
  A = zeros(N,N+2)
  for i in 1:N, j in 1:N+2
      (j-i==0 || j-i==2) && (A[i,j]=1)
      j-i==1 && (A[i,j]=-2)
  end
  A
end

# Analytic solutions to higher order operators.
# Do not modify unless you are completely certain of the changes.
function fourth_deriv_approx_stencil(N, scale)
    A = zeros(N,N+2)
    A[1,1:8] = [3.5 -56/3 42.5 -54.0 251/6 -20.0 5.5 -2/3]
    A[2,1:8] = [2/3 -11/6 0.0 31/6 -22/3 4.5 -4/3 1/6]
    A[N-1,N-5:end] = reverse([2/3 -11/6 0.0 31/6 -22/3 4.5 -4/3 1/6], dims=2)
    A[N,N-5:end] = reverse([3.5 -56/3 42.5 -54.0 251/6 -20.0 5.5 -2/3], dims=2)
    for i in 3:N-2
        A[i,i-2:i+4] = [-1/6 2.0 -13/2 28/3 -13/2 2.0 -1/6]
    end
    return A*scale
end

function second_deriv_fourth_approx_stencil(N, scale)
    A = zeros(N,N+2)
    A[1,1:6] = [5/6 -1.25 -1/3 7/6 -0.5 5/60]
    A[N,N-3:end] = reverse([5/6 -1.25 -1/3 7/6 -0.5 5/60], dims=2)
    for i in 2:N-1
        A[i,i-1:i+3] = [-1/12 4/3 -5/2 4/3 -1/12]
    end
    return A*scale
end

# Tests the corrrectness of stencils.
# Do not modify the following test-set unless you are completely certain of your changes.
@testset "Correctness of Stencils on Uniform grid" begin
    N = 20
    dx = ones(Float64, N+1)

    L = CenteredDifference(2,2, dx, N)
    correct = convert(Array{Float64, 1}, [1,-2,1])

    for stencil in L.stencil.coefs
        @test stencil ≈ correct
    end

    L = CenteredDifference(4,4, dx, N)
    correct = fourth_deriv_approx_stencil(N, 1.0)

    # Check that stencils (according to convert_by_multiplication) agree with correct
    @test convert_by_multiplication(Array, L, N) ≈ correct

    # Check that concretization agrees correct
    @test Array(L) ≈ correct
    @test sparse(L) ≈ correct
    @test BandedMatrix(L) ≈ correct

    scale = 0.1
    L = CenteredDifference(2,4, dx/scale, N)
    correct = second_deriv_fourth_approx_stencil(N, scale)

    # Check that stencils (according to convert_by_multiplication) agree with correct
    @test convert_by_multiplication(Array, L, N) ≈ correct

    # Check that concretization agrees correct
    @test Array(L) ≈ correct
    @test sparse(L) ≈ correct
    @test BandedMatrix(L) ≈ correct
end

# Cross check from http://web.media.mit.edu/~crtaylor/calculator.html
@testset "Correctness of Stencils on Non-Uniform grid" begin
    N = 4
    dx = [0.1, 0.2, 0.3, 0.4, 0.5]

    L = CenteredDifference(2,2, dx, N)
    correct_interior = [[200/3,  -300/2,  100/3],
                        [60/3,   -100/3,  40/3],
                        [200/21, -350/21, 150/21],
                        [50/9,   -90/9,   40/9]]

    for i in length(L.stencil_coefs)
        @test L.stencil_coefs[i] ≈ correct_interior[i]
    end
end

# Cross check from http://web.media.mit.edu/~crtaylor/calculator.html
@testset "Correctness of Left Boundary Stencils on Non-Uniform grid" begin
    N = 14
    dx = [0.1*i for i in 1:15]

    L = CenteredDifference(2,3, dx, N)
    correct_left_boundary = [[8890/63,-15120/63,7600/63,-1505/63,135/63],
                            [5110/63,-7980/63,3100/63,-245/63,15/63]]

    for i in length(L.low_boundary_coefs)
        @test L.stencil_coefs[i] ≈ correct_left_boundary[i]
    end
end

# tests for full and sparse function
@testset "Full and Sparse functions:" begin
    N = 10
    d_order = 2
    approx_order = 2
    correct = second_derivative_stencil(N)
    A = CenteredDifference(d_order,approx_order,1.0,N)

    @test convert_by_multiplication(Array,A,N) == correct
    @test Array(A) == second_derivative_stencil(N)
    @test sparse(A) == second_derivative_stencil(N)
    @test BandedMatrix(A) == second_derivative_stencil(N)
    @test opnorm(A, Inf) == opnorm(correct, Inf)


    # testing higher derivative and approximation concretization
    N = 20
    d_order = 4
    approx_order = 4
    A = CenteredDifference(d_order,approx_order,1.0,N)
    correct = convert_by_multiplication(Array,A,N)

    @test Array(A) ≈ correct
    @test sparse(A) ≈ correct
    @test BandedMatrix(A) ≈ correct

    N = 26
    d_order = 8
    approx_order = 8
    A = CenteredDifference(d_order,approx_order,1.0,N)
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

    A = CenteredDifference(d_order,approx_order,one(BigFloat),N)
    correct = convert_by_multiplication(Array,A,N)
    @test Array(A) ≈ correct
    @test sparse(A) ≈ correct
    @test BandedMatrix(A) ≈ correct
    @test A*y ≈ Array(A)*y
end

@testset "Indexing tests" begin
    N = 1000
    d_order = 4
    approx_order = 10

    A = CenteredDifference(d_order,approx_order,1.0,N)
    @test A[1,1] == Array(A)[1,1]
    @test A[10,20] == 0

    correct = Array(A)
    for i in 1:N
        @test A[i,i] == correct[i,i]
    end

    # Indexing Tests
    N = 1000
    d_order = 2
    approx_order = 2

    A = CenteredDifference(d_order,approx_order,1.0,N)
    M = Array(A,1000)
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

    A = CenteredDifference(d_order,approx_order,dx,length(xarr)-2)
    B = CenteredDifference(d_order,approx_order,dy,length(yarr))


    @test A*F ≈ 2*ones(N-2,M) atol=1e-2
    F*B
    A*F*B

    G = [x^2+y^2 for x = xarr, y = yarr]

    @test A*G ≈ 2*ones(N-2,M) atol=1e-2
    G*B
    A*G*B
end
