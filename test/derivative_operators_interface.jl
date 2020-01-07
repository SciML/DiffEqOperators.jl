using SparseArrays, DiffEqOperators, LinearAlgebra, Random,
      Test, BandedMatrices, FillArrays


# Analytic solutions to higher order operators.
# Do not modify unless you are completely certain of the changes.

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

# analytically derived central difference operators for the irregular grid:
# [0.0, 0.08, 0.1, 0.15, 0.19, 0.26, 0.29], and spacing:
# dx = [0.08, 0.02, 0.05, 0.04, 0.07, 0.03]
# with convention that the first number is the derivative order
# and the second is the approximation order

function analyticCtrOneTwoIrr()
    A = zeros(5,7)
    A[1,1:3] = [-5. / 2., -75. / 2., 40.]
    A[2,2:4] = [-250. / 7., 30., 40. / 7.]
    A[3,3:5] = [-80. / 9., -5., 125. / 9.]
    A[4,4:6] = [-1225. / 77., 825. / 77., 400. / 77.]
    A[5,5:7] = [-30. / 7., -400. / 21., 70. / 3.]
    A
end

function analyticCtrTwoTwoIrr()
    A = zeros(5,7)
    A[1,1:3] = [250., -1250., 1000.]
    A[2,2:4] = [10000. / 7., -2000., 4000. / 7.]
    A[3,3:5] = [4000. / 9., -1000., 5000. / 9.]
    A[4,4:6] = [454. + 42. / 77., -5000. / 7., 20000. / 77.]
    A[5,5:7] = [6000. / 21., -20000. / 21., 2000. / 3.]
    A
end

function analyticCtrTwoFourIrr()
    A = zeros(3,7)
    A[1,1:5] = [14. + 12012. / 13167., 1542. + 2736. / 13167.,
                -2288. - 11704. / 13167., 838. + 1254. / 13167.,
                -106. - 4298. / 13167.]
    A[2,2:6] = [-223. - 461. / 693., 847. + 154. / 693.,
                -1311. - 477. / 693., 699. + 593. / 693.,
                -11. - 502. / 693.]
    A[3,3:7] = [2. + 12166. / 13167., 538. + 12654. / 13167.,
                -912. - 9196. / 13167., 508. + 8664. / 13167.,
                -137. - 11121. / 13167.]
    A
end

function analyticCtrFourTwoIrr()
    A = zeros(3,7)
    A[1,1:5] = [462000000. / 4389.,
                -8550000000. / 4389.,
                11704000000. / 4389.,
                -5016000000. / 4389.,
                1400000000. / 4389.]
    A[2,2:6] = [200000000. / 231.,
                -385000000. / 231.,
                360000000. / 231.,
                -200000000. / 231.,
                25000000. / 231.]
    A[3,3:7] = [770000000. / 4389.,
                -3420000000. / 4389.,
                4180000000. / 4389.,
                -2850000000. / 4389.,
                1320000000 / 4389.]
    A
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
@testset "Correctness of Uniform Stencils" begin
    N = 20
    L = CenteredDifference(4,4, 1.0, N)
    correct = fourth_deriv_approx_stencil(N)

    # Check that stencils (according to convert_by_multiplication) agree with correct
    @test convert_by_multiplication(Array, L, N) ≈ correct

    # Check that concretization agrees correct
    @test Array(L) ≈ correct
    @test sparse(L) ≈ correct
    @test BandedMatrix(L) ≈ correct

    L = CenteredDifference(2,4, 1.0, N)
    correct = second_deriv_fourth_approx_stencil(N)

    # Check that stencils (according to convert_by_multiplication) agree with correct
    @test convert_by_multiplication(Array, L, N) ≈ correct

    # Check that concretization agrees correct
    @test Array(L) ≈ correct
    @test sparse(L) ≈ correct
    @test BandedMatrix(L) ≈ correct
end

@testset "Correctness of Non-Uniform Stencils" begin
    x = [0., 0.08, 0.1, 0.15, 0.19, 0.26, 0.29]
    dx = diff(x)

    # Second-Order First Derivative
    L = CenteredDifference(1, 2, dx, 5)
    correct = analyticCtrOneTwoIrr()

    # Check that stencils agree with correct
    for (i,coefs) in enumerate(L.stencil_coefs)
        @test Array(coefs) ≈ correct[i,correct[i,:] .!= 0.]
    end
    @test_broken Array(L) ≈ correct # All of these concretizations
    @test_broken sparse(L) ≈ correct # only give the first three
    @test_broken BandedMatrix(L) ≈ correct # rows of the computed stencil coefficients

    # Second-Order Second Derivative
    L = CenteredDifference(2, 2, dx, 5)
    correct = analyticCtrTwoTwoIrr()

    # Check that stencils agree with correct
    for (i,coefs) in enumerate(L.stencil_coefs)
        @test Array(coefs) ≈ correct[i,correct[i,:] .!= 0.]
    end
    @test_broken Array(L) ≈ correct # same issue as previous derivative
    @test_broken sparse(L) ≈ correct
    @test_broken BandedMatrix(L) ≈ correct

    # Fourth-Order Second Derivative
    L = CenteredDifference(2, 4, dx, 5)
    correct = analyticCtrTwoFourIrr()

    # Check that stencils agree with correct
    for (i,coefs) in enumerate(L.stencil_coefs)
        @test Array(coefs) ≈ correct[i,correct[i,:] .!= 0.]
    end
    @test_broken Array(L) ≈ correct # L.stencil_coefs is populated, but the concretization doesn't work. It appears to be an issue of improper calculation of indexing from the various lengths computed during construction (e.g. boundary_stencil_length, len) and potentially the fact that "len" doesn't seem to specify the number of grid points at which we compute finite differences but appears to specify the location of the last grid point at which we compute finite differences (so if X is a 5-length vector, entering len = 2 means computing FDs for X[2] and X[3])
    @test_broken sparse(L) ≈ correct
    @test_broken BandedMatrix(L) ≈ correct

    # Second-Order Fourth Derivative
    L = CenteredDifference(4, 2, dx, 5)
    correct = analyticCtrFourTwoIrr()

    # Check that stencils agree with correct
    for (i,coefs) in enumerate(L.stencil_coefs)
        @test Array(coefs) ≈ correct[i,correct[i,:] .!= 0.]
    end
    @test_broken Array(L) ≈ correct
    @test_broken sparse(L) ≈ correct
    @test_broken BandedMatrix(L) ≈ correct
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

@testset "Operations on matrices" begin
    N = 51
    M = 101
    d_order = 2
    approx_order = 2

    xarrs = Vector{Union{Array,StepRangeLen}}(undef,0)
    yarrs = Vector{Union{Array,StepRangeLen}}(undef,0)
    push!(xarrs, range(0,stop=1,length=N))
    push!(yarrs, range(0,stop=1,length=M))
    push!(xarrs, vcat(xarrs[1][1:floor(Int,N/2)].^0.2020, xarrs[1][ceil(Int,N/2):end].^2.015))
    push!(yarrs, vcat(yarrs[1][1:floor(Int,M/2)].^1.793, yarrs[1][ceil(Int,M/2):end].^2.019))

    for (i,xarr,yarr) in zip(1:2,xarrs,yarrs)
        if i == 1
            dx = xarr[2] - xarr[1]
            dy = yarr[2] - yarr[1]
        elseif i == 2
            dx = diff(xarr)
            dy = diff(yarr)
        end
        F = [x^2+y for x = xarr, y = yarr]

        A = CenteredDifference(d_order,approx_order,dx,length(xarr)-2)
        B = i == 1 ? CenteredDifference(d_order,approx_order,dy,length(yarr)) :
            CenteredDifference(d_order,approx_order,dy,length(yarr)-2)

        @test A*F ≈ 2*ones(N-2,M) atol=1e-9

        # Operators are defined such that
        # B must be applied to F from the left
        @test all(abs.(B * F') .<= 1e-9)
        @test all(abs.(A * (B * F')') .<= 1e-4)

        G = [x^2+y^2 for x = xarr, y = yarr]

        @test A*G ≈ 2*ones(N-2,M) atol=1e-9
        @test B * G' ≈ 2 * ones(M-2,N) atol=1e-8
        @test all(abs.(A * (B * G')') .<= 1e-4)
    end
end

# These tests are broken due to the implementation 2.2*LD creating a DerivativeOperator
# rather than an Array
#=
@testset "Linear combinations of operators" begin
    N = 10
    Random.seed!(0); LA = DiffEqArrayOperator(rand(N,N+2))
    LD = CenteredDifference(2,2,1.0,N)
    L = 1.1*LA - 2.2*LD + 3.3*Eye(N,N+2)
    # Builds convert(L) the brute-force way
    fullL = zeros(N,N+2)
    v = zeros(N+2)
    for i = 1:N+2
        v[i] = 1.0
        fullL[:,i] = L*v
        v[i] = 0.0
    end
    @test_broken convert(AbstractMatrix,L) ≈ fullL
    for p in [1,2,Inf]
        @test_broken opnorm(L,p) ≈ opnorm(fullL,p)
    end
end
=#
