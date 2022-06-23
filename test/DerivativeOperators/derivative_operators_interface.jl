using SparseArrays, DiffEqOperators, LinearAlgebra, Random, Test, BandedMatrices, FillArrays


# Analytic solutions to higher order operators.
# Do not modify unless you are completely certain of the changes.

# 4th derivative, 4th order
function fourth_deriv_approx_stencil(N)
    A = zeros(N, N + 2)
    A[1, 1:8] = [3.5 -56 / 3 42.5 -54.0 251 / 6 -20.0 5.5 -2 / 3]
    A[2, 1:8] = [2 / 3 -11 / 6 0.0 31 / 6 -22 / 3 4.5 -4 / 3 1 / 6]

    A[N-1, N-5:end] = reverse([2 / 3 -11 / 6 0.0 31 / 6 -22 / 3 4.5 -4 / 3 1 / 6], dims = 2)
    A[N, N-5:end] = reverse([3.5 -56 / 3 42.5 -54.0 251 / 6 -20.0 5.5 -2 / 3], dims = 2)

    for i = 3:N-2
        A[i, i-2:i+4] = [-1 / 6 2.0 -13 / 2 28 / 3 -13 / 2 2.0 -1 / 6]
    end
    return A
end


function second_deriv_fourth_approx_stencil(N)
    A = zeros(N, N + 2)
    A[1, 1:6] = [5 / 6 -15 / 12 -1 / 3 7 / 6 -6 / 12 5 / 60]
    A[N, N-3:end] = [1 / 12 -6 / 12 14 / 12 -4 / 12 -15 / 12 10 / 12]
    for i = 2:N-1
        A[i, i-1:i+3] = [-1 / 12 4 / 3 -5 / 2 4 / 3 -1 / 12]
    end
    return A
end

function second_derivative_stencil(N)
    A = zeros(N, N + 2)
    for i = 1:N, j = 1:N+2
        (j - i == 0 || j - i == 2) && (A[i, j] = 1)
        j - i == 1 && (A[i, j] = -2)
    end
    A
end

# analytically derived central difference operators for the irregular grid:
# [0.0, 0.08, 0.1, 0.15, 0.19, 0.26, 0.29], and spacing:
# dx = [0.08, 0.02, 0.05, 0.04, 0.07, 0.03]
# with convention that the first number is the derivative order
# and the second is the approximation order

function analyticCtrOneTwoIrr()
    A = zeros(5, 7)
    A[1, 1:3] = [-5.0 / 2.0, -75.0 / 2.0, 40.0]
    A[2, 2:4] = [-250.0 / 7.0, 30.0, 40.0 / 7.0]
    A[3, 3:5] = [-80.0 / 9.0, -5.0, 125.0 / 9.0]
    A[4, 4:6] = [-1225.0 / 77.0, 825.0 / 77.0, 400.0 / 77.0]
    A[5, 5:7] = [-30.0 / 7.0, -400.0 / 21.0, 70.0 / 3.0]
    A
end

function analyticCtrTwoTwoIrr()
    A = zeros(5, 7)
    A[1, 1:3] = [250.0, -1250.0, 1000.0]
    A[2, 2:4] = [10000.0 / 7.0, -2000.0, 4000.0 / 7.0]
    A[3, 3:5] = [4000.0 / 9.0, -1000.0, 5000.0 / 9.0]
    A[4, 4:6] = [454.0 + 42.0 / 77.0, -5000.0 / 7.0, 20000.0 / 77.0]
    A[5, 5:7] = [6000.0 / 21.0, -20000.0 / 21.0, 2000.0 / 3.0]
    A
end

function analyticCtrTwoFourIrr()
    A = zeros(3, 7)
    A[1, 1:5] = [
        14.0 + 12012.0 / 13167.0,
        1542.0 + 2736.0 / 13167.0,
        -2288.0 - 11704.0 / 13167.0,
        838.0 + 1254.0 / 13167.0,
        -106.0 - 4298.0 / 13167.0,
    ]
    A[2, 2:6] = [
        -223.0 - 461.0 / 693.0,
        847.0 + 154.0 / 693.0,
        -1311.0 - 477.0 / 693.0,
        699.0 + 593.0 / 693.0,
        -11.0 - 502.0 / 693.0,
    ]
    A[3, 3:7] = [
        2.0 + 12166.0 / 13167.0,
        538.0 + 12654.0 / 13167.0,
        -912.0 - 9196.0 / 13167.0,
        508.0 + 8664.0 / 13167.0,
        -137.0 - 11121.0 / 13167.0,
    ]
    A
end

function analyticCtrFourTwoIrr()
    A = zeros(3, 7)
    A[1, 1:5] = [
        462000000.0 / 4389.0,
        -8550000000.0 / 4389.0,
        11704000000.0 / 4389.0,
        -5016000000.0 / 4389.0,
        1400000000.0 / 4389.0,
    ]
    A[2, 2:6] = [
        200000000.0 / 231.0,
        -385000000.0 / 231.0,
        360000000.0 / 231.0,
        -200000000.0 / 231.0,
        25000000.0 / 231.0,
    ]
    A[3, 3:7] = [
        770000000.0 / 4389.0,
        -3420000000.0 / 4389.0,
        4180000000.0 / 4389.0,
        -2850000000.0 / 4389.0,
        1320000000 / 4389.0,
    ]
    A
end


function convert_by_multiplication(
    ::Type{Array},
    A::AbstractDerivativeOperator{T},
    N::Int = A.dimension,
) where {T}
    @assert N >= A.stencil_length # stencil must be able to fit in the matrix
    mat = zeros(T, (N, N + 2))
    v = zeros(T, N + 2)
    for i = 1:N+2
        v[i] = one(T)
        #=
            calculating the effect on a unit vector to get the matrix of transformation
            to get the vector in the new vector space.
        =#
        mul!(view(mat, :, i), A, v)
        v[i] = zero(T)
    end
    return mat
end

# Tests the corrrectness of stencils, along with concretization.
# Do not modify the following test-set unless you are completely certain of your changes.
@testset "Correctness of Uniform Stencils" begin
    N = 20
    L = CenteredDifference(4, 4, 1.0, N)
    correct = fourth_deriv_approx_stencil(N)

    # Check that stencils (according to convert_by_multiplication) agree with correct
    @test convert_by_multiplication(Array, L, N) ≈ correct

    # Check that concretization agrees correct
    @test Array(L) ≈ correct
    @test sparse(L) ≈ correct
    @test BandedMatrix(L) ≈ correct

    L = CenteredDifference(2, 4, 1.0, N)
    correct = second_deriv_fourth_approx_stencil(N)

    # Check that stencils (according to convert_by_multiplication) agree with correct
    @test convert_by_multiplication(Array, L, N) ≈ correct

    # Check that concretization agrees correct
    @test Array(L) ≈ correct
    @test sparse(L) ≈ correct
    @test BandedMatrix(L) ≈ correct
end

@testset "Correctness of Uniform Stencils, Complete" begin
    weights = []

    push!(weights, ([-0.5, 0, 0.5], [1.0, -2.0, 1.0], [-1 / 2, 1.0, 0.0, -1.0, 1 / 2]))
    push!(
        weights,
        (
            [1 / 12, -2 / 3, 0, 2 / 3, -1 / 12],
            [-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12],
            [1 / 8, -1.0, 13 / 8, 0.0, -13 / 8, 1.0, -1 / 8],
        ),
    )

    for d = 1:3
        for (i, a) in enumerate([2, 4])
            D = CompleteCenteredDifference(d, a, 1.0)

            @test all(isapprox.(D.stencil_coefs, weights[i][d], atol = 1e-10))
        end
    end
end

@testset "Correctness of Non-Uniform Stencils, Complete" begin
    weights = []

    push!(weights, ([-0.5, 0, 0.5], [1.0, -2.0, 1.0], [-1 / 2, 1.0, 0.0, -1.0, 1 / 2]))
    push!(
        weights,
        (
            [1 / 12, -2 / 3, 0, 2 / 3, -1 / 12],
            [-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12],
            [1 / 8, -1.0, 13 / 8, 0.0, -13 / 8, 1.0, -1 / 8],
        ),
    )

    for d = 1:3
        for (i, a) in enumerate([2, 4])
            D = CompleteCenteredDifference(d, a, 0.0:1.0:10.0)
            @show D.stencil_coefs

            @test all(isapprox.(D.stencil_coefs[end], weights[i][d], atol = 1e-10))
        end
    end
end

@testset "Correctness of Uniform Stencils, Complete Half" begin
    weights = (
        ([0.5, 0.5], [-1 / 16, 9 / 16, 9 / 16, -1 / 16]),
        ([-1.0, 1.0], [1 / 24, -9 / 8, 9 / 8, -1 / 24]),
    )
    for (i, a) in enumerate([2, 4])
        for d = 0:1
            D = CompleteHalfCenteredDifference(d, a, 1.0)
            @test all(D.stencil_coefs .≈ weights[d+1][i])
        end
    end

end

@testset "Correctness of Non-Uniform Stencils, Complete Half" begin
    weights = (
        ([0.5, 0.5], [-1 / 16, 9 / 16, 9 / 16, -1 / 16]),
        ([-1.0, 1.0], [1 / 24, -9 / 8, 9 / 8, -1 / 24]),
    )
    for (i, a) in enumerate([2, 4])
        for d = 0:1
            D = CompleteHalfCenteredDifference(d, a, 0.0:1.0:5.0)
            @test all(D.stencil_coefs[end] .≈ weights[d+1][i])
        end
    end

end


@testset "Correctness of Uniform Upwind Stencils" begin
    weights = (
        ([-1.0, 1.0], [-1.0, 3.0, -3.0, 1.0]),
        ([-3 / 2, 2.0, -1 / 2], [-5 / 2, 9.0, -12.0, 7.0, -3 / 2]),
    )
    for (i, a) in enumerate(1:2)
        for (j, d) in enumerate([1, 3])
            D = CompleteUpwindDifference(d, a, 1.0, 0)
            @test all(isapprox.(D.stencil_coefs, weights[i][j], atol = 1e-10))
        end
    end
end

@testset "Correctness of Non-Uniform Upwind Stencils" begin
    weights = (
        ([-1.0, 1.0], [-1.0, 3.0, -3.0, 1.0]),
        ([-3 / 2, 2.0, -1 / 2], [-5 / 2, 9.0, -12.0, 7.0, -3 / 2]),
    )
    for (i, a) in enumerate(1:2)
        for (j, d) in enumerate([1, 3])
            D = CompleteUpwindDifference(d, a, 0.0:1.0:10.0, 0)
            @test all(isapprox.(D.stencil_coefs[end], weights[i][j], atol = 1e-10))
        end
    end
end


@testset "Correctness of Non-Uniform Stencils" begin
    x = [0.0, 0.08, 0.1, 0.15, 0.19, 0.26, 0.29]
    nx = length(x)
    dx = diff(x)

    # Second-Order First Derivative
    L = CenteredDifference(1, 2, dx, nx - 2)
    correct = analyticCtrOneTwoIrr()

    # Check that stencils agree with correct
    for (i, coefs) in enumerate(L.stencil_coefs)
        @test Array(coefs) ≈ correct[i, correct[i, :].!=0.0]
    end
    @test Array(L) ≈ correct
    @test sparse(L) ≈ correct
    @test BandedMatrix(L) ≈ correct

    # Second-Order Second Derivative
    L = CenteredDifference(2, 2, dx, nx - 2)
    correct = analyticCtrTwoTwoIrr()

    # Check that stencils agree with correct
    for (i, coefs) in enumerate(L.stencil_coefs)
        @test Array(coefs) ≈ correct[i, correct[i, :].!=0.0]
    end
    @test Array(L) ≈ correct
    @test sparse(L) ≈ correct
    @test BandedMatrix(L) ≈ correct

    # Fourth-Order Second Derivative
    L = CenteredDifference(2, 4, dx, nx - 2)
    correct = analyticCtrTwoFourIrr()

    # Check that stencils agree with correct
    for (i, coefs) in enumerate(L.stencil_coefs)
        @test Array(coefs) ≈ correct[i, correct[i, :].!=0.0]
    end
    @test Array(L)[2:end-1, :] ≈ correct
    @test sparse(L)[2:end-1, :] ≈ correct
    @test BandedMatrix(L)[2:end-1, :] ≈ correct

    # Second-Order Fourth Derivative
    L = CenteredDifference(4, 2, dx, nx - 2)
    correct = analyticCtrFourTwoIrr()

    # Check that stencils agree with correct
    for (i, coefs) in enumerate(L.stencil_coefs)
        @test Array(coefs) ≈ correct[i, correct[i, :].!=0.0]
    end
    @test Array(L)[2:end-1, :] ≈ correct
    @test sparse(L)[2:end-1, :] ≈ correct
    @test BandedMatrix(L)[2:end-1, :] ≈ correct
end

# tests for full and sparse function
@testset "Full and Sparse functions:" begin
    N = 10
    d_order = 2
    approx_order = 2
    correct = second_derivative_stencil(N)
    A = CenteredDifference(d_order, approx_order, 1.0, N)

    @test convert_by_multiplication(Array, A, N) == correct
    @test Array(A) == second_derivative_stencil(N)
    @test sparse(A) == second_derivative_stencil(N)
    @test BandedMatrix(A) == second_derivative_stencil(N)
    @test opnorm(A, Inf) == opnorm(correct, Inf)


    # testing higher derivative and approximation concretization
    N = 20
    d_order = 4
    approx_order = 4
    A = CenteredDifference(d_order, approx_order, 1.0, N)
    correct = convert_by_multiplication(Array, A, N)

    @test Array(A) ≈ correct
    @test sparse(A) ≈ correct
    @test BandedMatrix(A) ≈ correct

    N = 26
    d_order = 8
    approx_order = 8
    A = CenteredDifference(d_order, approx_order, 1.0, N)
    correct = convert_by_multiplication(Array, A, N)

    @test Array(A) ≈ correct
    @test sparse(A) ≈ correct
    @test BandedMatrix(A) ≈ correct

    # testing correctness of multiplication
    N = 1000
    d_order = 4
    approx_order = 10
    y = collect(1:1.0:N+2) .^ 4 - 2 * collect(1:1.0:N+2) .^ 3 + collect(1:1.0:N+2) .^ 2
    y = convert(Array{BigFloat,1}, y)

    A = CenteredDifference(d_order, approx_order, one(BigFloat), N)
    correct = convert_by_multiplication(Array, A, N)
    @test Array(A) ≈ correct
    @test sparse(A) ≈ correct
    @test BandedMatrix(A) ≈ correct
    @test A * y ≈ Array(A) * y
end

@testset "Indexing tests" begin
    N = 1000
    d_order = 4
    approx_order = 10

    A = CenteredDifference(d_order, approx_order, 1.0, N)
    @test A[1, 1] == Array(A)[1, 1]
    @test A[10, 20] == 0

    correct = Array(A)
    for i = 1:N
        @test A[i, i] == correct[i, i]
    end

    # Indexing Tests
    N = 1000
    d_order = 2
    approx_order = 2

    A = CenteredDifference(d_order, approx_order, 1.0, N)
    M = Array(A, 1000)
    @test A[1, 1] == M[1, 1]
    @test A[1:4, 1] == M[1:4, 1]
    @test A[5, 2:10] == M[5, 2:10]
    @test A[60:100, 500:600] == M[60:100, 500:600]
end

@testset "Operations on matrices" begin
    N = 51
    M = 101
    d_order = 2
    approx_order = 2

    xarrs = Vector{Union{Array,StepRangeLen}}(undef, 0)
    yarrs = Vector{Union{Array,StepRangeLen}}(undef, 0)
    push!(xarrs, range(0, stop = 1, length = N))
    push!(yarrs, range(0, stop = 1, length = M))
    push!(
        xarrs,
        vcat(
            xarrs[1][1:floor(Int, N / 2)] .^ 0.2020,
            xarrs[1][ceil(Int, N / 2):end] .^ 2.015,
        ),
    )
    push!(
        yarrs,
        vcat(
            yarrs[1][1:floor(Int, M / 2)] .^ 1.793,
            yarrs[1][ceil(Int, M / 2):end] .^ 2.019,
        ),
    )

    for (i, xarr, yarr) in zip(1:2, xarrs, yarrs)
        if i == 1
            dx = xarr[2] - xarr[1]
            dy = yarr[2] - yarr[1]
        elseif i == 2
            dx = diff(xarr)
            dy = diff(yarr)
        end
        F = [x^2 + y for x in xarr, y in yarr]

        A = CenteredDifference(d_order, approx_order, dx, length(xarr) - 2)
        B =
            i == 1 ? CenteredDifference(d_order, approx_order, dy, length(yarr)) :
            CenteredDifference(d_order, approx_order, dy, length(yarr) - 2)

        @test A * F ≈ 2 * ones(N - 2, M) atol = 1e-9

        # Operators are defined such that
        # B must be applied to F from the left
        @test all(abs.(B * F') .<= 1e-8)
        @test all(abs.(A * (B * F')') .<= 1e-4)

        G = [x^2 + y^2 for x in xarr, y in yarr]

        @test A * G ≈ 2 * ones(N - 2, M) atol = 1e-9
        @test B * G' ≈ 2 * ones(M - 2, N) atol = 1e-8
        @test all(abs.(A * (B * G')') .<= 1e-4)
    end
end

@testset "Scalar multiplication with operators" begin
    A = CenteredDifference(2, 2, 10.0, 3)

    scalar_ans = [
        0.033 -0.066 0.033 0.0 0.0
        0.0 0.033 -0.066 0.033 0.0
        0.0 0.0 0.033 -0.066 0.033
    ]
    @test Array(3.3 * A) ≈ scalar_ans
end

@testset "Left-multiplying operators with a vector of coefficients" begin
    A = CenteredDifference(2, 2, 1.0, 3)
    B = UpwindDifference(1, 1, 1.0, 3, 1.0)
    C = UpwindDifference(1, 1, 1.0, 3, [2.0, 3.0, 4.0])
    c = [1.0, 1.0, 3.0]
    cA = c * A
    cB = c * B
    cC = c * C

    @test cA.coefficients == c
    @test cB.coefficients == c
    @test cC.coefficients == [2.0, 3.0, 12.0]

    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    @test (A * x) .* c ≈ cA * x
    @test (B * x) .* c ≈ cB * x
    @test (C * x) .* c ≈ cC * x

    @test_throws DimensionMismatch ones(10) * A
    @test_throws DimensionMismatch ones(10) * cA
    @test_throws DimensionMismatch ones(10) * B
    @test_throws DimensionMismatch ones(10) * C
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
