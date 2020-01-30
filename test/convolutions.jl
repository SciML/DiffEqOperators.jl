using SparseArrays, DiffEqOperators, LinearAlgebra, Random,
      Test, BandedMatrices, FillArrays

import DiffEqOperators: BoundaryPaddedVector

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


# The following test-sets test for the correctness of the convolutions against
# AbstractVector and BoundaryPaddedArray in one dimension.

@testset "L.boundary_point_count is zero" begin
    N = 20
    L = CenteredDifference(2, 2, 1.0, N)
    cl = rand()
    cr = rand()
    u = rand(N)
    Qu = BoundaryPaddedVector(cl, cr, u)
    arrayQu = [cl;u;cr]

    # Check that BoundaryPaddedVector is constructed correctly
    @test Qu == arrayQu

    # Test for correctness of DerivativeOperator*AbstractVector
    @test second_derivative_stencil(N)*arrayQu ≈ L*arrayQu

    # Test for correctness of DerivativeOperator*BoundaryPaddedVector
    @test_broken second_derivative_stencil(N)*arrayQu ≈ L*Qu
end

@testset "Fourth Derivative and Approximation Order Test" begin
    N = 20
    L = CenteredDifference(4, 4, 1.0, N)
    cl = rand()
    cr = rand()
    u = rand(N)
    Qu = BoundaryPaddedVector(cl, cr, u)
    arrayQu = [cl;u;cr]

    # Check that BoundaryPaddedVector is constructed correctly
    @test Qu == arrayQu

    # Test for correctness of DerivativeOperator*AbstractVector
    @test fourth_deriv_approx_stencil(N)*arrayQu ≈ L*arrayQu

    # Test for correctness of DerivativeOperator*BoundaryPaddedVector
    @test any(abs.(fourth_deriv_approx_stencil(N)*arrayQu - L*Qu) .>= 1e-16)
end
