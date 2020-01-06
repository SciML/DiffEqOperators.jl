using SparseArrays, DiffEqOperators, LinearAlgebra, Random,
      Test, BandedMatrices, FillArrays


# These functions create analytic solutions to upwind operators. In the function name,
# the first number indicates the derivative order, the second number indicates
# the approximation order,and the last three characters "Pos" or "Neg" indicating
# the winding direction. For instance analyticOneTwoPos corresponds to the positive
# upwind second order approximation of the first derivative.

function analyticOneOnePos()
      A = zeros(5,7)
      for i in 1:5
            A[i,i+1:i+2] = [-1 1]
      end
      return A
end

function analyticOneOneNeg()
      A = zeros(5,7)
      for i in 1:5
            A[i,i:i+1] = [-1 1]
      end
      return A
end

function analyticOneTwoPos()
      A = zeros(5,7)
      for i in 1:4
            A[i,i+1:i+3] = [-3/2 2 -1/2]
      end
      A[5,5:7] = [-1/2 0 1/2]
      return A
end

function analyticOneTwoNeg()
      A = zeros(5,7)
      A[1,1:3] = [-1/2 0 1/2]
      for i in 2:5
            A[i,i-1:i+1] = [1/2 -2 3/2]
      end
      return A
end

function analyticTwoTwoPos()
      A = zeros(5,7)
      for i in 1:3
            A[i,i+1:i+4] = [2 -5 4 -1]
      end
      A[4,4:7] = [1 -2 1 0]
      A[5, 4:7] = [0 1 -2 1]
      return A
end

function analyticTwoTwoNeg()
      A = zeros(5,7)
      A[1,1:4] = [1 -2 1 0]
      A[2,1:4] = [0 1 -2 1]
      for i in 3:5
            A[i,i-2:i+1] = [-1 4 -5 2]
      end
      return A
end

function analyticTwoThreePos()
      A = zeros(7,9)
      for i in 1:4
            A[i,i+1:i+5] = [35/12 -104/12 114/12 -56/12 11/12]
      end
      A[5,5:9] = [11/12 -20/12 6/12 4/12 -1/12]
      A[6,5:9] = [-1/12 16/12 -30/12 16/12 -1/12]
      A[7,5:9] = [-1/12 4/12 6/12 -20/12 11/12]
      return A
end

function analyticTwoThreeNeg()
      A = zeros(7,9)
      A[1,1:5] = [11/12 -20/12 6/12 4/12 -1/12]
      A[2,1:5] = [-1/12 16/12 -30/12 16/12 -1/12]
      A[3,1:5] = [-1/12 4/12 6/12 -20/12 11/12]
      for i in 4:7
            A[i,i-3:i+1] = [11/12 -56/12 114/12 -104/12 35/12]
      end
      return A
end

# analytically derived upwind operators for the irregular grid:
# [0.0, 0.08, 0.1, 0.15, 0.19, 0.26, 0.29], and spacing:
# dx = [0.08, 0.02, 0.05, 0.04, 0.07, 0.03]

function analyticOneOnePosIrr()
      A = zeros(5,7)
      A[1,2:3] = [-50, 50]
      A[2,3:4] = [-20, 20]
      A[3,4:5] = [-25, 25]
      A[4,5:6] = [-100/7, 100/7]
      A[5,6:7] = [-100/3, 100/3]
      return A
end

function analyticOneOneNegIrr()
      A = zeros(5,7)
      A[1,1:2] = [-25/2, 25/2]
      A[2,2:3] = [-50, 50]
      A[3,3:4] = [-20, 20]
      A[4,4:5] = [-25, 25]
      A[5,5:6] = [-100/7, 100/7]
      return A
end


@testset "Test: Derivative Order = 1, Approx Order = 1, Winding = Positive" begin
      N = 5
      L = UpwindDifference(1,1, 1.0, N, t->1.0)
      analyticL = analyticOneOnePos()
      x = rand(7)

      # Test that multiplication agrees with analytic multiplication
      @test L*x ≈ analyticL*x

      # Test that concretized multiplication agrees with analytic multiplication
      @test Array(L)*x ≈ analyticL*x

      # Test that matrix-free multiplication agrees with concretized multiplication
      @test L*x ≈ Array(L)*x

      # Test that concretized matrix agrees with analytic matrix
      @test Array(L) == analyticL

      # TODO: add tests for sparse and banded concretizations

end

@testset "Test: Derivative Order = 1, Approx Order = 1, Winding = Negative" begin
      N = 5
      L = UpwindDifference(1,1, 1.0, N, t->-1.0)
      analyticL = -1*analyticOneOneNeg()
      x = rand(7)

      # Test that multiplication agrees with analytic multiplication
      @test L*x ≈ analyticL*x

      # Test that concretized multiplication agrees with analytic multiplication
      @test Array(L)*x ≈ analyticL*x

      # Test that matrix-free multiplication agrees with concretized multiplication
      @test L*x ≈ Array(L)*x

      # Test that concretized matrix agrees with analytic matrix
      @test Array(L) == analyticL

      # TODO: add tests for sparse and banded concretizations

end

@testset "Test: Derivative Order = 1, Approx Order = 2, Winding = Positive" begin
      N = 5
      L = UpwindDifference(1,2, 1.0, N, t->1.0)
      analyticL = analyticOneTwoPos()
      x = rand(7)

      # Test that multiplication agrees with analytic multiplication
      @test L*x ≈ analyticL*x

      # Test that concretized multiplication agrees with analytic multiplication
      @test Array(L)*x ≈ analyticL*x

      # Test that matrix-free multiplication agrees with concretized multiplication
      @test L*x ≈ Array(L)*x

      # Test that concretized matrix agrees with analytic matrix
      @test Array(L) == analyticL

      # TODO: add tests for sparse and banded concretizations

end

@testset "Test: Derivative Order = 1, Approx Order = 2, Winding = Negative" begin
      N = 5
      L = UpwindDifference(1,2, 1.0, N, t->-1.0)
      analyticL = -1*analyticOneTwoNeg()
      x = rand(7)

      # Test that multiplication agrees with analytic multiplication
      @test L*x ≈ analyticL*x

      # Test that concretized multiplication agrees with analytic multiplication
      @test Array(L)*x ≈ analyticL*x

      # Test that matrix-free multiplication agrees with concretized multiplication
      @test L*x ≈ Array(L)*x

      # Test that concretized matrix agrees with analytic matrix
      @test Array(L) == analyticL

      # TODO: add tests for sparse and banded concretizations

end

@testset "Test: Derivative Order = 2, Approx Order = 2, Winding = Positive" begin
      N = 5
      L = UpwindDifference(2,2, 1.0, N, t->1.0)
      analyticL = analyticTwoTwoPos()
      x = rand(7)

      # Test that multiplication agrees with analytic multiplication
      @test L*x ≈ analyticL*x

      # Test that concretized multiplication agrees with analytic multiplication
      @test Array(L)*x ≈ analyticL*x

      # Test that matrix-free multiplication agrees with concretized multiplication
      @test L*x ≈ Array(L)*x

      # Test that concretized matrix agrees with analytic matrix
      @test Array(L) == analyticL

      # TODO: add tests for sparse and banded concretizations

end

@testset "Test: Derivative Order = 2, Approx Order = 2, Winding = Negative" begin
      N = 5
      L = UpwindDifference(2,2, 1.0, N, t->-1.0)
      analyticL = -1*analyticTwoTwoNeg()
      x = rand(7)

      # Test that multiplication agrees with analytic multiplication
      @test L*x ≈ analyticL*x

      # Test that concretized multiplication agrees with analytic multiplication
      @test Array(L)*x ≈ analyticL*x

      # Test that matrix-free multiplication agrees with concretized multiplication
      @test L*x ≈ Array(L)*x

      # Test that concretized matrix agrees with analytic matrix
      @test Array(L) == analyticL

      # TODO: add tests for sparse and banded concretizations

end

# Here the operators are too big for five grid points, so weird corner cases must be accounted for
# We should be able to assume that users will not have cases like this.
@testset "Test: Derivative Order = 2, Approx Order = 3, Winding = Positive" begin
      N = 7
      L = UpwindDifference(2,3, 1.0, N, t->1.0)
      analyticL = analyticTwoThreePos()
      x = rand(9)

      # Test that multiplication agrees with analytic multiplication
      @test L*x ≈ analyticL*x

      # Test that concretized multiplication agrees with analytic multiplication
      @test Array(L)*x ≈ analyticL*x

      # Test that matrix-free multiplication agrees with concretized multiplication
      @test L*x ≈ Array(L)*x

      # Test that concretized matrix agrees with analytic matrix
      @test Array(L) ≈ analyticL

      # TODO: add tests for sparse and banded concretizations

end

@testset "Test: Derivative Order = 2, Approx Order = 3, Winding = Negative" begin
      N = 7
      L = UpwindDifference(2,3, 1.0, N, t->-1.0)
      analyticL = -1*analyticTwoThreeNeg()
      x = rand(9)

      # Test that multiplication agrees with analytic multiplication
      @test L*x ≈ analyticL*x

      # Test that concretized multiplication agrees with analytic multiplication
      @test Array(L)*x ≈ analyticL*x

      # Test that matrix-free multiplication agrees with concretized multiplication
      @test L*x ≈ Array(L)*x

      # Test that concretized matrix agrees with analytic matrix
      @test Array(L) ≈ analyticL

      # TODO: add tests for sparse and banded concretizations

end

# TODO: tests for non-uniform grid

@testset "Test: Derivative Order = 1, Approx Order = 1, Winding = Positive, Grid = Irregular" begin
      N = 5
      # constructor throws an error at the moment
      L = UpwindDifference(1,1, [0.08, 0.02, 0.05, 0.04, 0.07, 0.03], N, t->1.0)
      analyticL = analyticOneOnePosIrr()
      x = rand(7)

      # Test that multiplication agrees with analytic multiplication
      @test_broken L*x ≈ analyticL*x

      # Test that concretized multiplication agrees with analytic multiplication
      @test Array(L)*x ≈ analyticL*x

      # Test that matrix-free multiplication agrees with concretized multiplication
      @test_broken L*x ≈ Array(L)*x

      # Test that concretized matrix agrees with analytic matrix
      @test Array(L) ≈ analyticL

      # TODO: add tests for sparse and banded concretizations
end

@testset "Test: Derivative Order = 1, Approx Order = 1, Winding = Negative, Grid = Irregular" begin
      N = 5
      # constructor throws an error at the moment
      L = UpwindDifference(1,1, [0.08, 0.02, 0.05, 0.04, 0.07, 0.03], N, t->-1.0)
      analyticL = -1*analyticOneOneNegIrr()
      x = rand(7)

      # Test that multiplication agrees with analytic multiplication
      @test_broken L*x ≈ analyticL*x

      # Test that concretized multiplication agrees with analytic multiplication
      @test Array(L)*x ≈ analyticL*x

      # Test that matrix-free multiplication agrees with concretized multiplication
      @test_broken L*x ≈ Array(L)*x

      # Test that concretized matrix agrees with analytic matrix
      @test Array(L) ≈ analyticL

      # TODO: add tests for sparse and banded concretizations
end
