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
    A[1,1:2] = [-25. / 2., 25. / 2.]
    A[2,2:3] = [-50., 50.]
    A[3,3:4] = [-20., 20.]
    A[4,4:5] = [-25., 25.]
    A[5,5:6] = [-100/7, 100/7]
    return A
end

function analyticOneTwoPosIrr()
    A = zeros(5,7)
    A[1,2:4] = [-450. / 7., 70., -40. / 7.]
    A[2,3:5] = [-280. / 9., 45., -125. / 9.]
    A[3,4:6] = [-2625. / 77., 3025. / 77., -400. / 77.]
    A[4,5:7] = [-510. / 21., 1000. / 21., -490. / 21.]
    A[5,5:7] = [-90. / 21., -400. / 21., 490. / 21.]
    return A
end

function analyticOneTwoNegIrr()
    A = zeros(5,7)
    A[1,1:3] = [-5. / 2., -75. / 2., 40.]
    A[2,1:3] = [5. / 2., -125. / 2., 60.]
    A[3,2:4] = [250. / 7., -70., 240. / 7.]
    A[4,3:5] = [80. / 9., -45., 325. / 9.]
    A[5,4:6] = [1225. / 77., -39. - 2. / 7., 23. + 29. / 77.]
    return A
end

function analyticTwoTwoPosIrr()
    A = zeros(5,7)
    A[1,2:5] = [2597. + 31. / 77., -4000., 1857. + 1. / 7., -454. - 42. / 77.]
    A[2,3:6] = [833. + 11. / 33., -2272. - 24. / 33., 1666. + 2. / 3., -227. - 3. / 11.]
    A[3,4:7] = [941. + 43. / 77., -1785 - 5. / 7., 1558. + 34. / 77., -714. - 2. / 7.]
    A[4,4:7] = [551. + 73. / 77., -928. - 4. / 7., 519. + 37. / 77., -142. - 6. / 7.]
    A[5,4:7] = [-129. - 67. / 77., 571. + 3. / 7., -1298. - 54. / 77., 857. + 1. / 7.]
    return A
end

function analyticTwoTwoNegIrr()
    A = zeros(5,7)
    A[1,1:4] = [150., -178. - 4. / 7., -200., 228. + 4. / 7.]
    A[2,1:4] = [50., 892. + 6. / 7., -1400., 457. + 1. / 7.]
    A[3,2:5] = [ -200., 3571. + 3. / 7., -4400., 1028. + 4. / 7.]
    A[4,3:6] = [-1688 - 72. / 231., 3333. + 77. / 231.,
                -2857. - 33 / 231., 1212. + 28. / 231.]
    A[5,4:7] = [-500., 2090. + 10. / 11.,
                -2142. - 6. / 7., 551. + 73. / 77.]
    return A
end

function analyticTwoThreePosIrr()
    A = zeros(5,7)
    A[1,2:6] = [3412. + 484. / 693., -5569. - 308. / 693.,
                3324. + 468. / 693., -1269. - 583. / 693.,
                101. + 632. / 693.]
    A[2,3:7] = [1226. + 8008. / 13167., -4019. - 6327. / 13167.,
                3801. + 7733. / 13167., -1682. - 11856. / 13167.,
                674. + 2442. / 13167.]
    A[3,3:7] = [371. + 4543. / 13167., -707. - 10431. / 13167.,
                230. + 2090. / 13167., 183. + 12939. / 13167.,
                -77. - 9141. / 13167.]
    A[4,3:7] = [2. + 12166. / 13167., 538. + 12654. / 13167.,
                -912. - 9196. / 13167., 508. + 8664. / 13167.,
                -137. - 11121. / 13167.]
    A[5,3:7] = [33. + 8239. / 13167., -279. - 2907. / 13167.,
                753. + 12749. / 13167., -1423. - 2109. / 13167.,
                914. + 10362. / 13167.]
    return A
end

function analyticTwoThreeNegIrr()
    A = zeros(5,7)
    A[1,1:5] = [99. + 1617. / 13167., 762. + 12996. / 13167.,
                -1488. - 11704. / 13167., 780. + 12540. / 13167.,
                -154. - 2282. / 13167.]
    A[2,1:5] = [14. + 12012. / 13167., 1542. + 2736. / 13167.,
                -2288. - 11704. / 13167., 838. + 1254. / 13167.,
                -106. - 4298. / 13167.]
    A[3,1:5] = [-11. - 5313. / 13167., 81. + 2223. / 13167.,
                377. + 10241. / 13167., -1019. - 627. / 13167.,
                571. + 6643. / 13167.]
    A[4,1:5] = [157. + 231. / 13167., -4594. - 2052. / 13167.,
                7311. + 1463. / 13167., -4561. - 11913. / 13167.,
                1687. + 12271. / 13167.]
    A[5,2:6] = [2633. + 331. / 693., -5569. - 308. / 693.,
                6831. + 117. / 693., -4776. - 232. / 693.,
                881. + 92. / 693.]
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
      @test_broken L = UpwindDifference(1,1, [0.08, 0.02, 0.05, 0.04, 0.07, 0.03], N, t->1.0)
      analyticL = analyticOneOnePosIrr()
      x = rand(5)

      # Test that multiplication agrees with analytic multiplication
      @test_broken L*x ≈ analyticL*x

      # Test that concretized multiplication agrees with analytic multiplication
      @test_broken Array(L)*x ≈ analyticL*x

      # Test that matrix-free multiplication agrees with concretized multiplication
      @test_broken L*x ≈ Array(L)*x

      # Test that concretized matrix agrees with analytic matrix
      @test_broken Array(L) ≈ analyticL

      # TODO: add tests for sparse and banded concretizations
end

@testset "Test: Derivative Order = 1, Approx Order = 1, Winding = Negative, Grid = Irregular" begin
      N = 5
      # constructor throws an error at the moment
      @test_broken L = UpwindDifference(1,1, [0.08, 0.02, 0.05, 0.04, 0.07, 0.03], N, t->-1.0)
      analyticL = analyticOneOnePosIrr()
      x = rand(5)

      # Test that multiplication agrees with analytic multiplication
      @test_broken L*x ≈ analyticL*x

      # Test that concretized multiplication agrees with analytic multiplication
      @test_broken Array(L)*x ≈ analyticL*x

      # Test that matrix-free multiplication agrees with concretized multiplication
      @test_broken L*x ≈ Array(L)*x

      # Test that concretized matrix agrees with analytic matrix
      @test_broken Array(L) ≈ analyticL

      # TODO: add tests for sparse and banded concretizations
end
