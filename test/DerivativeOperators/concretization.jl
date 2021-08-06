using SparseArrays, DiffEqOperators, LinearAlgebra, Random,
      Test, BandedMatrices, FillArrays, LazyArrays, BlockBandedMatrices

# This test file tests for the correctness of higher dimensional concretization.
# The tests verify that multiplication in the concretized case agrees with the matrix-free
# multiplication

@testset "First Dimension" begin

      # Test that even when we have a vector, the concretizations using the higher dimension dispatch still function
      # correctly
      M = rand(22)

      L1 = CenteredDifference(1,2,1.0,20)
      L2 = CenteredDifference(1,2,1.0,20)
      L3 = CenteredDifference(3,3,1.0,20)

      @test L1*M ≈ Array(L1, size(M))*vec(M) ≈ sparse(L1,size(M))*M ≈ BandedMatrix(L1, size(M))*M ≈ BandedBlockBandedMatrix(L1,size(M))*M
      @test L2*M ≈ Array(L2, size(M))*vec(M) ≈ sparse(L2,size(M))*M ≈ BandedMatrix(L2, size(M))*M ≈ BandedBlockBandedMatrix(L2,size(M))*M
      @test L3*M ≈ Array(L3, size(M))*vec(M) ≈ sparse(L3,size(M))*M ≈ BandedMatrix(L3, size(M))*M ≈ BandedBlockBandedMatrix(L3,size(M))*M

      M = rand(22,2,2,2,2)

      @test vec(L1*M) ≈ Array(L1, size(M))*vec(M) ≈ sparse(L1,size(M))*vec(M) ≈ BandedMatrix(L1, size(M))*vec(M) ≈ BandedBlockBandedMatrix(L1,size(M))*vec(M)
      @test vec(L2*M) ≈ Array(L2, size(M))*vec(M) ≈ sparse(L2,size(M))*vec(M) ≈ BandedMatrix(L2, size(M))*vec(M) ≈ BandedBlockBandedMatrix(L2,size(M))*vec(M)
      @test vec(L3*M) ≈ Array(L3, size(M))*vec(M) ≈ sparse(L3,size(M))*vec(M) ≈ BandedMatrix(L3, size(M))*vec(M) ≈ BandedBlockBandedMatrix(L3,size(M))*vec(M)

end

@testset "Second Dimension" begin

      M = rand(2,22,2)

      L1 = CenteredDifference{2}(1,2,1.0,20)
      L2 = CenteredDifference{2}(1,2,1.0,20)
      L3 = CenteredDifference{2}(3,3,1.0,20)

      @test vec(L1*M) ≈ Array(L1, size(M))*vec(M) ≈ sparse(L1,size(M))*vec(M) ≈ BandedMatrix(L1, size(M))*vec(M) ≈ BandedBlockBandedMatrix(L1,size(M))*vec(M)
      @test vec(L2*M) ≈ Array(L2, size(M))*vec(M) ≈ sparse(L2,size(M))*vec(M) ≈ BandedMatrix(L2, size(M))*vec(M) ≈ BandedBlockBandedMatrix(L2,size(M))*vec(M)
      @test vec(L3*M) ≈ Array(L3, size(M))*vec(M) ≈ sparse(L3,size(M))*vec(M) ≈ BandedMatrix(L3, size(M))*vec(M) ≈ BandedBlockBandedMatrix(L3,size(M))*vec(M)

      M = rand(2,22,2,2)

      @test vec(L1*M) ≈ Array(L1, size(M))*vec(M) ≈ sparse(L1,size(M))*vec(M) ≈ BandedMatrix(L1, size(M))*vec(M) ≈ BandedBlockBandedMatrix(L1,size(M))*vec(M)
      @test vec(L2*M) ≈ Array(L2, size(M))*vec(M) ≈ sparse(L2,size(M))*vec(M) ≈ BandedMatrix(L2, size(M))*vec(M) ≈ BandedBlockBandedMatrix(L2,size(M))*vec(M)
      @test vec(L3*M) ≈ Array(L3, size(M))*vec(M) ≈ sparse(L3,size(M))*vec(M) ≈ BandedMatrix(L3, size(M))*vec(M) ≈ BandedBlockBandedMatrix(L3,size(M))*vec(M)

      M = rand(2,22,2,2,3)

      @test vec(L1*M) ≈ Array(L1, size(M))*vec(M) ≈ sparse(L1,size(M))*vec(M) ≈ BandedMatrix(L1, size(M))*vec(M) ≈ BandedBlockBandedMatrix(L1,size(M))*vec(M)
      @test vec(L2*M) ≈ Array(L2, size(M))*vec(M) ≈ sparse(L2,size(M))*vec(M) ≈ BandedMatrix(L2, size(M))*vec(M) ≈ BandedBlockBandedMatrix(L2,size(M))*vec(M)
      @test vec(L3*M) ≈ Array(L3, size(M))*vec(M) ≈ sparse(L3,size(M))*vec(M) ≈ BandedMatrix(L3, size(M))*vec(M) ≈ BandedBlockBandedMatrix(L3,size(M))*vec(M)

end

@testset "Third Dimension" begin

      M = rand(3,2,22)

      L1 = CenteredDifference{3}(1,2,1.0,20)
      L2 = CenteredDifference{3}(1,2,1.0,20)
      L3 = CenteredDifference{3}(3,3,1.0,20)


      @test vec(L1*M) ≈ Array(L1, size(M))*vec(M) ≈ sparse(L1,size(M))*vec(M) ≈ BandedMatrix(L1, size(M))*vec(M) ≈ BandedBlockBandedMatrix(L1,size(M))*vec(M)
      @test vec(L2*M) ≈ Array(L2, size(M))*vec(M) ≈ sparse(L2,size(M))*vec(M) ≈ BandedMatrix(L2, size(M))*vec(M) ≈ BandedBlockBandedMatrix(L2,size(M))*vec(M)
      @test vec(L3*M) ≈ Array(L3, size(M))*vec(M) ≈ sparse(L3,size(M))*vec(M) ≈ BandedMatrix(L3, size(M))*vec(M) ≈ BandedBlockBandedMatrix(L3,size(M))*vec(M)

      M = rand(3,2,22,2)

      @test vec(L1*M) ≈ Array(L1, size(M))*vec(M) ≈ sparse(L1,size(M))*vec(M) ≈ BandedMatrix(L1, size(M))*vec(M) ≈ BandedBlockBandedMatrix(L1,size(M))*vec(M)
      @test vec(L2*M) ≈ Array(L2, size(M))*vec(M) ≈ sparse(L2,size(M))*vec(M) ≈ BandedMatrix(L2, size(M))*vec(M) ≈ BandedBlockBandedMatrix(L2,size(M))*vec(M)
      @test vec(L3*M) ≈ Array(L3, size(M))*vec(M) ≈ sparse(L3,size(M))*vec(M) ≈ BandedMatrix(L3, size(M))*vec(M) ≈ BandedBlockBandedMatrix(L3,size(M))*vec(M)

      M = rand(3,2,22,2,3)

      @test vec(L1*M) ≈ Array(L1, size(M))*vec(M) ≈ sparse(L1,size(M))*vec(M) ≈ BandedMatrix(L1, size(M))*vec(M) ≈ BandedBlockBandedMatrix(L1,size(M))*vec(M)
      @test vec(L2*M) ≈ Array(L2, size(M))*vec(M) ≈ sparse(L2,size(M))*vec(M) ≈ BandedMatrix(L2, size(M))*vec(M) ≈ BandedBlockBandedMatrix(L2,size(M))*vec(M)
      @test vec(L3*M) ≈ Array(L3, size(M))*vec(M) ≈ sparse(L3,size(M))*vec(M) ≈ BandedMatrix(L3, size(M))*vec(M) ≈ BandedBlockBandedMatrix(L3,size(M))*vec(M)

end

@testset "Fifth Dimension" begin

      M = rand(3,2,3,2,22)

      L1 = CenteredDifference{5}(1,2,1.0,20)
      L2 = CenteredDifference{5}(1,2,1.0,20)
      L3 = CenteredDifference{5}(3,3,1.0,20)

      @test vec(L1*M) ≈ Array(L1, size(M))*vec(M) ≈ sparse(L1,size(M))*vec(M) ≈ BandedMatrix(L1, size(M))*vec(M) ≈ BandedBlockBandedMatrix(L1,size(M))*vec(M)
      @test vec(L2*M) ≈ Array(L2, size(M))*vec(M) ≈ sparse(L2,size(M))*vec(M) ≈ BandedMatrix(L2, size(M))*vec(M) ≈ BandedBlockBandedMatrix(L2,size(M))*vec(M)
      @test vec(L3*M) ≈ Array(L3, size(M))*vec(M) ≈ sparse(L3,size(M))*vec(M) ≈ BandedMatrix(L3, size(M))*vec(M) ≈ BandedBlockBandedMatrix(L3,size(M))*vec(M)

      M = rand(3,2,3,2,22,3)

      @test vec(L1*M) ≈ Array(L1, size(M))*vec(M) ≈ sparse(L1,size(M))*vec(M) ≈ BandedMatrix(L1, size(M))*vec(M) ≈ BandedBlockBandedMatrix(L1,size(M))*vec(M)
      @test vec(L2*M) ≈ Array(L2, size(M))*vec(M) ≈ sparse(L2,size(M))*vec(M) ≈ BandedMatrix(L2, size(M))*vec(M) ≈ BandedBlockBandedMatrix(L2,size(M))*vec(M)
      @test vec(L3*M) ≈ Array(L3, size(M))*vec(M) ≈ sparse(L3,size(M))*vec(M) ≈ BandedMatrix(L3, size(M))*vec(M) ≈ BandedBlockBandedMatrix(L3,size(M))*vec(M)

      M = rand(2,2,2,2,22,3,2)

      @test vec(L1*M) ≈ Array(L1, size(M))*vec(M) ≈ sparse(L1,size(M))*vec(M) ≈ BandedMatrix(L1, size(M))*vec(M) ≈ BandedBlockBandedMatrix(L1,size(M))*vec(M)
      @test vec(L2*M) ≈ Array(L2, size(M))*vec(M) ≈ sparse(L2,size(M))*vec(M) ≈ BandedMatrix(L2, size(M))*vec(M) ≈ BandedBlockBandedMatrix(L2,size(M))*vec(M)
      @test vec(L3*M) ≈ Array(L3, size(M))*vec(M) ≈ sparse(L3,size(M))*vec(M) ≈ BandedMatrix(L3, size(M))*vec(M) ≈ BandedBlockBandedMatrix(L3,size(M))*vec(M)

end

@testset "Biased Upwind : Positive Wind with offside, regular grid" begin

      # Test that Biased Upwind with maximum offside concretizes as CenteredDifference of same order
      M = rand(22)
      P = PeriodicBC(Float64)*M[2:21]

      L1 = CenteredDifference(1,2,1.0,20)
      L2 = CenteredDifference(1,4,1.0,20)
      L3 = CenteredDifference(1,6,1.0,20)
      L4 = CenteredDifference(2,3,1.0,20)
      L5 = CenteredDifference(3,4,1.0,20)

      K1 = UpwindDifference(1,2,1.0,20,offside=1)
      K2 = UpwindDifference(1,4,1.0,20,offside=2)
      K3 = UpwindDifference(1,6,1.0,20,offside=3)
      K4 = UpwindDifference(2,3,1.0,20,offside=2)
      K5 = UpwindDifference(3,4,1.0,20,offside=3)

      @test L1*M ≈ Array(L1, size(M))*vec(M) ≈ sparse(L1,size(M))*M ≈ BandedMatrix(L1, size(M))*M ≈ BandedBlockBandedMatrix(L1,size(M))*M ≈ K1*M ≈ Array(K1, size(M))*vec(M) ≈ sparse(K1,size(M))*M ≈ BandedMatrix(K1, size(M))*M ≈ BandedBlockBandedMatrix(K1,size(M))*M
      @test L2*M ≈ Array(L2, size(M))*vec(M) ≈ sparse(L2,size(M))*M ≈ BandedMatrix(L2, size(M))*M ≈ BandedBlockBandedMatrix(L2,size(M))*M ≈ K2*M ≈ Array(K2, size(M))*vec(M) ≈ sparse(K2,size(M))*M ≈ BandedMatrix(K2, size(M))*M ≈ BandedBlockBandedMatrix(K2,size(M))*M
      @test L2*P ≈ Array(L2, size(P))*vec(P) ≈ sparse(L2,size(P))*P ≈ BandedMatrix(L2, size(P))*P ≈ BandedBlockBandedMatrix(L2,size(P))*P ≈ K2*P ≈ Array(K2, size(P))*vec(P) ≈ sparse(K2,size(P))*P ≈ BandedMatrix(K2, size(P))*P ≈ BandedBlockBandedMatrix(K2,size(P))*P
      @test L3*M ≈ Array(L3, size(M))*vec(M) ≈ sparse(L3,size(M))*M ≈ BandedMatrix(L3, size(M))*M ≈ BandedBlockBandedMatrix(L3,size(M))*M ≈ K3*M ≈ Array(K3, size(M))*vec(M) ≈ sparse(K3,size(M))*M ≈ BandedMatrix(K3, size(M))*M ≈ BandedBlockBandedMatrix(K3,size(M))*M
      @test L4*M ≈ Array(L4, size(M))*vec(M) ≈ sparse(L4,size(M))*M ≈ BandedMatrix(L4, size(M))*M ≈ BandedBlockBandedMatrix(L4,size(M))*M ≈ K4*M ≈ Array(K4, size(M))*vec(M) ≈ sparse(K4,size(M))*M ≈ BandedMatrix(K4, size(M))*M ≈ BandedBlockBandedMatrix(K4,size(M))*M
      @test L5*M ≈ Array(L5, size(M))*vec(M) ≈ sparse(L5,size(M))*M ≈ BandedMatrix(L5, size(M))*M ≈ BandedBlockBandedMatrix(L5,size(M))*M ≈ K5*M ≈ Array(K5, size(M))*vec(M) ≈ sparse(K5,size(M))*M ≈ BandedMatrix(K5, size(M))*M ≈ BandedBlockBandedMatrix(K5,size(M))*M

      M = rand(22,2,2,2,2)

      @test vec(L1*M) ≈ Array(L1, size(M))*vec(M) ≈ sparse(L1,size(M))*vec(M) ≈ BandedMatrix(L1, size(M))*vec(M) ≈ BandedBlockBandedMatrix(L1,size(M))*vec(M) ≈ vec(K1*M) ≈ Array(K1, size(M))*vec(M) ≈ sparse(K1,size(M))*vec(M) ≈ BandedMatrix(K1, size(M))*vec(M) ≈ BandedBlockBandedMatrix(K1,size(M))*vec(M)
      @test vec(L2*M) ≈ Array(L2, size(M))*vec(M) ≈ sparse(L2,size(M))*vec(M) ≈ BandedMatrix(L2, size(M))*vec(M) ≈ BandedBlockBandedMatrix(L2,size(M))*vec(M) ≈ vec(K2*M) ≈ Array(K2, size(M))*vec(M) ≈ sparse(K2,size(M))*vec(M) ≈ BandedMatrix(K2, size(M))*vec(M) ≈ BandedBlockBandedMatrix(K2,size(M))*vec(M)
      @test vec(L3*M) ≈ Array(L3, size(M))*vec(M) ≈ sparse(L3,size(M))*vec(M) ≈ BandedMatrix(L3, size(M))*vec(M) ≈ BandedBlockBandedMatrix(L3,size(M))*vec(M) ≈ vec(K3*M) ≈ Array(K3, size(M))*vec(M) ≈ sparse(K3,size(M))*vec(M) ≈ BandedMatrix(K3, size(M))*vec(M) ≈ BandedBlockBandedMatrix(K3,size(M))*vec(M)
      @test vec(L4*M) ≈ Array(L4, size(M))*vec(M) ≈ sparse(L4,size(M))*vec(M) ≈ BandedMatrix(L4, size(M))*vec(M) ≈ BandedBlockBandedMatrix(L4,size(M))*vec(M) ≈ vec(K4*M) ≈ Array(K4, size(M))*vec(M) ≈ sparse(K4,size(M))*vec(M) ≈ BandedMatrix(K4, size(M))*vec(M) ≈ BandedBlockBandedMatrix(K4,size(M))*vec(M)
      @test vec(L5*M) ≈ Array(L5, size(M))*vec(M) ≈ sparse(L5,size(M))*vec(M) ≈ BandedMatrix(L5, size(M))*vec(M) ≈ BandedBlockBandedMatrix(L5,size(M))*vec(M) ≈ vec(K5*M) ≈ Array(K5, size(M))*vec(M) ≈ sparse(K5,size(M))*vec(M) ≈ BandedMatrix(K5, size(M))*vec(M) ≈ BandedBlockBandedMatrix(K5,size(M))*vec(M)

end

@testset "Biased Upwind : Negative Wind with offside, regular grid" begin

      M = rand(22)

      L1 = CenteredDifference(1,2,1.0,20,-1)
      L2 = CenteredDifference(1,4,1.0,20,-1)
      L3 = CenteredDifference(1,6,1.0,20,-1)
      L4 = CenteredDifference(2,3,1.0,20,-1)
      L5 = CenteredDifference(3,4,1.0,20,-1)

      K1 = UpwindDifference(1,2,1.0,20,-1,offside=1)
      K2 = UpwindDifference(1,4,1.0,20,-1,offside=2)
      K3 = UpwindDifference(1,6,1.0,20,-1,offside=3)
      K4 = UpwindDifference(2,3,1.0,20,-1,offside=2)
      K5 = UpwindDifference(3,4,1.0,20,-1,offside=3)

      @test L1*M ≈ Array(L1, size(M))*vec(M) ≈ sparse(L1,size(M))*M ≈ BandedMatrix(L1, size(M))*M ≈ BandedBlockBandedMatrix(L1,size(M))*M ≈ K1*M ≈ Array(K1, size(M))*vec(M) ≈ sparse(K1,size(M))*M ≈ BandedMatrix(K1, size(M))*M ≈ BandedBlockBandedMatrix(K1,size(M))*M
      @test L2*M ≈ Array(L2, size(M))*vec(M) ≈ sparse(L2,size(M))*M ≈ BandedMatrix(L2, size(M))*M ≈ BandedBlockBandedMatrix(L2,size(M))*M ≈ K2*M ≈ Array(K2, size(M))*vec(M) ≈ sparse(K2,size(M))*M ≈ BandedMatrix(K2, size(M))*M ≈ BandedBlockBandedMatrix(K2,size(M))*M
      @test L3*M ≈ Array(L3, size(M))*vec(M) ≈ sparse(L3,size(M))*M ≈ BandedMatrix(L3, size(M))*M ≈ BandedBlockBandedMatrix(L3,size(M))*M ≈ K3*M ≈ Array(K3, size(M))*vec(M) ≈ sparse(K3,size(M))*M ≈ BandedMatrix(K3, size(M))*M ≈ BandedBlockBandedMatrix(K3,size(M))*M
      @test L4*M ≈ Array(L4, size(M))*vec(M) ≈ sparse(L4,size(M))*M ≈ BandedMatrix(L4, size(M))*M ≈ BandedBlockBandedMatrix(L4,size(M))*M ≈ K4*M ≈ Array(K4, size(M))*vec(M) ≈ sparse(K4,size(M))*M ≈ BandedMatrix(K4, size(M))*M ≈ BandedBlockBandedMatrix(K4,size(M))*M
      @test L5*M ≈ Array(L5, size(M))*vec(M) ≈ sparse(L5,size(M))*M ≈ BandedMatrix(L5, size(M))*M ≈ BandedBlockBandedMatrix(L5,size(M))*M ≈ K5*M ≈ Array(K5, size(M))*vec(M) ≈ sparse(K5,size(M))*M ≈ BandedMatrix(K5, size(M))*M ≈ BandedBlockBandedMatrix(K5,size(M))*M

      M = rand(22,2,2,2,2)

      @test vec(L1*M) ≈ Array(L1, size(M))*vec(M) ≈ sparse(L1,size(M))*vec(M) ≈ BandedMatrix(L1, size(M))*vec(M) ≈ BandedBlockBandedMatrix(L1,size(M))*vec(M) ≈ vec(K1*M) ≈ Array(K1, size(M))*vec(M) ≈ sparse(K1,size(M))*vec(M) ≈ BandedMatrix(K1, size(M))*vec(M) ≈ BandedBlockBandedMatrix(K1,size(M))*vec(M)
      @test vec(L2*M) ≈ Array(L2, size(M))*vec(M) ≈ sparse(L2,size(M))*vec(M) ≈ BandedMatrix(L2, size(M))*vec(M) ≈ BandedBlockBandedMatrix(L2,size(M))*vec(M) ≈ vec(K2*M) ≈ Array(K2, size(M))*vec(M) ≈ sparse(K2,size(M))*vec(M) ≈ BandedMatrix(K2, size(M))*vec(M) ≈ BandedBlockBandedMatrix(K2,size(M))*vec(M)
      @test vec(L3*M) ≈ Array(L3, size(M))*vec(M) ≈ sparse(L3,size(M))*vec(M) ≈ BandedMatrix(L3, size(M))*vec(M) ≈ BandedBlockBandedMatrix(L3,size(M))*vec(M) ≈ vec(K3*M) ≈ Array(K3, size(M))*vec(M) ≈ sparse(K3,size(M))*vec(M) ≈ BandedMatrix(K3, size(M))*vec(M) ≈ BandedBlockBandedMatrix(K3,size(M))*vec(M)
      @test vec(L4*M) ≈ Array(L4, size(M))*vec(M) ≈ sparse(L4,size(M))*vec(M) ≈ BandedMatrix(L4, size(M))*vec(M) ≈ BandedBlockBandedMatrix(L4,size(M))*vec(M) ≈ vec(K4*M) ≈ Array(K4, size(M))*vec(M) ≈ sparse(K4,size(M))*vec(M) ≈ BandedMatrix(K4, size(M))*vec(M) ≈ BandedBlockBandedMatrix(K4,size(M))*vec(M)
      @test vec(L5*M) ≈ Array(L5, size(M))*vec(M) ≈ sparse(L5,size(M))*vec(M) ≈ BandedMatrix(L5, size(M))*vec(M) ≈ BandedBlockBandedMatrix(L5,size(M))*vec(M) ≈ vec(K5*M) ≈ Array(K5, size(M))*vec(M) ≈ sparse(K5,size(M))*vec(M) ≈ BandedMatrix(K5, size(M))*vec(M) ≈ BandedBlockBandedMatrix(K5,size(M))*vec(M)

end

@testset "Biased Upwind : Positive Wind with offside, irregular grid" begin

      M = rand(7)
      P = PeriodicBC(Float64)*M[2:6]

      L1 = CenteredDifference(1,2,[0.08, 0.02, 0.05, 0.04, 0.07, 0.03],5)
      L2 = CenteredDifference(1,4,[0.08, 0.02, 0.05, 0.04, 0.07, 0.03],5)
      L3 = CenteredDifference(2,3,[0.08, 0.02, 0.05, 0.04, 0.07, 0.03],5)

      K1 = UpwindDifference(1,2,[0.08, 0.02, 0.05, 0.04, 0.07, 0.03],5,offside=1)
      K2 = UpwindDifference(1,4,[0.08, 0.02, 0.05, 0.04, 0.07, 0.03],5,offside=2)
      K3 = UpwindDifference(2,3,[0.08, 0.02, 0.05, 0.04, 0.07, 0.03],5,offside=2)

      @test L1*M ≈ Array(L1, size(M))*vec(M) ≈ sparse(L1,size(M))*M ≈ BandedMatrix(L1, size(M))*M ≈ BandedBlockBandedMatrix(L1,size(M))*M ≈ K1*M ≈ Array(K1, size(M))*vec(M) ≈ sparse(K1,size(M))*M ≈ BandedMatrix(K1, size(M))*M ≈ BandedBlockBandedMatrix(K1,size(M))*M
      @test L2*M ≈ Array(L2, size(M))*vec(M) ≈ sparse(L2,size(M))*M ≈ BandedMatrix(L2, size(M))*M ≈ BandedBlockBandedMatrix(L2,size(M))*M ≈ K2*M ≈ Array(K2, size(M))*vec(M) ≈ sparse(K2,size(M))*M ≈ BandedMatrix(K2, size(M))*M ≈ BandedBlockBandedMatrix(K2,size(M))*M
      @test L2*P ≈ Array(L2, size(P))*vec(P) ≈ sparse(L2,size(P))*P ≈ BandedMatrix(L2, size(P))*P ≈ BandedBlockBandedMatrix(L2,size(P))*P ≈ K2*P ≈ Array(K2, size(P))*vec(P) ≈ sparse(K2,size(P))*P ≈ BandedMatrix(K2, size(P))*P ≈ BandedBlockBandedMatrix(K2,size(P))*P
      @test L3*M ≈ Array(L3, size(M))*vec(M) ≈ sparse(L3,size(M))*M ≈ BandedMatrix(L3, size(M))*M ≈ BandedBlockBandedMatrix(L3,size(M))*M ≈ K3*M ≈ Array(K3, size(M))*vec(M) ≈ sparse(K3,size(M))*M ≈ BandedMatrix(K3, size(M))*M ≈ BandedBlockBandedMatrix(K3,size(M))*M

      M = rand(7,2,2,2)

      @test vec(L1*M) ≈ Array(L1, size(M))*vec(M) ≈ sparse(L1,size(M))*vec(M) ≈ BandedMatrix(L1, size(M))*vec(M) ≈ BandedBlockBandedMatrix(L1,size(M))*vec(M) ≈ vec(K1*M) ≈ Array(K1, size(M))*vec(M) ≈ sparse(K1,size(M))*vec(M) ≈ BandedMatrix(K1, size(M))*vec(M) ≈ BandedBlockBandedMatrix(K1,size(M))*vec(M)
      @test vec(L2*M) ≈ Array(L2, size(M))*vec(M) ≈ sparse(L2,size(M))*vec(M) ≈ BandedMatrix(L2, size(M))*vec(M) ≈ BandedBlockBandedMatrix(L2,size(M))*vec(M) ≈ vec(K2*M) ≈ Array(K2, size(M))*vec(M) ≈ sparse(K2,size(M))*vec(M) ≈ BandedMatrix(K2, size(M))*vec(M) ≈ BandedBlockBandedMatrix(K2,size(M))*vec(M)
      @test vec(L3*M) ≈ Array(L3, size(M))*vec(M) ≈ sparse(L3,size(M))*vec(M) ≈ BandedMatrix(L3, size(M))*vec(M) ≈ BandedBlockBandedMatrix(L3,size(M))*vec(M) ≈ vec(K3*M) ≈ Array(K3, size(M))*vec(M) ≈ sparse(K3,size(M))*vec(M) ≈ BandedMatrix(K3, size(M))*vec(M) ≈ BandedBlockBandedMatrix(K3,size(M))*vec(M)

end

@testset "Biased Upwind : Negative Wind with offside, irregular grid" begin

      M = rand(7)

      L1 = CenteredDifference(1,2,[0.08, 0.02, 0.05, 0.04, 0.07, 0.03],5,-1)
      L2 = CenteredDifference(1,4,[0.08, 0.02, 0.05, 0.04, 0.07, 0.03],5,-1)
      L3 = CenteredDifference(2,3,[0.08, 0.02, 0.05, 0.04, 0.07, 0.03],5,-1)

      K1 = UpwindDifference(1,2,[0.08, 0.02, 0.05, 0.04, 0.07, 0.03],5,-1,offside=1)
      K2 = UpwindDifference(1,4,[0.08, 0.02, 0.05, 0.04, 0.07, 0.03],5,-1,offside=2)
      K3 = UpwindDifference(2,3,[0.08, 0.02, 0.05, 0.04, 0.07, 0.03],5,-1,offside=2)

      @test L1*M ≈ Array(L1, size(M))*vec(M) ≈ sparse(L1,size(M))*M ≈ BandedMatrix(L1, size(M))*M ≈ BandedBlockBandedMatrix(L1,size(M))*M ≈ K1*M ≈ Array(K1, size(M))*vec(M) ≈ sparse(K1,size(M))*M ≈ BandedMatrix(K1, size(M))*M ≈ BandedBlockBandedMatrix(K1,size(M))*M
      @test L2*M ≈ Array(L2, size(M))*vec(M) ≈ sparse(L2,size(M))*M ≈ BandedMatrix(L2, size(M))*M ≈ BandedBlockBandedMatrix(L2,size(M))*M ≈ K2*M ≈ Array(K2, size(M))*vec(M) ≈ sparse(K2,size(M))*M ≈ BandedMatrix(K2, size(M))*M ≈ BandedBlockBandedMatrix(K2,size(M))*M
      @test L3*M ≈ Array(L3, size(M))*vec(M) ≈ sparse(L3,size(M))*M ≈ BandedMatrix(L3, size(M))*M ≈ BandedBlockBandedMatrix(L3,size(M))*M ≈ K3*M ≈ Array(K3, size(M))*vec(M) ≈ sparse(K3,size(M))*M ≈ BandedMatrix(K3, size(M))*M ≈ BandedBlockBandedMatrix(K3,size(M))*M

      M = rand(7,2,2,2)

      @test vec(L1*M) ≈ Array(L1, size(M))*vec(M) ≈ sparse(L1,size(M))*vec(M) ≈ BandedMatrix(L1, size(M))*vec(M) ≈ BandedBlockBandedMatrix(L1,size(M))*vec(M) ≈ vec(K1*M) ≈ Array(K1, size(M))*vec(M) ≈ sparse(K1,size(M))*vec(M) ≈ BandedMatrix(K1, size(M))*vec(M) ≈ BandedBlockBandedMatrix(K1,size(M))*vec(M)
      @test vec(L2*M) ≈ Array(L2, size(M))*vec(M) ≈ sparse(L2,size(M))*vec(M) ≈ BandedMatrix(L2, size(M))*vec(M) ≈ BandedBlockBandedMatrix(L2,size(M))*vec(M) ≈ vec(K2*M) ≈ Array(K2, size(M))*vec(M) ≈ sparse(K2,size(M))*vec(M) ≈ BandedMatrix(K2, size(M))*vec(M) ≈ BandedBlockBandedMatrix(K2,size(M))*vec(M)
      @test vec(L3*M) ≈ Array(L3, size(M))*vec(M) ≈ sparse(L3,size(M))*vec(M) ≈ BandedMatrix(L3, size(M))*vec(M) ≈ BandedBlockBandedMatrix(L3,size(M))*vec(M) ≈ vec(K3*M) ≈ Array(K3, size(M))*vec(M) ≈ sparse(K3,size(M))*vec(M) ≈ BandedMatrix(K3, size(M))*vec(M) ≈ BandedBlockBandedMatrix(K3,size(M))*vec(M)

end

@testset "BC concretizations" begin

      M = rand(20)
      L1 = CenteredDifference(1,2,1.0,20)
      L2 = CenteredDifference(1,2,1.0,18)

      deriv_start, deriv_end = (L2*M)[1], (L2*M)[end]
      params = [1.0,0.5]
      left_RBC = params[1]*M[1] - params[2]*deriv_start
      right_RBC = params[1]*M[end] + params[2]*deriv_end
      bc1 = Dirichlet0BC(Float64)
      bc2 = DirichletBC(0.0,0.0)
      bc3 = NeumannBC((deriv_start,deriv_end),1.0,1)
      bc4 = RobinBC((params[1],-params[2],left_RBC), (params[1],params[2],right_RBC),1.0,1)
      bc5 = GeneralBC([-left_RBC,params[1],-params[2]],[-right_RBC,params[1],params[2]],1.0,1)
      bc6 = PeriodicBC(Float64)

      @test L1*bc1*M ≈ L1*(Array(bc1, length(M))[1]*vec(M) + Array(bc1, length(M))[2]) ≈ L1*(sparse(bc1, length(M))[1]*vec(M) + sparse(bc1, length(M))[2]) ≈ L1*(BandedMatrix(bc1, length(M))[1]*vec(M) + BandedMatrix(bc1, length(M))[2])
      @test L1*bc2*M ≈ L1*(Array(bc2, length(M))[1]*vec(M) + Array(bc2, length(M))[2]) ≈ L1*(sparse(bc2, length(M))[1]*vec(M) + sparse(bc2, length(M))[2]) ≈ L1*(BandedMatrix(bc2, length(M))[1]*vec(M) + BandedMatrix(bc2, length(M))[2])
      @test L1*bc3*M ≈ L1*(Array(bc3, length(M))[1]*vec(M) + Array(bc3, length(M))[2]) ≈ L1*(sparse(bc3, length(M))[1]*vec(M) + sparse(bc3, length(M))[2]) ≈ L1*(BandedMatrix(bc3, length(M))[1]*vec(M) + BandedMatrix(bc3, length(M))[2])
      @test L1*bc4*M ≈ L1*(Array(bc4, length(M))[1]*vec(M) + Array(bc4, length(M))[2]) ≈ L1*(sparse(bc4, length(M))[1]*vec(M) + sparse(bc4, length(M))[2]) ≈ L1*(BandedMatrix(bc4, length(M))[1]*vec(M) + BandedMatrix(bc4, length(M))[2])
      @test L1*bc5*M ≈ L1*(Array(bc5, length(M))[1]*vec(M) + Array(bc5, length(M))[2]) ≈ L1*(sparse(bc5, length(M))[1]*vec(M) + sparse(bc5, length(M))[2]) ≈ L1*(BandedMatrix(bc5, length(M))[1]*vec(M) + BandedMatrix(bc5, length(M))[2])
      @test L1*bc6*M ≈ L1*(Array(bc6, length(M))[1]*vec(M) + Array(bc6, length(M))[2]) ≈ L1*(sparse(bc6, length(M))[1]*vec(M) + sparse(bc6, length(M))[2]) ≈ L1*(BandedMatrix(bc6, length(M))[1]*vec(M) + BandedMatrix(bc6, length(M))[2])
end