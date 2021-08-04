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