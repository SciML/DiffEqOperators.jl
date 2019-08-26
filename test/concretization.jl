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
