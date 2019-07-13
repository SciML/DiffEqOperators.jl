using SparseArrays, DiffEqOperators, LinearAlgebra, Random,
      Test, BandedMatrices, FillArrays, LazyArrays


# The following test checks that left division in third dimensions is consistent
# with applying left division ot each slice in the third dimension.
@testset "Basic Third Dimension Division" begin


    A = zeros(2,2,7)
    for i in 1:2
        for j in 1:2
            for k in 1:7
                A[i,j,k] = (0.25i + j)*k^2
            end
        end
    end

    L = CenteredDifference{3}(1,3, 1.0, 5)
    L1 = CenteredDifference{1}(1,3, 1.0, 5)
    B = L*A

    C = L\B
    for i in 1:2
        for j in 1:2
            @test C[i,j,:] ≈ L1 \ B[i,j,:]
        end
    end


end
