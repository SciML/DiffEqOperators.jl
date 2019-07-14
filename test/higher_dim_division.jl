using SparseArrays, DiffEqOperators, LinearAlgebra, Random,
      Test, BandedMatrices, FillArrays, LazyArrays


# The following test checks that left division in the second dimension is consistent
# with applying left division at each slice in the second dimension.
@testset "Basic Second Dimension Division" begin

    # The second dimension is the last dimension of the array
    A = zeros(5,7)
    for i in 1:5
        for j in 1:7
            A[i,j] = sin(0.1i+j)
        end
    end
    L = CenteredDifference{2}(1,3, 1.0, 5)
    L1 = CenteredDifference{1}(1,3, 1.0, 5)
    B = L*A

    C = L\B
    for i in 1:5
        @test C[i,:] ≈ L1 \ B[i,:]
    end

    # The second dimension is the second last dimension of the array
    A = zeros(5,7,5)
    for i in 1:5
        for j in 1:7
            for k in 1:5
                A[i,j,k] = sin(0.1i+0.3k+j)
            end
        end
    end
    B = L*A

    C = L\B
    for i in 1:5
        for k in 1:5
            @test C[i,:,k] ≈ L1 \ B[i,:,k]
        end
    end

end



# The following test checks that left division in the third dimension is consistent
# with applying left division at each slice in the third dimension.
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
