using LinearAlgebra, DiffEqOperators, Random, Test, BandedMatrices, SparseArrays

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

function second_derivative_stencil(N)
  A = zeros(N,N+2)
  for i in 1:N, j in 1:N+2
      (j-i==0 || j-i==2) && (A[i,j]=1)
      j-i==1 && (A[i,j]=-2)
  end
  A
end

@testset "Differentiation on 2D array" begin
    M = zeros(22,22)
    M_temp = zeros(20,22)
    indices = Iterators.drop(CartesianIndices((22,22)), 0)
    for idx in indices
        M[idx] = sin(idx[1]*0.1)
    end
    L = CenteredDifference(4,4,0.1,20)

    #test mul!
    mul!(M_temp, L, M)

    correct_row = 10000.0*fourth_deriv_approx_stencil(20)*M[:,1]

    for i in 1:22
        @test M_temp[:,i] ≈ correct_row
    end

    # Test that * agrees will mul!
    @test M_temp == L*M

    # Differentiation along second dimension
    L = CenteredDifference{2}(4,4,0.1,20)
    M_temp_2 = zeros(22,20)
    indices = Iterators.drop(CartesianIndices((22,22)), 0)
    for idx in indices
        M[idx] = sin(idx[2]*0.1)
    end

    #test mul!
    mul!(M_temp_2, L, M)
    for i in 1:22
        @test M_temp_2[i,:] ≈ correct_row
    end

    # Test that * agrees will mul!
    @test M_temp_2 == L*M
end

@testset "Differenting on 3D array with L2" begin
    M = zeros(22,22,22)
    M_temp = zeros(20,22,22)
    indices = Iterators.drop(CartesianIndices((22,22,22)), 0)
    for idx in indices
        M[idx] = sin(idx[1]*0.1)
    end
    L = CenteredDifference(2,2,0.1,20)

    #test mul!
    mul!(M_temp, L, M)

    correct_row = 100.0*second_derivative_stencil(20)*M[:,1,1]

    for i in 1:22
        for j in 1:22
            @test M_temp[:,i,j] ≈ correct_row
        end
    end

    # Test that * agrees will mul!
    @test M_temp == L*M

    # Differentiation along second dimension
    L = CenteredDifference{2}(2,2,0.1,20)
    M_temp_2 = zeros(22,20,22)
    indices = Iterators.drop(CartesianIndices((22,22,22)), 0)
    for idx in indices
        M[idx] = sin(idx[2]*0.1)
    end

    #test mul!
    mul!(M_temp_2, L, M)
    for i in 1:22
        for j in 1:22
            @test M_temp_2[i,:,j] ≈ correct_row
        end
    end

    # Test that * agrees will mul!
    @test M_temp_2 == L*M

    # Differentiation along third dimension
    L = CenteredDifference{3}(2,2,0.1,20)
    M_temp_3 = zeros(22,22,20)
    indices = Iterators.drop(CartesianIndices((22,22,22)), 0)
    for idx in indices
        M[idx] = sin(idx[3]*0.1)
    end

    #test mul!
    mul!(M_temp_3, L, M)
    for i in 1:22
        for j in 1:22
            @test M_temp_3[i,j,:] ≈ correct_row
        end
    end

    # Test that * agrees will mul!
    @test M_temp_3 == L*M
end

@testset "Differentiation on 3D array with L4" begin
    M = zeros(22,22,22)
    M_temp = zeros(20,22,22)
    indices = Iterators.drop(CartesianIndices((22,22,22)), 0)
    for idx in indices
        M[idx] = sin(idx[1]*0.1)
    end
    L = CenteredDifference(4,4,0.1,20)

    #test mul!
    mul!(M_temp, L, M)

    correct_row = 10000.0*fourth_deriv_approx_stencil(20)*M[:,1,1]

    for i in 1:22
        for j in 1:22
            @test M_temp[:,i,j] ≈ correct_row
        end
    end

    # Test that * agrees will mul!
    @test M_temp == L*M

    # Differentiation along second dimension
    L = CenteredDifference{2}(4,4,0.1,20)
    M_temp_2 = zeros(22,20,22)
    for idx in indices
        M[idx] = sin(idx[2]*0.1)
    end

    #test mul!
    mul!(M_temp_2, L, M)
    for i in 1:22
        for j in 1:22
            @test M_temp_2[i,:,j] ≈ correct_row
        end
    end

    # Test that * agrees will mul!
    @test M_temp_2 == L*M

    # Differentiation along third dimension
    L = CenteredDifference{3}(4,4,0.1,20)
    M_temp_3 = zeros(22,22,20)
    for idx in indices
        M[idx] = sin(idx[3]*0.1)
    end

    #test mul!
    mul!(M_temp_3, L, M)
    for i in 1:22
        for j in 1:22
            @test M_temp_3[i,j,:] ≈ correct_row
        end
    end

    # Test that * agrees will mul!
    @test M_temp_3 == L*M
end

@testset "Differentiating an arbitrary higher dimension" begin
    N = 6
    L = CenteredDifference{N}(4,4,0.1,30)
    M = zeros(5,5,5,5,5,32,5);
    M_temp = zeros(5,5,5,5,5,30,5);
    indices = Iterators.drop(CartesianIndices((5,5,5,5,5,32,5)), 0);
    for idx in indices
        M[idx] = cos(idx[N]*0.1)
    end

    correct_row = (10.0^4)*fourth_deriv_approx_stencil(30)*M[1,1,1,1,1,:,1]

    #test mul!
    mul!(M_temp, L, M)
    indices = Iterators.drop(CartesianIndices((5,5,5,5,5,30,5)), 0);
    for idx in indices
        @test M_temp[idx] ≈ correct_row[idx[N]]
    end

    # Test that * agrees will mul!
    @test M_temp == L*M
end

@testset "Differentiating with non-symmetric interior stencil" begin

    # The following tests check that multiplication of an operator with a
    # non-symmetric interior stencil is consistent with what we expect

    N = 1
    L = CenteredDifference{N}(3,4,0.1,30)
    M = zeros(32,32)
    for i in 1:32
        for j in 1:32
            M[i,j] = cos(0.1i)
        end
    end

    M_temp = zeros(30,32)
    mul!(M_temp, L, M)

    @test M_temp ≈ Array(L)*M

    N = 2
    L = CenteredDifference{N}(3,4,0.1,30)
    M = zeros(32,32)
    for i in 1:32
        for j in 1:32
            M[i,j] = cos(0.1j)
        end
    end

    M_temp = zeros(32,30)
    mul!(M_temp, L, M)

    @test M_temp ≈ transpose(Array(L)*transpose(M))

    # Three dimensions

    N = 1
    L = CenteredDifference{N}(3,4,0.1,30)
    M = zeros(32,32,32)
    for i in 1:32
        for j in 1:32
            for k in 1:32
                M[i,j,k] = cos(0.1i)
            end
        end
    end

    M_temp = zeros(30,32,32)
    mul!(M_temp, L, M)
    for i in 1:32
        @test M_temp[:,:,i] ≈ Array(L)*M[:,:,i]
    end

    correct_row = L*M[:,1,1]

    N = 2
    L = CenteredDifference{N}(3,4,0.1,30)
    M = zeros(32,32,32)
    for i in 1:32
        for j in 1:32
            for k in 1:32
                M[i,j,k] = cos(0.1j)
            end
        end
    end

    M_temp = zeros(32,30,32)
    mul!(M_temp, L, M)

    for i in 1:32
        for j in 1:32
            @test M_temp[i,:,j] ≈ correct_row
        end
    end

    N = 3
    L = CenteredDifference{N}(3,4,0.1,30)
    M = zeros(32,32,32)
    for i in 1:32
        for j in 1:32
            for k in 1:32
                M[i,j,k] = cos(0.1k)
            end
        end
    end

    M_temp = zeros(32,32,30)
    mul!(M_temp, L, M)

    for i in 1:32
        for j in 1:32
            @test M_temp[i,j,:] ≈ correct_row
        end
    end
end

@testset "Differentiating with coefficients" begin

    # Three dimensions
    N = 1
    µ = 5.5
    L = CenteredDifference{N}(3,4,0.1,30)

    µL = µ*L
    M = zeros(32,32,32)
    for i in 1:32
        for j in 1:32
            for k in 1:32
                M[i,j,k] = cos(0.1i)
            end
        end
    end

    correct_slice = 5.5*(L*M[:,1,1])
    M_temp = zeros(30,32,32)
    mul!(M_temp, µL, M)

    for i in 1:32
        for j in 1:32
            @test M_temp[:,i,j] ≈ correct_slice
        end
    end

    N = 2
    µL = µ*CenteredDifference{N}(3,4,0.1,30)
    M = zeros(32,32,32)
    for i in 1:32
        for j in 1:32
            for k in 1:32
                M[i,j,k] = cos(0.1j)
            end
        end
    end

    M_temp = zeros(32,30,32)
    mul!(M_temp, µL, M)

    for i in 1:32
        for j in 1:32
            @test M_temp[i,:,j] ≈ correct_slice
        end
    end

    N = 3
    µL = µ*CenteredDifference{N}(3,4,0.1,30)
    M = zeros(32,32,32)
    for i in 1:32
        for j in 1:32
            for k in 1:32
                M[i,j,k] = cos(0.1k)
            end
        end
    end

    M_temp = zeros(32,32,30)
    mul!(M_temp, µL, M)

    for i in 1:32
        for j in 1:32
            @test M_temp[i,j,:] ≈ correct_slice
        end
    end
end
