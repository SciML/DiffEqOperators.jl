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

@testset "2D Multiplication with no boundary points and dx = 1.0" begin

    # Test (Lxx + Lyy)*M, dx = 1.0, no coefficient
    N = 100
    M = zeros(N+2,N+2)
    M_temp = zeros(N,N)

    for i in 1:N+2
        for j in 1:N+2
            M[i,j] = cos(0.1i)+sin(0.1j)
        end
    end

    Lxx = CenteredDifference{1}(2,2,1.0,N)
    Lyy = CenteredDifference{2}(2,2,1.0,N)
    A = Lxx + Lyy

    mul!(M_temp, A, M)

    @test M_temp ≈ ((Lxx*M)[1:N,2:N+1] +(Lyy*M)[2:N+1,1:N])

    # Test a single axis, multiple operators: (Lx + Lxx)*M, dx = 1.0
    Lx = CenteredDifference{1}(1,2,1.0,N)
    A = Lx + Lxx

    M_temp = zeros(100,102)
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Lx*M)+(Lxx*M))

    # Test a single axis, multiple operators: (Ly + Lyy)*M, dx = 1.0, no coefficient
    Ly = CenteredDifference{2}(1,2,1.0,N)
    A = Ly + Lyy

    M_temp = zeros(102,100)
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Ly*M)+(Lyy*M))

    # Test multiple operators on both axis: (Lx + Ly + Lxx + Lyy)*M, no coefficient
    A = Lx + Ly + Lxx + Lyy
    M_temp = zeros(100,100)
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Lx*M)[1:N,2:N+1] +(Ly*M)[2:N+1,1:N] + (Lxx*M)[1:N,2:N+1] +(Lyy*M)[2:N+1,1:N])
end

@testset "2D Multiplication with identical bpc and dx = 1.0" begin

    # Test (Lxxxx + Lyyyy)*M, dx = 1.0, no coefficient, two boundary points on each axis
    N = 100
    M = zeros(N+2,N+2)
    M_temp = zeros(N,N)

    for i in 1:N+2
        for j in 1:N+2
            M[i,j] = cos(0.1i)+sin(0.1j)
        end
    end

    Lx4 = CenteredDifference{1}(4,4,1.0,N)
    Ly4 = CenteredDifference{2}(4,4,1.0,N)
    A = Lx4 + Ly4

    mul!(M_temp, A, M)

    @test M_temp ≈ ((Lx4*M)[1:N,2:N+1] +(Ly4*M)[2:N+1,1:N])

    # Test a single axis, multiple operators: (Lxxx + Lxxxx)*M, dx = 1.0
    Lx3 = CenteredDifference{1}(3,4,1.0,N)
    A = Lx3 + Lx4

    M_temp = zeros(100,102)
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Lx3*M)+(Lx4*M))

    # Test a single axis, multiple operators: (Lyyy + Lyyyy)*M, dx = 1.0, no coefficient
    Ly3 = CenteredDifference{2}(3,4,1.0,N)
    A = Ly3 + Ly4

    M_temp = zeros(102,100)
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Ly3*M)+(Ly4*M))

    # Test multiple operators on both axis: (Lxxx + Lyyy + Lxxxx + Lyyyy)*M, no coefficient
    A = Lx3 + Ly3 + Lx4 + Ly4
    M_temp = zeros(100,100)
    mul!(M_temp, A, M)

    # Test is broken due to mul! overload having a bug for non symmetric stencils
    @test_broken M_temp ≈ ((Lx3*M)[1:N,2:N+1] +(Ly3*M)[2:N+1,1:N] + (Lx4*M)[1:N,2:N+1] +(Ly4*M)[2:N+1,1:N])

    # Test (Lxxx + Lyyy)*M, no coefficient
    A = Lx3 + Ly3
    M_temp = zeros(100,100)
    mul!(M_temp, A, M)

    # Test is broken due to mul! overload having a bug for non symmetric stencils
    @test_broken M_temp ≈ ((Lx3*M)[1:N,2:N+1] +(Ly3*M)[2:N+1,1:N])


end
