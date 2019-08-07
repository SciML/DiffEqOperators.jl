using LinearAlgebra, DiffEqOperators, Random, Test, BandedMatrices, SparseArrays

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

    @test M_temp ≈ ((Lx3*M)[1:N,2:N+1] +(Ly3*M)[2:N+1,1:N] + (Lx4*M)[1:N,2:N+1] +(Ly4*M)[2:N+1,1:N])

    # Test (Lxxx + Lyyy)*M, no coefficient. These operators have non-symmetric interior stencils
    A = Lx3 + Ly3
    M_temp = zeros(100,100)
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Lx3*M)[1:N,2:N+1] +(Ly3*M)[2:N+1,1:N])

end

@testset "2D Multiplication with differing bpc and dx = 1.0" begin

    N = 100
    M = zeros(N+2,N+2)
    M_temp = zeros(N,N+2)
    for i in 1:N+2
        for j in 1:N+2
            M[i,j] = cos(0.1i)+sin(0.1j)
        end
    end

    # Lx2 has 0 boundary points
    Lx2 = CenteredDifference{1}(2,2,1.0,N)
    # Lx3 has 1 boundary point
    Lx3 = CenteredDifference{1}(3,3,1.0,N)
    # Lx4 has 2 boundary points
    Lx4 = CenteredDifference{1}(4,4,1.0,N)

    # Test a single axis, multiple operators: (Lxx+Lxxxx)*M, dx = 1.0
    A = Lx2+Lx4
    mul!(M_temp, A, M)
    @test M_temp ≈ ((Lx2*M) + (Lx4*M))

    # Test a single axis, multiple operators: (Lxx++Lxxx+Lxxxx)*M, dx = 1.0
    A += Lx3
    mul!(M_temp, A, M)
    @test M_temp ≈ ((Lx2*M) + (Lx3*M) + (Lx4*M))


    # Ly2 has 0 boundary points
    Ly2 = CenteredDifference{2}(2,2,1.0,N)
    # Ly3 has 1 boundary point
    Ly3 = CenteredDifference{2}(3,3,1.0,N)
    # Ly4 has 2 boundary points
    Ly4 = CenteredDifference{2}(4,4,1.0,N)
    M_temp = zeros(N+2,N)

    # Test a single axis, multiple operators: (Lyy+Lyyyy)*M, dx = 1.0
    A = Ly2+Ly4
    mul!(M_temp, A, M)
    @test M_temp ≈ ((Ly2*M) + (Ly4*M))

    # Test a single axis, multiple operators: (Lyy++Lyyy+Lyyyy)*M, dx = 1.0
    A += Ly3
    mul!(M_temp, A, M)
    @test M_temp ≈ ((Ly2*M) + (Ly3*M) + (Ly4*M))


    # Test multiple operators on both axis: (Lxx + Lyy + Lxxx + Lyyy + Lxxxx + Lyyyy)*M, no coefficient
    A = Lx2 + Ly2 + Lx3 + Ly3 + Lx4 + Ly4
    M_temp = zeros(100,100)
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Lx2*M)[1:N,2:N+1]+(Ly2*M)[2:N+1,1:N]+(Lx3*M)[1:N,2:N+1] +(Ly3*M)[2:N+1,1:N] + (Lx4*M)[1:N,2:N+1] +(Ly4*M)[2:N+1,1:N])

end

@testset "2D Multiplication with identical bpc and non-trivial dx = dy = 0.1" begin

    # Test (Lxxxx + Lyyyy)*M, dx = 0.1, dy = 0.01, no coefficient, two boundary points on each axis
    N = 100
    M = zeros(N+2,N+2)
    M_temp = zeros(N,N)

    for i in 1:N+2
        for j in 1:N+2
            M[i,j] = cos(0.1i)+sin(0.1j)
        end
    end

    Lx4 = CenteredDifference{1}(4,4,0.1,N)
    Ly4 = CenteredDifference{2}(4,4,0.1,N)
    A = Lx4 + Ly4

    mul!(M_temp, A, M)

    @test M_temp ≈ ((Lx4*M)[1:N,2:N+1] +(Ly4*M)[2:N+1,1:N])

    # Test a single axis, multiple operators: (Lxxx + Lxxxx)*M, dx = 0.1
    Lx3 = CenteredDifference{1}(3,4,0.1,N)
    A = Lx3 + Lx4

    M_temp = zeros(100,102)
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Lx3*M)+(Lx4*M))

    # Test a single axis, multiple operators: (Lyyy + Lyyyy)*M, dx = 0.01, no coefficient
    Ly3 = CenteredDifference{2}(3,4,0.1,N)
    A = Ly3 + Ly4

    M_temp = zeros(102,100)
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Ly3*M)+(Ly4*M))

    # Test multiple operators on both axis: (Lxxx + Lyyy + Lxxxx + Lyyyy)*M, no coefficient dx =
    A = Lx3 + Ly3 + Lx4 + Ly4
    M_temp = zeros(100,100)
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Lx3*M)[1:N,2:N+1] +(Ly3*M)[2:N+1,1:N] + (Lx4*M)[1:N,2:N+1] +(Ly4*M)[2:N+1,1:N])

    # Test (Lxxx + Lyyy)*M, no coefficient. These operators have non-symmetric interior stencils
    A = Lx3 + Ly3
    M_temp = zeros(100,100)
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Lx3*M)[1:N,2:N+1] +(Ly3*M)[2:N+1,1:N])

end

@testset "2D Multiplication with identical bpc and non-trivial dx = 0.1, dy = 0.25" begin

    # Test (Lxxxx + Lyyyy)*M, dx = 0.1, dy = 0.01, no coefficient, two boundary points on each axis
    dx = 0.1
    dy = 0.25
    N = 100
    M = zeros(N+2,N+2)
    M_temp = zeros(N,N)

    for i in 1:N+2
        for j in 1:N+2
            M[i,j] = cos(dx*i)+sin(dy*j)
        end
    end

    Lx4 = CenteredDifference{1}(4,4,dx,N)
    Ly4 = CenteredDifference{2}(4,4,dy,N)
    A = Lx4 + Ly4

    mul!(M_temp, A, M)

    @test M_temp ≈ ((Lx4*M)[1:N,2:N+1] +(Ly4*M)[2:N+1,1:N])

    # Test a single axis, multiple operators: (Lxxx + Lxxxx)*M, dx = 0.1
    Lx3 = CenteredDifference{1}(3,4,dx,N)
    A = Lx3 + Lx4

    M_temp = zeros(100,102)
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Lx3*M)+(Lx4*M))

    # Test a single axis, multiple operators: (Lyyy + Lyyyy)*M, dx = 0.01, no coefficient
    Ly3 = CenteredDifference{2}(3,4,dy,N)
    A = Ly3 + Ly4

    M_temp = zeros(102,100)
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Ly3*M)+(Ly4*M))

    # Test multiple operators on both axis: (Lxxx + Lyyy + Lxxxx + Lyyyy)*M, no coefficient dx =
    A = Lx3 + Ly3 + Lx4 + Ly4
    M_temp = zeros(100,100)
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Lx3*M)[1:N,2:N+1] +(Ly3*M)[2:N+1,1:N] + (Lx4*M)[1:N,2:N+1] +(Ly4*M)[2:N+1,1:N])

    # Test (Lxxx + Lyyy)*M, no coefficient. These operators have non-symmetric interior stencils
    A = Lx3 + Ly3
    M_temp = zeros(100,100)
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Lx3*M)[1:N,2:N+1] +(Ly3*M)[2:N+1,1:N])

end

@testset "2D Multiplication with differing bpc and non-trivial dx = dy = 0.1" begin

    dx = 0.1
    dy = 0.1
    N = 100
    M = zeros(N+2,N+2)
    M_temp = zeros(N,N+2)
    for i in 1:N+2
        for j in 1:N+2
            M[i,j] = cos(dx*i)+sin(dy*j)
        end
    end

    # Lx2 has 0 boundary points
    Lx2 = CenteredDifference{1}(2,2,dx,N)
    # Lx3 has 1 boundary point
    Lx3 = CenteredDifference{1}(3,3,dx,N)
    # Lx4 has 2 boundary points
    Lx4 = CenteredDifference{1}(4,4,dx,N)

    # Test a single axis, multiple operators: (Lxx+Lxxxx)*M, dx = 1.0
    A = Lx2+Lx4
    mul!(M_temp, A, M)
    @test M_temp ≈ ((Lx2*M) + (Lx4*M))

    # Test a single axis, multiple operators: (Lxx++Lxxx+Lxxxx)*M, dx = 1.0
    A += Lx3
    mul!(M_temp, A, M)
    @test M_temp ≈ ((Lx2*M) + (Lx3*M) + (Lx4*M))


    # Ly2 has 0 boundary points
    Ly2 = CenteredDifference{2}(2,2,dy,N)
    # Ly3 has 1 boundary point
    Ly3 = CenteredDifference{2}(3,3,dy,N)
    # Ly4 has 2 boundary points
    Ly4 = CenteredDifference{2}(4,4,dy,N)
    M_temp = zeros(N+2,N)

    # Test a single axis, multiple operators: (Lyy+Lyyyy)*M, dx = 1.0
    A = Ly2+Ly4
    mul!(M_temp, A, M)
    @test M_temp ≈ ((Ly2*M) + (Ly4*M))

    # Test a single axis, multiple operators: (Lyy++Lyyy+Lyyyy)*M, dx = 1.0
    A += Ly3
    mul!(M_temp, A, M)
    @test M_temp ≈ ((Ly2*M) + (Ly3*M) + (Ly4*M))


    # Test multiple operators on both axis: (Lxx + Lyy + Lxxx + Lyyy + Lxxxx + Lyyyy)*M, no coefficient
    A = Lx2 + Ly2 + Lx3 + Ly3 + Lx4 + Ly4
    M_temp = zeros(100,100)
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Lx2*M)[1:N,2:N+1]+(Ly2*M)[2:N+1,1:N]+(Lx3*M)[1:N,2:N+1] +(Ly3*M)[2:N+1,1:N] + (Lx4*M)[1:N,2:N+1] +(Ly4*M)[2:N+1,1:N])

end

@testset "2D Multiplication with differing bpc and non-trivial dx = 0.1, dy = 0.25" begin

    dx = 0.1
    dy = 0.25
    N = 100
    M = zeros(N+2,N+2)
    M_temp = zeros(N,N+2)
    for i in 1:N+2
        for j in 1:N+2
            M[i,j] = cos(dx*i)+sin(dy*j)
        end
    end

    # Lx2 has 0 boundary points
    Lx2 = CenteredDifference{1}(2,2,dx,N)
    # Lx3 has 1 boundary point
    Lx3 = CenteredDifference{1}(3,3,dx,N)
    # Lx4 has 2 boundary points
    Lx4 = CenteredDifference{1}(4,4,dx,N)

    # Test a single axis, multiple operators: (Lxx+Lxxxx)*M, dx = 1.0
    A = Lx2+Lx4
    mul!(M_temp, A, M)
    @test M_temp ≈ ((Lx2*M) + (Lx4*M))

    # Test a single axis, multiple operators: (Lxx++Lxxx+Lxxxx)*M, dx = 1.0
    A += Lx3
    mul!(M_temp, A, M)
    @test M_temp ≈ ((Lx2*M) + (Lx3*M) + (Lx4*M))


    # Ly2 has 0 boundary points
    Ly2 = CenteredDifference{2}(2,2,dy,N)
    # Ly3 has 1 boundary point
    Ly3 = CenteredDifference{2}(3,3,dy,N)
    # Ly4 has 2 boundary points
    Ly4 = CenteredDifference{2}(4,4,dy,N)
    M_temp = zeros(N+2,N)

    # Test a single axis, multiple operators: (Lyy+Lyyyy)*M, dx = 1.0
    A = Ly2+Ly4
    mul!(M_temp, A, M)
    @test M_temp ≈ ((Ly2*M) + (Ly4*M))

    # Test a single axis, multiple operators: (Lyy++Lyyy+Lyyyy)*M, dx = 1.0
    A += Ly3
    mul!(M_temp, A, M)
    @test M_temp ≈ ((Ly2*M) + (Ly3*M) + (Ly4*M))


    # Test multiple operators on both axis: (Lxx + Lyy + Lxxx + Lyyy + Lxxxx + Lyyyy)*M, no coefficient
    A = Lx2 + Ly2 + Lx3 + Ly3 + Lx4 + Ly4
    M_temp = zeros(100,100)
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Lx2*M)[1:N,2:N+1]+(Ly2*M)[2:N+1,1:N]+(Lx3*M)[1:N,2:N+1] +(Ly3*M)[2:N+1,1:N] + (Lx4*M)[1:N,2:N+1] +(Ly4*M)[2:N+1,1:N])

end

# THis testset uses the last testset which has a several non-trivial cases,
# and additionally tests coefficient handling. All operators are handled by the
# fast 2D/3D dispatch.
@testset "2D coefficient handling" begin

    dx = 0.1
    dy = 0.25
    N = 100
    M = zeros(N+2,N+2)
    M_temp = zeros(N,N+2)
    for i in 1:N+2
        for j in 1:N+2
            M[i,j] = cos(dx*i)+sin(dy*j)
        end
    end

    # Lx2 has 0 boundary points
    Lx2 = 5.5*CenteredDifference{1}(2,2,dx,N)
    # Lx3 has 1 boundary point
    Lx3 = 1.45*CenteredDifference{1}(3,3,dx,N)
    # Lx4 has 2 boundary points
    Lx4 = 0.5*CenteredDifference{1}(4,4,dx,N)

    # Test a single axis, multiple operators: (Lxx+Lxxxx)*M, dx = 1.0
    A = Lx2+Lx4
    mul!(M_temp, A, M)
    @test M_temp ≈ ((Lx2*M) + (Lx4*M))

    # Test a single axis, multiple operators: (Lxx++Lxxx+Lxxxx)*M, dx = 1.0
    A += Lx3
    mul!(M_temp, A, M)
    @test M_temp ≈ ((Lx2*M) + (Lx3*M) + (Lx4*M))


    # Ly2 has 0 boundary points
    Ly2 = 8.14*CenteredDifference{2}(2,2,dy,N)
    # Ly3 has 1 boundary point
    Ly3 = 2.0*CenteredDifference{2}(3,3,dy,N)
    # Ly4 has 2 boundary points
    Ly4 = 4.567*CenteredDifference{2}(4,4,dy,N)
    M_temp = zeros(N+2,N)

    # Test a single axis, multiple operators: (Lyy+Lyyyy)*M, dx = 1.0
    A = Ly2+Ly4
    mul!(M_temp, A, M)
    @test M_temp ≈ ((Ly2*M) + (Ly4*M))

    # Test a single axis, multiple operators: (Lyy++Lyyy+Lyyyy)*M, dx = 1.0
    A += Ly3
    mul!(M_temp, A, M)
    @test M_temp ≈ ((Ly2*M) + (Ly3*M) + (Ly4*M))


    # Test multiple operators on both axis: (Lxx + Lyy + Lxxx + Lyyy + Lxxxx + Lyyyy)*M, no coefficient
    A = Lx2 + Ly2 + Lx3 + Ly3 + Lx4 + Ly4
    M_temp = zeros(100,100)
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Lx2*M)[1:N,2:N+1]+(Ly2*M)[2:N+1,1:N]+(Lx3*M)[1:N,2:N+1] +(Ly3*M)[2:N+1,1:N] + (Lx4*M)[1:N,2:N+1] +(Ly4*M)[2:N+1,1:N])
end

@testset "x and y are both irregular grids" begin

    N = 100
    dx = cumsum(rand(N+2))
    dy = cumsum(rand(N+2))
    M = zeros(N+2,N+2)

    for i in 1:N+2
        for j in 1:N+2
            M[i,j] = cos(dx[i])+sin(dy[j])
        end
    end

    # Lx2 has 0 boundary points
    Lx2 = CenteredDifference{1}(2,2,dx,N)
    # Lx3 has 1 boundary point
    Lx3 = 1.45*CenteredDifference{1}(3,3,dx,N)
    # Lx4 has 2 boundary points
    Lx4 = CenteredDifference{1}(4,4,dx,N)

    # Ly2 has 0 boundary points
    Ly2 = 8.14*CenteredDifference{2}(2,2,dy,N)
    # Ly3 has 1 boundary point
    Ly3 = CenteredDifference{2}(3,3,dy,N)
    # Ly4 has 2 boundary points
    Ly4 = 4.567*CenteredDifference{2}(4,4,dy,N)

    # Test composition of all first-dimension operators
    A = Lx2+Lx3+Lx4
    M_temp = zeros(N,N+2)
    mul!(M_temp, A, M)
    @test M_temp ≈ (Lx2*M + Lx3*M + Lx4*M)

    # Test composition of all second-dimension operators
    A = Ly2+Ly3+Ly4
    M_temp = zeros(N+2,N)
    mul!(M_temp, A, M)
    @test M_temp ≈ (Ly2*M + Ly3*M + Ly4*M)

    # Test composition of all operators
    A = Lx2+Lx3+Lx4+Ly2+Ly3+Ly4
    M_temp = zeros(N,N)
    mul!(M_temp, A, M)
    @test M_temp ≈ ((Lx2*M)[1:N,2:N+1]+(Lx3*M)[1:N,2:N+1]+(Lx4*M)[1:N,2:N+1]+(Ly2*M)[2:N+1,1:N]+(Ly3*M)[2:N+1,1:N]+(Ly4*M)[2:N+1,1:N])

end

@testset "regular x grid (dx=0.25) and irregular y grid" begin

    N = 100
    dx = 0.25
    dx_vec = 0.25:0.25:(N+2)*0.25
    dy = cumsum(rand(N+2))
    M = zeros(N+2,N+2)

    for i in 1:N+2
        for j in 1:N+2
            M[i,j] = cos(dx*i)+sin(dy[j])
        end
    end

    # Lx2 has 0 boundary points
    Lx2 = CenteredDifference{1}(2,2,dx,N)
    # Lx3 has 1 boundary point
    Lx3 = 1.45*CenteredDifference{1}(3,3,dx,N)
    # Lx4 has 2 boundary points
    Lx4 = CenteredDifference{1}(4,4,dx,N)

    # Ly2 has 0 boundary points
    Ly2 = 8.14*CenteredDifference{2}(2,2,dy,N)
    # Ly3 has 1 boundary point
    Ly3 = CenteredDifference{2}(3,3,dy,N)
    # Ly4 has 2 boundary points
    Ly4 = 4.567*CenteredDifference{2}(4,4,dy,N)

    # Test that composition of all x-operators works
    A = Lx2 + Lx3 + Lx4
    M_temp = zeros(N,N+2)
    mul!(M_temp, A, M)
    @test M_temp ≈ (Lx2*M + Lx3*M + Lx4*M)

    # Test that composition of all y-operators works
    A = Ly2 + Ly3 + Ly4
    M_temp = zeros(N+2,N)
    mul!(M_temp, A, M)
    @test M_temp ≈ (Ly2*M + Ly3*M + Ly4*M)

    # Test that composition of both x and y operators works
    A = Lx2 + Ly2 + Lx3 + Ly3 + Ly4 + Lx4
    M_temp = zeros(N,N)
    mul!(M_temp, A, M)
    @test M_temp ≈ ((Lx2*M)[1:N,2:N+1]+(Lx3*M)[1:N,2:N+1]+(Lx4*M)[1:N,2:N+1]+(Ly2*M)[2:N+1,1:N]+(Ly3*M)[2:N+1,1:N]+(Ly4*M)[2:N+1,1:N])

    # Last case where we now have some `irregular-grid` operators operating on the
    # regular-spaced axis x

    # These operators are operating on the regular grid x, but are constructed as though
    # they were irregular grid operators. Hence we test if we can seperate irregular and
    # regular gird operators on the same axis
    # Lx2 has 0 boundary points
    _Lx2 = 4.532*CenteredDifference{1}(2,2,dx_vec,N)
    # Lx3 has 1 boundary point
    _Lx3 = 0.235*CenteredDifference{1}(3,3,dx_vec,N)
    # Lx4 has 2 boundary points
    _Lx4 = CenteredDifference{1}(4,4,dx_vec,N)

    A += _Lx2 + _Lx3 + _Lx4
    mul!(M_temp, A, M)
    @test M_temp ≈ ((_Lx2*M)[1:N,2:N+1]+(_Lx3*M)[1:N,2:N+1]+(_Lx4*M)[1:N,2:N+1]+(Lx2*M)[1:N,2:N+1]+(Lx3*M)[1:N,2:N+1]+(Lx4*M)[1:N,2:N+1]+(Ly2*M)[2:N+1,1:N]+(Ly3*M)[2:N+1,1:N]+(Ly4*M)[2:N+1,1:N])

end

@testset "irregular x grid and regular y grid (dy = 0.25)" begin

    N = 100
    dy = 0.25
    dy_vec = 0.25:0.25:(N+2)*0.25
    dx = cumsum(rand(N+2))
    M = zeros(N+2,N+2)

    for i in 1:N+2
        for j in 1:N+2
            M[i,j] = cos(dx[i])+sin(dy*j)
        end
    end

    # Lx2 has 0 boundary points
    Lx2 = CenteredDifference{1}(2,2,dx,N)
    # Lx3 has 1 boundary point
    Lx3 = 1.45*CenteredDifference{1}(3,3,dx,N)
    # Lx4 has 2 boundary points
    Lx4 = CenteredDifference{1}(4,4,dx,N)

    # Ly2 has 0 boundary points
    Ly2 = 8.14*CenteredDifference{2}(2,2,dy,N)
    # Ly3 has 1 boundary point
    Ly3 = CenteredDifference{2}(3,3,dy,N)
    # Ly4 has 2 boundary points
    Ly4 = 4.567*CenteredDifference{2}(4,4,dy,N)

    # Test that composition of all x-operators works
    A = Lx2 + Lx3 + Lx4
    M_temp = zeros(N,N+2)
    mul!(M_temp, A, M)
    @test M_temp ≈ (Lx2*M + Lx3*M + Lx4*M)

    # Test that composition of all y-operators works
    A = Ly2 + Ly3 + Ly4
    M_temp = zeros(N+2,N)
    mul!(M_temp, A, M)
    @test M_temp ≈ (Ly2*M + Ly3*M + Ly4*M)

    # Test that composition of both x and y operators works
    A = Lx2 + Ly2 + Lx3 + Ly3 + Ly4 + Lx4
    M_temp = zeros(N,N)
    mul!(M_temp, A, M)
    @test M_temp ≈ ((Lx2*M)[1:N,2:N+1]+(Lx3*M)[1:N,2:N+1]+(Lx4*M)[1:N,2:N+1]+(Ly2*M)[2:N+1,1:N]+(Ly3*M)[2:N+1,1:N]+(Ly4*M)[2:N+1,1:N])

    # Last case where we now have some `irregular-grid` operators operating on the
    # regular-spaced axis y

    # These operators are operating on the regular grid y, but are constructed as though
    # they were irregular grid operators. Hence we test if we can seperate irregular and
    # regular gird operators on the same axis
    # Ly2 has 0 boundary points
    _Ly2 = CenteredDifference{2}(2,2,dy_vec,N)
    # Ly3 has 1 boundary point
    _Ly3 = CenteredDifference{2}(3,3,dy_vec,N)
    # Ly4 has 2 boundary points
    _Ly4 = 12.1*CenteredDifference{2}(4,4,dy_vec,N)

    A += _Ly2 + _Ly3 + _Ly4
    mul!(M_temp, A, M)
    @test M_temp ≈ ((Lx2*M)[1:N,2:N+1]+(Lx3*M)[1:N,2:N+1]+(Lx4*M)[1:N,2:N+1]+(Ly2*M)[2:N+1,1:N]+(Ly3*M)[2:N+1,1:N]+(Ly4*M)[2:N+1,1:N]+(_Ly2*M)[2:N+1,1:N]+(_Ly3*M)[2:N+1,1:N]+(_Ly4*M)[2:N+1,1:N])

end

################################################################################
# 3D Multiplication Tests
################################################################################

@testset "3D Multiplication with no boundary points and dx = dy = dz = 1.0" begin

    # Test (Lxx + Lyy + Lzz)*M, dx = dy = dz = 1.0, no coefficient
    N = 100
    M = zeros(N+2,N+2,N+2)
    M_temp = zeros(N,N,N)

    for i in 1:N+2
        for j in 1:N+2
            for k in 1:N+2
                M[i,j,k] = cos(0.1i)+sin(0.1j) + exp(0.01k)
            end
        end
    end

    Lxx = CenteredDifference{1}(2,2,1.0,N)
    Lyy = CenteredDifference{2}(2,2,1.0,N)
    Lzz = CenteredDifference{3}(2,2,1.0,N)
    A = Lxx + Lyy + Lzz

    mul!(M_temp, A, M)

    @test M_temp ≈ ((Lxx*M)[1:N,2:N+1,2:N+1] + (Lyy*M)[2:N+1,1:N,2:N+1] + (Lzz*M)[2:N+1,2:N+1,1:N])

    # Test a single axis, multiple operators: (Lx + Lxx)*M, dx = 1.0
    Lx = CenteredDifference{1}(1,2,1.0,N)
    A = Lx + Lxx

    M_temp = zeros(N,N+2,N+2)
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Lx*M)+(Lxx*M))

    # Test a single axis, multiple operators: (Ly + Lyy)*M, dy = 1.0, no coefficient
    Ly = CenteredDifference{2}(1,2,1.0,N)
    A = Ly + Lyy

    M_temp = zeros(N+2,N,N+2)
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Ly*M)+(Lyy*M))

    # Test a single axis, multiple operators: (Lz + Lzz)*M, dz = 1.0, no coefficient
    Lz = CenteredDifference{3}(1,2,1.0,N)
    A = Lz + Lzz

    M_temp = zeros(N+2,N+2,N)
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Lz*M)+(Lzz*M))

    # Test multiple operators on both axis: (Lx + Ly + Lxx + Lyy)*M, no coefficient
    A = Lx + Ly + Lxx + Lyy
    M_temp = zeros(N,N,N+2)
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Lx*M)[1:N,2:N+1,:] +(Ly*M)[2:N+1,1:N,:] + (Lxx*M)[1:N,2:N+1,:] +(Lyy*M)[2:N+1,1:N,:])

    # Test multiple operators on both axis: (Lx + Lxx + Lz + Lzz)*M, no coefficient
    A = Lx + Lxx + Lz + Lzz
    M_temp = zeros(N,N+2,N)
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Lx*M)[1:N,:,2:N+1] + (Lxx*M)[1:N,:,2:N+1]  + (Lz*M)[2:N+1,:,1:N] +(Lzz*M)[2:N+1,:,1:N])


    # Test multiple operators on both axis: (Ly + Lyy + Lz + Lzz)*M, no coefficient
    A = Ly + Lyy + Lz + Lzz
    M_temp = zeros(N+2,N,N)
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Ly*M)[:,1:N,2:N+1] + (Lyy*M)[:,1:N,2:N+1] + (Lz*M)[:,2:N+1,1:N] +(Lzz*M)[:,2:N+1,1:N])

    # Test multiple operators on both axis: (Lx + Ly + Lxx + Lyy + Lz + Lzz)*M, no coefficient
    A = Lx + Ly + Lxx + Lyy + Lz + Lzz
    M_temp = zeros(N,N,N)
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Lx*M)[1:N,2:N+1,2:N+1] +(Ly*M)[2:N+1,1:N,2:N+1] + (Lxx*M)[1:N,2:N+1,2:N+1] +(Lyy*M)[2:N+1,1:N,2:N+1] + (Lz*M)[2:N+1,2:N+1,1:N] +(Lzz*M)[2:N+1,2:N+1,1:N])

end
