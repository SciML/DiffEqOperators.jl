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

@testset "3D Multiplication with identical bpc and dx = dy = dz = 1.0" begin

    N = 100
    M = zeros(N+2,N+2,N+2)
    for i in 1:N+2
        for j in 1:N+2
            for k in 1:N+2
                M[i,j,k] = cos(0.1i)+sin(0.1j) + exp(0.01k)
            end
        end
    end

    # Test a single axis, multiple operators: (Lxxx + Lxxxx)*M, dx = dy = dz = 1.0
    # Lx3 and Lx4 have the same number of boundary points
    Lx3 = CenteredDifference{1}(3,4,1.0,N)
    Lx4 = CenteredDifference{1}(4,4,1.0,N)
    A = Lx3 + Lx4

    M_temp = zeros(N,N+2,N+2)
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Lx3*M)+(Lx4*M))

    # Test a single axis, multiple operators: (Lyyy + Lyyyy)*M, dx = dy = dz = 1.0, no coefficient
    # Ly3 and Ly4 have the same number of boundary points
    Ly3 = CenteredDifference{2}(3,4,1.0,N)
    Ly4 = CenteredDifference{2}(4,4,1.0,N)
    A = Ly3 + Ly4

    M_temp = zeros(N+2,N,N+2)
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Ly3*M)+(Ly4*M))

    # Test a single axis, multiple operators: (Lzzz + Lzzzz)*M, dx = dy = dz = 1.0, no coefficient
    # Lz3 and Lz4 have the same number of boundary points
    Lz3 = CenteredDifference{3}(3,4,1.0,N)
    Lz4 = CenteredDifference{3}(4,4,1.0,N)
    A = Lz3 + Lz4

    M_temp = zeros(N+2,N+2,N)
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Lz3*M)+(Lz4*M))

    # Test (Lxxxx + Lyyyy)*M, dx = 1.0, no coefficient, two boundary points on each axis
    M_temp = zeros(N,N,N+2)
    A = Lx4 + Ly4
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Lx4*M)[1:N,2:N+1,:] +(Ly4*M)[2:N+1,1:N,:])

    # Test (Lxxx + Lyyy)*M, no coefficient. These operators have non-symmetric interior stencils
    A = Lx3 + Ly3
    M_temp = zeros(N,N,N+2)
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Lx3*M)[1:N,2:N+1,:] +(Ly3*M)[2:N+1,1:N,:])

    # Test multiple operators on both axis: (Lxxx + Lyyy + Lxxxx + Lyyyy)*M, no coefficient
    A = Lx3 + Ly3 + Lx4 + Ly4
    M_temp = zeros(N,N,N+2)
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Lx3*M)[1:N,2:N+1,:] +(Ly3*M)[2:N+1,1:N,:] + (Lx4*M)[1:N,2:N+1,:] +(Ly4*M)[2:N+1,1:N,:])

    # Test (Lxxxx + Lzzzz)*M, dx = 1.0, no coefficient, two boundary points on each axis
    M_temp = zeros(N,N+2,N)
    A = Lx4 + Lz4
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Lx4*M)[1:N,:,2:N+1] +(Lz4*M)[2:N+1,:,1:N])

    # Test (Lxxx + Lzzz)*M, no coefficient. These operators have non-symmetric interior stencils
    A = Lx3 + Lz3
    M_temp = zeros(N,N+2,N)
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Lx3*M)[1:N,:,2:N+1] +(Lz3*M)[2:N+1,:,1:N])

    # Test multiple operators on both axis: (Lxxx + Lzzz + Lxxxx + Lzzzz)*M, no coefficient
    A = Lx3 + Lz3 + Lx4 + Lz4
    M_temp = zeros(N,N+2,N)
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Lx3*M)[1:N,:,2:N+1] +(Lz3*M)[2:N+1,:,1:N] + (Lx4*M)[1:N,:,2:N+1] +(Lz4*M)[2:N+1,:,1:N])

    # Test (Lyyyy + Lzzzz)*M, dx = 1.0, no coefficient, two boundary points on each axis
    M_temp = zeros(N+2,N,N)
    A = Ly4 + Lz4
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Ly4*M)[:,1:N,2:N+1] +(Lz4*M)[:,2:N+1,1:N])

    # Test (Lyyy + Lzzz)*M, no coefficient. These operators have non-symmetric interior stencils
    A = Ly3 + Lz3
    M_temp = zeros(N+2,N,N)
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Ly3*M)[:,1:N,2:N+1] +(Lz3*M)[:,2:N+1,1:N])

    # Test multiple operators on both axis: (Lyyy + Lzzz + Lyyyy + Lzzzz)*M, no coefficient
    A = Ly3 + Lz3 + Ly4 + Lz4
    M_temp = zeros(N+2,N,N)
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Ly3*M)[:,1:N,2:N+1] +(Lz3*M)[:,2:N+1,1:N] + (Ly4*M)[:,1:N,2:N+1] +(Lz4*M)[:,2:N+1,1:N])

    # Test a single operator on each axis: (Lx3 + Ly3 + Lz3)*M, no coefficient
    A = Lx3 + Ly3 + Lz3
    M_temp = zeros(N,N,N)
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Lx3*M)[1:N,2:N+1,2:N+1] + (Ly3*M)[2:N+1,1:N,2:N+1] +(Lz3*M)[2:N+1,2:N+1,1:N])

    # Test a single operator on each axis: (Lx4 + Ly4 + Lz4)*M, no coefficient
    A = Lx4 + Ly4 + Lz4
    M_temp = zeros(N,N,N)
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Lx4*M)[1:N,2:N+1,2:N+1] + (Ly4*M)[2:N+1,1:N,2:N+1] +(Lz4*M)[2:N+1,2:N+1,1:N])

    # Test multiple operators on each axis: (Lx3 + Ly3 + Lz3 + Lx4 + Ly4 + Lz4)*M, no coefficient
    A = Lx3 + Ly3 + Lz3 + Lx4 + Ly4 + Lz4
    M_temp = zeros(N,N,N)
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Lx3*M)[1:N,2:N+1,2:N+1] + (Ly3*M)[2:N+1,1:N,2:N+1] +(Lz3*M)[2:N+1,2:N+1,1:N] + (Lx4*M)[1:N,2:N+1,2:N+1] + (Ly4*M)[2:N+1,1:N,2:N+1] +(Lz4*M)[2:N+1,2:N+1,1:N])

end

@testset "3D Multiplication with differing bpc and dx = dy = dz = 1.0" begin

    N = 100
    M = zeros(N+2,N+2,N+2)
    for i in 1:N+2
        for j in 1:N+2
            for k in 1:N+2
                M[i,j,k] = cos(0.1i)+sin(0.1j) + exp(0.01k)
            end
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
    M_temp = zeros(N,N+2,N+2)
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

    # Test a single axis, multiple operators: (Lyy+Lyyyy)*M, dx = 1.0
    A = Ly2+Ly4
    M_temp = zeros(N+2,N,N+2)
    mul!(M_temp, A, M)
    @test M_temp ≈ ((Ly2*M) + (Ly4*M))

    # Test a single axis, multiple operators: (Lyy++Lyyy+Lyyyy)*M, dx = 1.0
    A += Ly3
    mul!(M_temp, A, M)
    @test M_temp ≈ ((Ly2*M) + (Ly3*M) + (Ly4*M))


    # Lz2 has 0 boundary points
    Lz2 = CenteredDifference{3}(2,2,1.0,N)
    # Lz3 has 1 boundary point
    Lz3 = CenteredDifference{3}(3,3,1.0,N)
    # Lz4 has 2 boundary points
    Lz4 = CenteredDifference{3}(4,4,1.0,N)

    # Test a single axis, multiple operators: (Lzy+Lzzzz)*M, dz = 1.0
    A = Lz2+Lz4
    M_temp = zeros(N+2,N+2,N)
    mul!(M_temp, A, M)
    @test M_temp ≈ ((Lz2*M) + (Lz4*M))

    # Test a single axis, multiple operators: (Lzz++Lzzz+Lzzzz)*M, dz = 1.0
    A += Lz3
    mul!(M_temp, A, M)
    @test M_temp ≈ ((Lz2*M) + (Lz3*M) + (Lz4*M))

    # Test multiple operators on two axis: (Lxx + Lyy + Lxxx + Lyyy + Lxxxx + Lyyyy)*M, no coefficient
    A = Lx2 + Ly2 + Lx3 + Ly3 + Lx4 + Ly4
    M_temp = zeros(N,N,N+2)
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Lx2*M)[1:N,2:N+1,:]+(Ly2*M)[2:N+1,1:N,:]+(Lx3*M)[1:N,2:N+1,:] +(Ly3*M)[2:N+1,1:N,:] + (Lx4*M)[1:N,2:N+1,:] +(Ly4*M)[2:N+1,1:N,:])

    # Test multiple operators on two axis: (Lxx + Lzz + Lxxx + Lzzz + Lxxxx + Lzzzz)*M, no coefficient
    A = Lx2 + Lz2 + Lx3 + Lz3 + Lx4 + Lz4
    M_temp = zeros(N,N+2,N)
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Lx2*M)[1:N,:,2:N+1]+(Lz2*M)[2:N+1,:,1:N]+(Lx3*M)[1:N,:,2:N+1] +(Lz3*M)[2:N+1,:,1:N] + (Lx4*M)[1:N,:,2:N+1] +(Lz4*M)[2:N+1,:,1:N])

    # Test multiple operators on two axis: (Lxx + Lzz + Lxxx + Lzzz + Lxxxx + Lzzzz)*M, no coefficient
    A = Ly2 + Lz2 + Ly3 + Lz3 + Ly4 + Lz4
    M_temp = zeros(N+2,N,N)
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Ly2*M)[:,1:N,2:N+1]+(Lz2*M)[:,2:N+1,1:N]+(Ly3*M)[:,1:N,2:N+1] +(Lz3*M)[:,2:N+1,1:N] + (Ly4*M)[:,1:N,2:N+1] +(Lz4*M)[:,2:N+1,1:N])

    # Test operators on all three axis (Lxx + Lyy + Lzz + Lxxx + Lyyy + Lzzz + Lxxxx + Lyyyy + Lzzzz)*M, no coefficient
    A = Lx2 + Ly2 + Lz2 + Lx3 + Ly3 + Lz3 + Lx4 + Ly4 + Lz4
    M_temp = zeros(N,N,N)
    mul!(M_temp, A, M)
    @test M_temp ≈ ((Lx2*M)[1:N,2:N+1,2:N+1]+(Ly2*M)[2:N+1,1:N,2:N+1]+(Lz2*M)[2:N+1,2:N+1,1:N]+(Lx3*M)[1:N,2:N+1,2:N+1] +(Ly3*M)[2:N+1,1:N,2:N+1]
     + (Lz3*M)[2:N+1,2:N+1,1:N] + (Lx4*M)[1:N,2:N+1,2:N+1] +(Ly4*M)[2:N+1,1:N,2:N+1])+(Lz4*M)[2:N+1,2:N+1,1:N]

end

@testset "3D Multiplication with identical bpc and dx = 0.5, dy = 0.25, dz = 0.05, no coefficients" begin

    N = 100
    M = zeros(N+2,N+2,N+2)
    dx = 0.5
    dy = 0.25
    dz = 0.05
    for i in 1:N+2
        for j in 1:N+2
            for k in 1:N+2
                M[i,j,k] = cos(dx*i)+sin(dy*j) + exp(dz*k)
            end
        end
    end

    # Test a single axis, multiple operators: (Lxxx + Lxxxx)*M
    # Lx3 and Lx4 have the same number of boundary points
    Lx3 = CenteredDifference{1}(3,4,dx,N)
    Lx4 = CenteredDifference{1}(4,4,dx,N)
    A = Lx3 + Lx4

    M_temp = zeros(N,N+2,N+2)
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Lx3*M)+(Lx4*M))

    # Test a single axis, multiple operators: (Lyyy + Lyyyy)*M
    # Ly3 and Ly4 have the same number of boundary points
    Ly3 = CenteredDifference{2}(3,4,dy,N)
    Ly4 = CenteredDifference{2}(4,4,dy,N)
    A = Ly3 + Ly4

    M_temp = zeros(N+2,N,N+2)
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Ly3*M)+(Ly4*M))

    # Test a single axis, multiple operators: (Lzzz + Lzzzz)*M
    # Lz3 and Lz4 have the same number of boundary points
    Lz3 = CenteredDifference{3}(3,4,dz,N)
    Lz4 = CenteredDifference{3}(4,4,dz,N)
    A = Lz3 + Lz4

    M_temp = zeros(N+2,N+2,N)
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Lz3*M)+(Lz4*M))

    # Test (Lxxxx + Lyyyy)*M, two boundary points on each axis
    M_temp = zeros(N,N,N+2)
    A = Lx4 + Ly4
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Lx4*M)[1:N,2:N+1,:] +(Ly4*M)[2:N+1,1:N,:])

    # Test (Lxxx + Lyyy)*M, no coefficient. These operators have non-symmetric interior stencils
    A = Lx3 + Ly3
    M_temp = zeros(N,N,N+2)
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Lx3*M)[1:N,2:N+1,:] +(Ly3*M)[2:N+1,1:N,:])

    # Test multiple operators on both axis: (Lxxx + Lyyy + Lxxxx + Lyyyy)*M
    A = Lx3 + Ly3 + Lx4 + Ly4
    M_temp = zeros(N,N,N+2)
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Lx3*M)[1:N,2:N+1,:] +(Ly3*M)[2:N+1,1:N,:] + (Lx4*M)[1:N,2:N+1,:] +(Ly4*M)[2:N+1,1:N,:])

    # Test (Lxxxx + Lzzzz)*M, two boundary points on each axis
    M_temp = zeros(N,N+2,N)
    A = Lx4 + Lz4
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Lx4*M)[1:N,:,2:N+1] +(Lz4*M)[2:N+1,:,1:N])

    # Test (Lxxx + Lzzz)*M, no coefficient. These operators have non-symmetric interior stencils
    A = Lx3 + Lz3
    M_temp = zeros(N,N+2,N)
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Lx3*M)[1:N,:,2:N+1] +(Lz3*M)[2:N+1,:,1:N])

    # Test multiple operators on both axis: (Lxxx + Lzzz + Lxxxx + Lzzzz)*M
    A = Lx3 + Lz3 + Lx4 + Lz4
    M_temp = zeros(N,N+2,N)
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Lx3*M)[1:N,:,2:N+1] +(Lz3*M)[2:N+1,:,1:N] + (Lx4*M)[1:N,:,2:N+1] +(Lz4*M)[2:N+1,:,1:N])

    # Test (Lyyyy + Lzzzz)*M, dx = 1.0, no coefficient, two boundary points on each axis
    M_temp = zeros(N+2,N,N)
    A = Ly4 + Lz4
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Ly4*M)[:,1:N,2:N+1] +(Lz4*M)[:,2:N+1,1:N])

    # Test (Lyyy + Lzzz)*M, no coefficient. These operators have non-symmetric interior stencils
    A = Ly3 + Lz3
    M_temp = zeros(N+2,N,N)
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Ly3*M)[:,1:N,2:N+1] +(Lz3*M)[:,2:N+1,1:N])

    # Test multiple operators on both axis: (Lyyy + Lzzz + Lyyyy + Lzzzz)*M
    A = Ly3 + Lz3 + Ly4 + Lz4
    M_temp = zeros(N+2,N,N)
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Ly3*M)[:,1:N,2:N+1] +(Lz3*M)[:,2:N+1,1:N] + (Ly4*M)[:,1:N,2:N+1] +(Lz4*M)[:,2:N+1,1:N])

    # Test a single operator on each axis: (Lx3 + Ly3 + Lz3)*M
    A = Lx3 + Ly3 + Lz3
    M_temp = zeros(N,N,N)
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Lx3*M)[1:N,2:N+1,2:N+1] + (Ly3*M)[2:N+1,1:N,2:N+1] +(Lz3*M)[2:N+1,2:N+1,1:N])

    # Test a single operator on each axis: (Lx4 + Ly4 + Lz4)*M
    A = Lx4 + Ly4 + Lz4
    M_temp = zeros(N,N,N)
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Lx4*M)[1:N,2:N+1,2:N+1] + (Ly4*M)[2:N+1,1:N,2:N+1] +(Lz4*M)[2:N+1,2:N+1,1:N])

    # Test multiple operators on each axis: (Lx3 + Ly3 + Lz3 + Lx4 + Ly4 + Lz4)*M
    A = Lx3 + Ly3 + Lz3 + Lx4 + Ly4 + Lz4
    M_temp = zeros(N,N,N)
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Lx3*M)[1:N,2:N+1,2:N+1] + (Ly3*M)[2:N+1,1:N,2:N+1] +(Lz3*M)[2:N+1,2:N+1,1:N] + (Lx4*M)[1:N,2:N+1,2:N+1] + (Ly4*M)[2:N+1,1:N,2:N+1] +(Lz4*M)[2:N+1,2:N+1,1:N])

end

@testset "3D Multiplication with differing bpc and dx = 0.5, dy = 0.25, dz = 0.05, no coefficients" begin

    N = 100
    M = zeros(N+2,N+2,N+2)
    dx = 0.5
    dy = 0.25
    dz = 0.05
    for i in 1:N+2
        for j in 1:N+2
            for k in 1:N+2
                M[i,j,k] = cos(dx*i)+sin(dy*j) + exp(dz*k)
            end
        end
    end

    # Lx2 has 0 boundary points
    Lx2 = CenteredDifference{1}(2,2,dx,N)
    # Lx3 has 1 boundary point
    Lx3 = CenteredDifference{1}(3,3,dx,N)
    # Lx4 has 2 boundary points
    Lx4 = CenteredDifference{1}(4,4,dx,N)

    # Test a single axis, multiple operators: (Lxx+Lxxxx)*M
    A = Lx2+Lx4
    M_temp = zeros(N,N+2,N+2)
    mul!(M_temp, A, M)
    @test M_temp ≈ ((Lx2*M) + (Lx4*M))

    # Test a single axis, multiple operators: (Lxx++Lxxx+Lxxxx)*M
    A += Lx3
    mul!(M_temp, A, M)
    @test M_temp ≈ ((Lx2*M) + (Lx3*M) + (Lx4*M))


    # Ly2 has 0 boundary points
    Ly2 = CenteredDifference{2}(2,2,dy,N)
    # Ly3 has 1 boundary point
    Ly3 = CenteredDifference{2}(3,3,dy,N)
    # Ly4 has 2 boundary points
    Ly4 = CenteredDifference{2}(4,4,dy,N)

    # Test a single axis, multiple operators: (Lyy+Lyyyy)*M
    A = Ly2+Ly4
    M_temp = zeros(N+2,N,N+2)
    mul!(M_temp, A, M)
    @test M_temp ≈ ((Ly2*M) + (Ly4*M))

    # Test a single axis, multiple operators: (Lyy++Lyyy+Lyyyy)*M
    A += Ly3
    mul!(M_temp, A, M)
    @test M_temp ≈ ((Ly2*M) + (Ly3*M) + (Ly4*M))


    # Lz2 has 0 boundary points
    Lz2 = CenteredDifference{3}(2,2,dz,N)
    # Lz3 has 1 boundary point
    Lz3 = CenteredDifference{3}(3,3,dz,N)
    # Lz4 has 2 boundary points
    Lz4 = CenteredDifference{3}(4,4,dz,N)

    # Test a single axis, multiple operators: (Lzy+Lzzzz)*M
    A = Lz2+Lz4
    M_temp = zeros(N+2,N+2,N)
    mul!(M_temp, A, M)
    @test M_temp ≈ ((Lz2*M) + (Lz4*M))

    # Test a single axis, multiple operators: (Lzz++Lzzz+Lzzzz)*M
    A += Lz3
    mul!(M_temp, A, M)
    @test M_temp ≈ ((Lz2*M) + (Lz3*M) + (Lz4*M))

    # Test multiple operators on two axis: (Lxx + Lyy + Lxxx + Lyyy + Lxxxx + Lyyyy)*M
    A = Lx2 + Ly2 + Lx3 + Ly3 + Lx4 + Ly4
    M_temp = zeros(N,N,N+2)
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Lx2*M)[1:N,2:N+1,:]+(Ly2*M)[2:N+1,1:N,:]+(Lx3*M)[1:N,2:N+1,:] +(Ly3*M)[2:N+1,1:N,:] + (Lx4*M)[1:N,2:N+1,:] +(Ly4*M)[2:N+1,1:N,:])

    # Test multiple operators on two axis: (Lxx + Lzz + Lxxx + Lzzz + Lxxxx + Lzzzz)*M
    A = Lx2 + Lz2 + Lx3 + Lz3 + Lx4 + Lz4
    M_temp = zeros(N,N+2,N)
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Lx2*M)[1:N,:,2:N+1]+(Lz2*M)[2:N+1,:,1:N]+(Lx3*M)[1:N,:,2:N+1] +(Lz3*M)[2:N+1,:,1:N] + (Lx4*M)[1:N,:,2:N+1] +(Lz4*M)[2:N+1,:,1:N])

    # Test multiple operators on two axis: (Lxx + Lzz + Lxxx + Lzzz + Lxxxx + Lzzzz)*M
    A = Ly2 + Lz2 + Ly3 + Lz3 + Ly4 + Lz4
    M_temp = zeros(N+2,N,N)
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Ly2*M)[:,1:N,2:N+1]+(Lz2*M)[:,2:N+1,1:N]+(Ly3*M)[:,1:N,2:N+1] +(Lz3*M)[:,2:N+1,1:N] + (Ly4*M)[:,1:N,2:N+1] +(Lz4*M)[:,2:N+1,1:N])

    # Test operators on all three axis (Lxx + Lyy + Lzz + Lxxx + Lyyy + Lzzz + Lxxxx + Lyyyy + Lzzzz)*M, no coefficient
    A = Lx2 + Ly2 + Lz2 + Lx3 + Ly3 + Lz3 + Lx4 + Ly4 + Lz4
    M_temp = zeros(N,N,N)
    mul!(M_temp, A, M)
    @test M_temp ≈ ((Lx2*M)[1:N,2:N+1,2:N+1]+(Ly2*M)[2:N+1,1:N,2:N+1]+(Lz2*M)[2:N+1,2:N+1,1:N]+(Lx3*M)[1:N,2:N+1,2:N+1] +(Ly3*M)[2:N+1,1:N,2:N+1]
     + (Lz3*M)[2:N+1,2:N+1,1:N] + (Lx4*M)[1:N,2:N+1,2:N+1] +(Ly4*M)[2:N+1,1:N,2:N+1])+(Lz4*M)[2:N+1,2:N+1,1:N]

end

# This is the same test set as the above test set with the addition of coefficients
@testset "3D Multiplication with coefficients" begin

    N = 100
    M = zeros(N+2,N+2,N+2)
    dx = 0.5
    dy = 0.25
    dz = 0.05
    for i in 1:N+2
        for j in 1:N+2
            for k in 1:N+2
                M[i,j,k] = cos(dx*i)+sin(dy*j) + exp(dz*k)
            end
        end
    end

    # Lx2 has 0 boundary points
    Lx2 = 1.234*CenteredDifference{1}(2,2,dx,N)
    # Lx3 has 1 boundary point
    Lx3 = 2.456*CenteredDifference{1}(3,3,dx,N)
    # Lx4 has 2 boundary points
    Lx4 = CenteredDifference{1}(4,4,dx,N)

    # Test a single axis, multiple operators: (Lxx+Lxxxx)*M
    A = Lx2+Lx4
    M_temp = zeros(N,N+2,N+2)
    mul!(M_temp, A, M)
    @test M_temp ≈ ((Lx2*M) + (Lx4*M))

    # Test a single axis, multiple operators: (Lxx++Lxxx+Lxxxx)*M
    A += Lx3
    mul!(M_temp, A, M)
    @test M_temp ≈ ((Lx2*M) + (Lx3*M) + (Lx4*M))


    # Ly2 has 0 boundary points
    Ly2 = CenteredDifference{2}(2,2,dy,N)
    # Ly3 has 1 boundary point
    Ly3 = 4.014*CenteredDifference{2}(3,3,dy,N)
    # Ly4 has 2 boundary points
    Ly4 = 1.49*CenteredDifference{2}(4,4,dy,N)

    # Test a single axis, multiple operators: (Lyy+Lyyyy)*M
    A = Ly2+Ly4
    M_temp = zeros(N+2,N,N+2)
    mul!(M_temp, A, M)
    @test M_temp ≈ ((Ly2*M) + (Ly4*M))

    # Test a single axis, multiple operators: (Lyy++Lyyy+Lyyyy)*M
    A += Ly3
    mul!(M_temp, A, M)
    @test M_temp ≈ ((Ly2*M) + (Ly3*M) + (Ly4*M))


    # Lz2 has 0 boundary points
    Lz2 = 1.546*CenteredDifference{3}(2,2,dz,N)
    # Lz3 has 1 boundary point
    Lz3 = CenteredDifference{3}(3,3,dz,N)
    # Lz4 has 2 boundary points
    Lz4 = 0.55*CenteredDifference{3}(4,4,dz,N)

    # Test a single axis, multiple operators: (Lzy+Lzzzz)*M
    A = Lz2+Lz4
    M_temp = zeros(N+2,N+2,N)
    mul!(M_temp, A, M)
    @test M_temp ≈ ((Lz2*M) + (Lz4*M))

    # Test a single axis, multiple operators: (Lzz++Lzzz+Lzzzz)*M
    A += Lz3
    mul!(M_temp, A, M)
    @test M_temp ≈ ((Lz2*M) + (Lz3*M) + (Lz4*M))

    # Test multiple operators on two axis: (Lxx + Lyy + Lxxx + Lyyy + Lxxxx + Lyyyy)*M
    A = Lx2 + Ly2 + Lx3 + Ly3 + Lx4 + Ly4
    M_temp = zeros(N,N,N+2)
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Lx2*M)[1:N,2:N+1,:]+(Ly2*M)[2:N+1,1:N,:]+(Lx3*M)[1:N,2:N+1,:] +(Ly3*M)[2:N+1,1:N,:] + (Lx4*M)[1:N,2:N+1,:] +(Ly4*M)[2:N+1,1:N,:])

    # Test multiple operators on two axis: (Lxx + Lzz + Lxxx + Lzzz + Lxxxx + Lzzzz)*M
    A = Lx2 + Lz2 + Lx3 + Lz3 + Lx4 + Lz4
    M_temp = zeros(N,N+2,N)
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Lx2*M)[1:N,:,2:N+1]+(Lz2*M)[2:N+1,:,1:N]+(Lx3*M)[1:N,:,2:N+1] +(Lz3*M)[2:N+1,:,1:N] + (Lx4*M)[1:N,:,2:N+1] +(Lz4*M)[2:N+1,:,1:N])

    # Test multiple operators on two axis: (Lxx + Lzz + Lxxx + Lzzz + Lxxxx + Lzzzz)*M
    A = Ly2 + Lz2 + Ly3 + Lz3 + Ly4 + Lz4
    M_temp = zeros(N+2,N,N)
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Ly2*M)[:,1:N,2:N+1]+(Lz2*M)[:,2:N+1,1:N]+(Ly3*M)[:,1:N,2:N+1] +(Lz3*M)[:,2:N+1,1:N] + (Ly4*M)[:,1:N,2:N+1] +(Lz4*M)[:,2:N+1,1:N])

    # Test operators on all three axis (Lxx + Lyy + Lzz + Lxxx + Lyyy + Lzzz + Lxxxx + Lyyyy + Lzzzz)*M, no coefficient
    A = Lx2 + Ly2 + Lz2 + Lx3 + Ly3 + Lz3 + Lx4 + Ly4 + Lz4
    M_temp = zeros(N,N,N)
    mul!(M_temp, A, M)
    @test M_temp ≈ ((Lx2*M)[1:N,2:N+1,2:N+1]+(Ly2*M)[2:N+1,1:N,2:N+1]+(Lz2*M)[2:N+1,2:N+1,1:N]+(Lx3*M)[1:N,2:N+1,2:N+1] +(Ly3*M)[2:N+1,1:N,2:N+1]
     + (Lz3*M)[2:N+1,2:N+1,1:N] + (Lx4*M)[1:N,2:N+1,2:N+1] +(Ly4*M)[2:N+1,1:N,2:N+1])+(Lz4*M)[2:N+1,2:N+1,1:N]

end

@testset "x, y, and z are all irregular grids" begin

    N = 100
    dx = cumsum(rand(N+2))
    dy = cumsum(rand(N+2))
    dz = cumsum(rand(N+2))
    M = zeros(N+2,N+2,N+2)

    for i in 1:N+2
        for j in 1:N+2
            for k in 1:N+2
                M[i,j,k] = cos(dx[i])+sin(dy[j]) + exp(dz[k])
            end
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

    # Lz2 has 0 boundary points
    Lz2 = CenteredDifference{3}(2,2,dz,N)
    # Lz3 has 1 boundary point
    Lz3 = CenteredDifference{3}(3,3,dz,N)
    # Lz4 has 2 boundary points
    Lz4 = CenteredDifference{3}(4,4,dz,N)

    # Test composition of all first-dimension operators
    A = Lx2+Lx3+Lx4
    M_temp = zeros(N,N+2,N+2)
    mul!(M_temp, A, M)
    @test M_temp ≈ (Lx2*M + Lx3*M + Lx4*M)

    # Test composition of all second-dimension operators
    A = Ly2+Ly3+Ly4
    M_temp = zeros(N+2,N,N+2)
    mul!(M_temp, A, M)
    @test M_temp ≈ (Ly2*M + Ly3*M + Ly4*M)

    # Test composition of all third-dimension operators
    A = Lz2+Lz3+Lz4
    M_temp = zeros(N+2,N+2,N)
    mul!(M_temp, A, M)
    @test M_temp ≈ (Lz2*M + Lz3*M + Lz4*M)

    # Test multiple operators on two axis: (Lxx + Lyy + Lxxx + Lyyy + Lxxxx + Lyyyy)*M, no coefficient
    A = Lx2 + Ly2 + Lx3 + Ly3 + Lx4 + Ly4
    M_temp = zeros(N,N,N+2)
    mul!(M_temp, A, M)

    # Numerical errors accumulating for this test case, more so than the other tests
    @test isapprox(M_temp, ((Lx2*M)[1:N,2:N+1,:]+(Ly2*M)[2:N+1,1:N,:]+(Lx3*M)[1:N,2:N+1,:] +(Ly3*M)[2:N+1,1:N,:] + (Lx4*M)[1:N,2:N+1,:] +(Ly4*M)[2:N+1,1:N,:]), rtol = sqrt(length(M_temp)*eps(Float64)))

    # Test multiple operators on two axis: (Lxx + Lzz + Lxxx + Lzzz + Lxxxx + Lzzzz)*M, no coefficient
    A = Lx2 + Lz2 + Lx3 + Lz3 + Lx4 + Lz4
    M_temp = zeros(N,N+2,N)
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Lx2*M)[1:N,:,2:N+1]+(Lz2*M)[2:N+1,:,1:N]+(Lx3*M)[1:N,:,2:N+1] +(Lz3*M)[2:N+1,:,1:N] + (Lx4*M)[1:N,:,2:N+1] +(Lz4*M)[2:N+1,:,1:N])

    # Test multiple operators on two axis: (Lxx + Lzz + Lxxx + Lzzz + Lxxxx + Lzzzz)*M, no coefficient
    A = Ly2 + Lz2 + Ly3 + Lz3 + Ly4 + Lz4
    M_temp = zeros(N+2,N,N)
    mul!(M_temp, A, M)

    @test M_temp ≈ ((Ly2*M)[:,1:N,2:N+1]+(Lz2*M)[:,2:N+1,1:N]+(Ly3*M)[:,1:N,2:N+1] +(Lz3*M)[:,2:N+1,1:N] + (Ly4*M)[:,1:N,2:N+1] +(Lz4*M)[:,2:N+1,1:N])

    # Test composition of all operators
    A = Lx2+Lx3+Lx4+Ly2+Ly3+Ly4+Lz2+Lz3+Lz4
    M_temp = zeros(N,N,N)
    mul!(M_temp, A, M)
    @test M_temp ≈ ((Lx2*M)[1:N,2:N+1,2:N+1]+(Ly2*M)[2:N+1,1:N,2:N+1]+(Lz2*M)[2:N+1,2:N+1,1:N]+(Lx3*M)[1:N,2:N+1,2:N+1] +(Ly3*M)[2:N+1,1:N,2:N+1]
     + (Lz3*M)[2:N+1,2:N+1,1:N] + (Lx4*M)[1:N,2:N+1,2:N+1] +(Ly4*M)[2:N+1,1:N,2:N+1])+(Lz4*M)[2:N+1,2:N+1,1:N]

end

@testset "irregular x grid and regular y and z grids (dy = 0.25, dz = 0.05)" begin

    N = 100
    dy = 0.25
    dz = 0.05
    dy_vec = 0.25:0.25:(N+2)*0.25
    dz_vec = 0.05:0.05:(N+2)*0.05
    dx = cumsum(rand(N+2))
    M = zeros(N+2,N+2,N+2)

    for i in 1:N+2
        for j in 1:N+2
            for k in 1:N+2
                M[i,j,k] = cos(dx[i])+sin(dy*j)+exp(dz*k)
            end
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

    # Lz2 has 0 boundary points
    Lz2 = 1.4*CenteredDifference{3}(2,2,dz,N)
    # Lz3 has 1 boundary point
    Lz3 = CenteredDifference{3}(3,3,dz,N)
    # Lz4 has 2 boundary points
    Lz4 = CenteredDifference{3}(4,4,dz,N)

    # Test that composition of both x and y operators works
    A = Lx2 + Ly2 + Lx3 + Ly3 + Ly4 + Lx4
    M_temp = zeros(N,N,N+2)
    mul!(M_temp, A, M)
    @test M_temp ≈ ((Lx2*M)[1:N,2:N+1,:]+(Lx3*M)[1:N,2:N+1,:]+(Lx4*M)[1:N,2:N+1,:]+(Ly2*M)[2:N+1,1:N,:]+(Ly3*M)[2:N+1,1:N,:]+(Ly4*M)[2:N+1,1:N,:])

    # Test that composition of both x and z operators works
    A = Lx2 + Lz2 + Lx3 + Lz3 + Lz4 + Lx4
    M_temp = zeros(N,N+2,N)
    mul!(M_temp, A, M)
    @test M_temp ≈ ((Lx2*M)[1:N,:,2:N+1]+(Lx3*M)[1:N,:,2:N+1]+(Lx4*M)[1:N,:,2:N+1]+(Lz2*M)[2:N+1,:,1:N]+(Lz3*M)[2:N+1,:,1:N]+(Lz4*M)[2:N+1,:,1:N])

    # Test that composition of x, y, and z operators works
    A = Lx2 + Lz2 + Lx3 + Lz3 + Lz4 + Lx4 + Ly2 + Ly3 + Ly4
    M_temp = zeros(N,N,N)
    mul!(M_temp, A, M)
    @test M_temp ≈ ((Lx2*M)[1:N,2:N+1,2:N+1]+(Ly2*M)[2:N+1,1:N,2:N+1]+(Lz2*M)[2:N+1,2:N+1,1:N]+(Lx3*M)[1:N,2:N+1,2:N+1] +(Ly3*M)[2:N+1,1:N,2:N+1]
     + (Lz3*M)[2:N+1,2:N+1,1:N] + (Lx4*M)[1:N,2:N+1,2:N+1] +(Ly4*M)[2:N+1,1:N,2:N+1]+(Lz4*M)[2:N+1,2:N+1,1:N])

    # Last case where we now have some `irregular-grid` operators operating on the
    # regular-spaced axis y

    # These operators are operating on the regular grid y, but are constructed as though
    # they were irregular grid operators. Hence we test if we can seperate irregular and
    # regular gird operators on the same axis
    # Ly2 has 0 boundary points
    _Ly2 = CenteredDifference{2}(2,2,dy_vec,N)
    # Lz3 has 1 boundary point
    _Lz3 = CenteredDifference{3}(3,3,dz_vec,N)
    # Ly4 has 2 boundary points
    _Ly4 = 12.1*CenteredDifference{2}(4,4,dy_vec,N)

    A += _Ly2 + _Lz3 + _Ly4
    mul!(M_temp, A, M)
    @test M_temp ≈ ((Lx2*M)[1:N,2:N+1,2:N+1]+(Ly2*M)[2:N+1,1:N,2:N+1]+(Lz2*M)[2:N+1,2:N+1,1:N]+(Lx3*M)[1:N,2:N+1,2:N+1] +(Ly3*M)[2:N+1,1:N,2:N+1]
     + (Lz3*M)[2:N+1,2:N+1,1:N] + (Lx4*M)[1:N,2:N+1,2:N+1] +(Ly4*M)[2:N+1,1:N,2:N+1]+(Lz4*M)[2:N+1,2:N+1,1:N] + (_Ly2*M)[2:N+1,1:N,2:N+1]+(_Lz3*M)[2:N+1,2:N+1,1:N]+(_Ly4*M)[2:N+1,1:N,2:N+1])

end

@testset "irregular y grid and regular x and z gris (dx = 0.25, dz = 0.05)" begin

    N = 100
    dx = 0.25
    dz = 0.05
    dx_vec = 0.25:0.25:(N+2)*0.25
    dz_vec = 0.05:0.05:(N+2)*0.05
    dy = cumsum(rand(N+2))
    M = zeros(N+2,N+2,N+2)

    for i in 1:N+2
        for j in 1:N+2
            for k in 1:N+2
                M[i,j,k] = cos(dx*i)+sin(dy[j])+exp(dz*k)
            end
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

    # Lz2 has 0 boundary points
    Lz2 = 1.4*CenteredDifference{3}(2,2,dz,N)
    # Lz3 has 1 boundary point
    Lz3 = CenteredDifference{3}(3,3,dz,N)
    # Lz4 has 2 boundary points
    Lz4 = CenteredDifference{3}(4,4,dz,N)

    # Test that composition of both x and y operators works
    A = Lx2 + Ly2 + Lx3 + Ly3 + Ly4 + Lx4
    M_temp = zeros(N,N,N+2)
    mul!(M_temp, A, M)
    @test M_temp ≈ ((Lx2*M)[1:N,2:N+1,:]+(Lx3*M)[1:N,2:N+1,:]+(Lx4*M)[1:N,2:N+1,:]+(Ly2*M)[2:N+1,1:N,:]+(Ly3*M)[2:N+1,1:N,:]+(Ly4*M)[2:N+1,1:N,:])

    # Test that composition of both y and z operators works
    A = Ly2 + Ly3 + Ly4 + Lz2 + Lz3 + Lz4
    M_temp = zeros(N+2,N,N)
    mul!(M_temp, A, M)
    # Need to figure out why this test is exploding
    @test_broken M_temp ≈ ((Ly2*M)[:,1:N,2:N+1]+(Ly3*M)[:,1:N,2:N+1]+(Ly4*M)[:,1:N,2:N+1]+(Lz2*M)[:,2:N+1,1:N]+(Lz3*M)[:,2:N+1,1:N]+(Lz4*M)[:,2:N+1,1:N])

    ############################################################################
    # Tests to isolate the above problem
    ############################################################################

    A = Ly2 + Ly3 + Ly4
    M_temp = zeros(N+2,N,N+2)
    mul!(M_temp, A, M)
    @test M_temp ≈ Ly2*M + Ly3*M + Ly4*M

    ### Test the addition of Lz2, Lz3, Lz4 seperately

    A = Ly2 + Ly3 + Ly4 + Lz2
    M_temp = zeros(N+2,N,N)
    mul!(M_temp, A, M)
    @test M_temp ≈ ((Ly2*M)[:,1:N,2:N+1]+(Ly3*M)[:,1:N,2:N+1]+(Ly4*M)[:,1:N,2:N+1]+(Lz2*M)[:,2:N+1,1:N])

    A = Ly2 + Ly3 + Ly4 + Lz3
    M_temp = zeros(N+2,N,N)
    mul!(M_temp, A, M)
    @test M_temp ≈ ((Ly2*M)[:,1:N,2:N+1]+(Ly3*M)[:,1:N,2:N+1]+(Ly4*M)[:,1:N,2:N+1]+(Lz3*M)[:,2:N+1,1:N])

    A = Ly2 + Ly3 + Ly4 + Lz4
    M_temp = zeros(N+2,N,N)
    mul!(M_temp, A, M)
    @test M_temp ≈ ((Ly2*M)[:,1:N,2:N+1]+(Ly3*M)[:,1:N,2:N+1]+(Ly4*M)[:,1:N,2:N+1]+(Lz4*M)[:,2:N+1,1:N])

    ###

    A = Ly2 + Ly3 + Ly4 + Lz2 + Lz3
    M_temp = zeros(N+2,N,N)
    mul!(M_temp, A, M)
    @test_broken M_temp ≈ ((Ly2*M)[:,1:N,2:N+1]+(Ly3*M)[:,1:N,2:N+1]+(Ly4*M)[:,1:N,2:N+1]+(Lz2*M)[:,2:N+1,1:N]+(Lz3*M)[:,2:N+1,1:N])

    A = Ly2 + Ly3 + Ly4 + Lz2 + Lz4
    M_temp = zeros(N+2,N,N)
    mul!(M_temp, A, M)
    @test_broken M_temp ≈ ((Ly2*M)[:,1:N,2:N+1]+(Ly3*M)[:,1:N,2:N+1]+(Ly4*M)[:,1:N,2:N+1]+(Lz2*M)[:,2:N+1,1:N]+(Lz4*M)[:,2:N+1,1:N])

    A = Ly2 + Ly3 + Ly4 + Lz3 + Lz4
    M_temp = zeros(N+2,N,N)
    mul!(M_temp, A, M)
    @test_broken M_temp ≈ ((Ly2*M)[:,1:N,2:N+1]+(Ly3*M)[:,1:N,2:N+1]+(Ly4*M)[:,1:N,2:N+1]+(Lz3*M)[:,2:N+1,1:N]+(Lz4*M)[:,2:N+1,1:N])


    ###

    A = Ly2 + Ly3 + Ly4 + Lz2 + Lz2
    M_temp = zeros(N+2,N,N)
    mul!(M_temp, A, M)
    @test_broken M_temp ≈ ((Ly2*M)[:,1:N,2:N+1]+(Ly3*M)[:,1:N,2:N+1]+(Ly4*M)[:,1:N,2:N+1]+(Lz2*M)[:,2:N+1,1:N]+(Lz2*M)[:,2:N+1,1:N])
    # It appears that multiple z operators with some y operators is causing the issues

    ###

    A = Lz2 + Lz3 + Lz4
    M_temp = zeros(N+2,N+2,N)
    mul!(M_temp, A, M)
    @test M_temp ≈ Lz2*M + Lz3*M + Lz4*M

    A = Lz2 + Lz3 + Lz4 + Ly2
    M_temp = zeros(N+2,N,N)
    mul!(M_temp, A, M)
    @test M_temp ≈ ((Ly2*M)[:,1:N,2:N+1]+(Lz2*M)[:,2:N+1,1:N]+(Lz3*M)[:,2:N+1,1:N]+(Lz4*M)[:,2:N+1,1:N])

    A = Lz2 + Lz3 + Lz4 + Ly3
    M_temp = zeros(N+2,N,N)
    mul!(M_temp, A, M)
    @test_broken M_temp ≈ ((Ly3*M)[:,1:N,2:N+1]+(Lz2*M)[:,2:N+1,1:N]+(Lz3*M)[:,2:N+1,1:N]+(Lz4*M)[:,2:N+1,1:N])
    # It seems that the y paddign could be the issue

    A = Lz2 + Lz3 + Lz4 + Ly4
    M_temp = zeros(N+2,N,N)
    mul!(M_temp, A, M)
    @test_broken M_temp ≈ ((Ly4*M)[:,1:N,2:N+1]+(Lz2*M)[:,2:N+1,1:N]+(Lz3*M)[:,2:N+1,1:N]+(Lz4*M)[:,2:N+1,1:N])
    # It seems that the y paddign could be the issue

    A = Lz2 + Lz3 + Lz4 + Ly4 + Ly3
    M_temp = zeros(N+2,N,N)
    mul!(M_temp, A, M)
    @test_broken M_temp ≈ ((Ly3*M)[:,1:N,2:N+1]+(Ly4*M)[:,1:N,2:N+1]+(Lz2*M)[:,2:N+1,1:N]+(Lz3*M)[:,2:N+1,1:N]+(Lz4*M)[:,2:N+1,1:N])
    # It seems that the y paddign could be the issue

    A = Lz2 + Lz3 + Lz4 + Ly4 + Ly2
    M_temp = zeros(N+2,N,N)
    mul!(M_temp, A, M)
    @test_broken M_temp ≈ ((Ly2*M)[:,1:N,2:N+1]+(Ly4*M)[:,1:N,2:N+1]+(Lz2*M)[:,2:N+1,1:N]+(Lz3*M)[:,2:N+1,1:N]+(Lz4*M)[:,2:N+1,1:N])
    # It seems that the y paddign could be the issue

    ###

    A = Lz2 + Lz3 +Ly3
    M_temp = zeros(N+2,N,N)
    mul!(M_temp, A, M)
    @test_broken M_temp ≈ ((Ly3*M)[:,1:N,2:N+1]+(Lz2*M)[:,2:N+1,1:N]+(Lz3*M)[:,2:N+1,1:N])

    A = Lz3 +Ly3
    M_temp = zeros(N+2,N,N)
    mul!(M_temp, A, M)
    @test M_temp ≈ ((Ly3*M)[:,1:N,2:N+1]+(Lz3*M)[:,2:N+1,1:N])

    A = Lz2 + Lz3 +Ly2
    M_temp = zeros(N+2,N,N)
    mul!(M_temp, A, M)
    @test M_temp ≈ ((Ly2*M)[:,1:N,2:N+1]+(Lz2*M)[:,2:N+1,1:N]+(Lz3*M)[:,2:N+1,1:N])

    # It seems that the padding of y is forcing multiple z operators to fail



    ############################################################################
    ############################################################################

    # Test that composition of x, y, and z operators works
    A = Lx2 + Lz2 + Lx3 + Lz3 + Lz4 + Lx4 + Ly2 + Ly3 + Ly4
    M_temp = zeros(N,N,N)
    mul!(M_temp, A, M)
    @test M_temp ≈ ((Lx2*M)[1:N,2:N+1,2:N+1]+(Ly2*M)[2:N+1,1:N,2:N+1]+(Lz2*M)[2:N+1,2:N+1,1:N]+(Lx3*M)[1:N,2:N+1,2:N+1] +(Ly3*M)[2:N+1,1:N,2:N+1]
     + (Lz3*M)[2:N+1,2:N+1,1:N] + (Lx4*M)[1:N,2:N+1,2:N+1] +(Ly4*M)[2:N+1,1:N,2:N+1]+(Lz4*M)[2:N+1,2:N+1,1:N])

    # Last case where we now have some `irregular-grid` operators operating on the
    # regular-spaced axis x and z

    # These operators are operating on the regular grid x and z, but are constructed as though
    # they were irregular grid operators. Hence we test if we can seperate irregular and
    # regular gird operators on the same axis
    # Lx2 has 0 boundary points
    _Lx2 = CenteredDifference{1}(2,2,dx_vec,N)
    # Lz3 has 1 boundary point
    _Lz3 = CenteredDifference{3}(3,3,dz_vec,N)
    # Lx4 has 2 boundary points
    _Lx4 = 12.1*CenteredDifference{1}(4,4,dx_vec,N)

    A += _Lx2 + _Lz3 + _Lx4
    mul!(M_temp, A, M)
    @test M_temp ≈ ((Lx2*M)[1:N,2:N+1,2:N+1]+(Ly2*M)[2:N+1,1:N,2:N+1]+(Lz2*M)[2:N+1,2:N+1,1:N]+(Lx3*M)[1:N,2:N+1,2:N+1] +(Ly3*M)[2:N+1,1:N,2:N+1]
     + (Lz3*M)[2:N+1,2:N+1,1:N] + (Lx4*M)[1:N,2:N+1,2:N+1] +(Ly4*M)[2:N+1,1:N,2:N+1]+(Lz4*M)[2:N+1,2:N+1,1:N] + (_Lx2*M)[1:N,2:N+1,2:N+1]+(_Lz3*M)[2:N+1,2:N+1,1:N]+(_Lx4*M)[1:N,2:N+1,2:N+1])

end
