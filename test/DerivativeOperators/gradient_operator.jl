using DiffEqOperators, Test

@testset "Gradient Operation on a 3-dimensional function" begin

    s = x, y, z = (-5:1.25:5, -5:1.25:5, -5:1.25:5)
    dx = dy = dz = x[2] - x[1]

    f(x::T, y::T, z::T) where T = x^2 + y^2 + z^2

    u0 = [f(X, Y, Z) for X in x, Y in y, Z in z]

    # Analytic Gradient of the function is given by u_analytic = 2x ê₁ + 2y ê₂  + 2z ê₃

    u_analytic = Array{Array{Float64,1},3}(undef,size(u0).-2)

    for I in CartesianIndices(u_analytic)
        u_analytic[I] = zeros(Float64,3)
        u_analytic[I][1] = 2*x[I[1]+1]
        u_analytic[I][2] = 2*y[I[2]+1]
        u_analytic[I][3] = 2*z[I[3]+1]
    end

    A = Gradient(4,(dx,dy,dz),size(u0).-2)

    u = A*u0

    for I in CartesianIndices(u)
        @test u[I] ≈ u_analytic[I] atol=1e-3
    end

    # check for multiplication with constant

    u1 = 2*A*u0
    
    for I in CartesianIndices(u)
        @test u1[I] ≈ 2*u[I] atol=1e-3
    end
    
    # check for non-uniform grid
    dx = dy = dz = 1.25*ones(10)

    A = Gradient(4,(dx,dy,dz),size(u0).-2)

    u = A*u0

    for I in CartesianIndices(u)
        @test u[I] ≈ u_analytic[I] atol=1e-3
    end
end