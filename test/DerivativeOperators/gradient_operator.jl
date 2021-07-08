using DiffEqOperators, Test

@testset "Gradient Operation on a 3-dimensional function" begin

    s = x, y, z = (-5:1.25:5, -5:1.25:5, -5:1.25:5)
    dx = dy = dz = x[2] - x[1]

    f(x::T, y::T, z::T) where T = x^2 + y^2 + z^2

    u0 = [f(X, Y, Z) for X in x, Y in y, Z in z]

    # Analytic Gradient of the function is given by u_analytic = 2x ê₁ + 2y ê₂  + 2z ê₃

    u_analytic = zeros(Float64,(size(u0).-2)...,3)

    for i in 1:length(x)-2, j in 1:length(y)-2, k in 1:length(z)-2
        u_analytic[i,j,k,1] = 2*x[i+1]
        u_analytic[i,j,k,2] = 2*y[j+1]
        u_analytic[i,j,k,3] = 2*z[k+1]
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