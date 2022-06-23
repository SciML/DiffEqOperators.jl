using DiffEqOperators, Test

@testset "Dot, Cross and L2-Norm of vectors" begin
    s = x, y, z = (-5:1.25:5, -5:1.25:5, -5:1.25:5)
    dx = dy = dz = x[2] - x[1]

    # Vector u0 = (x^2) ê₁ + (y^2) ê₂  + (z^2) ê₃
    # Vector u1 = (x) ê₁ + (y) ê₂  + (z) ê₃

    u0 = zeros(Float64, length(x), length(y), length(z), 3)

    for k = 1:length(z), j = 1:length(y), i = 1:length(x)
        u0[i, j, k, 1] = x[i]^2
        u0[i, j, k, 2] = y[j]^2
        u0[i, j, k, 3] = z[k]^2
    end

    u1 = zeros(Float64, length(x), length(y), length(z), 3)
    for k = 1:length(z), j = 1:length(y), i = 1:length(x)
        u1[i, j, k, 1] = x[i]
        u1[i, j, k, 2] = y[j]
        u1[i, j, k, 3] = z[k]
    end

    # Analytic dot u0 ⋅ u1 is given by u_analytic = x^3 + y^3 + z^3

    u_analytic = zeros(Float64, size(u0)[1:end-1]...)
    for I in CartesianIndices(u_analytic)
        u_analytic[I] = x[I[1]]^3 + y[I[2]]^3 + z[I[3]]^3
    end

    u = zeros(Float64, size(u0)[1:end-1]...)
    D = dot_product(u0, u1)
    dot_product!(u, u0, u1)

    for I in CartesianIndices(u)
        @test u[I] ≈ u_analytic[I] atol = 1e-3
        @test D[I] ≈ u_analytic[I] atol = 1e-3
    end

    # Analytic cross u0 × u1 is given by u_analytic2 = yz(y-z)ê₁ + xz(z-x)ê₂ + xy(x-y)ê₃

    u_analytic2 = zeros(Float64, size(u0))
    for k = 1:length(z), j = 1:length(y), i = 1:length(x)
        u_analytic2[i, j, k, 1] = y[j] * z[k] * (y[j] - z[k])
        u_analytic2[i, j, k, 2] = x[i] * z[k] * (z[k] - x[i])
        u_analytic2[i, j, k, 3] = x[i] * y[j] * (x[i] - y[j])
    end

    u2 = zeros(Float64, size(u0))

    C = cross_product(u0, u1)
    cross_product!(u2, u0, u1)

    for I in CartesianIndices(u2)
        @test u2[I] ≈ u_analytic2[I] atol = 1e-3
        @test C[I] ≈ u_analytic2[I] atol = 1e-3
    end

    # Analytic L2-norm of u1 is given by u_analytic3 = (x^2 + y^2 + z^2)^(0.5)

    u_analytic3 = zeros(Float64, size(u0)[1:end-1]...)
    for I in CartesianIndices(u_analytic3)
        u_analytic3[I] = (x[I[1]]^2 + y[I[2]]^2 + z[I[3]]^2)^0.5
    end

    u3 = zeros(Float64, size(u1)[1:end-1]...)

    N = square_norm(u1)
    square_norm!(u3, u1)
    for I in CartesianIndices(u_analytic3)
        @test N[I] ≈ u_analytic3[I] atol = 1e-3
        @test u3[I] ≈ u_analytic3[I] atol = 1e-3
    end
end
