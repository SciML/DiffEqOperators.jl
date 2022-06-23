using DiffEqOperators, Test

@testset "Gradient Operation on a 2-dimensional function" begin

    # For testing the faster implementation for a 2-dim function
    s = x, y = (-5:1.25:5, -5:1.25:5)
    dx = dy = x[2] - x[1]

    f(x::T, y::T) where {T} = x^2 + y^2

    u0 = [f(X, Y) for X in x, Y in y]

    # Analytic Gradient of the function is given by u_analytic = 2x ê₁ + 2y ê₂

    u_analytic = zeros(Float64, (size(u0) .- 2)..., 2)

    for j in 1:(length(y) - 2), i in 1:(length(x) - 2)
        u_analytic[i, j, 1] = 2 * x[i + 1]
        u_analytic[i, j, 2] = 2 * y[j + 1]
    end

    A = Gradient(4, (dx, dy), size(u0) .- 2)

    u = A * u0

    for I in CartesianIndices(u)
        @test u[I]≈u_analytic[I] atol=1e-3
    end

    # check for non-uniform grid
    dx = dy = 1.25 * ones(length(s[1]) - 1)

    A = Gradient(4, (dx, dy), size(u0) .- 2)

    u = A * u0

    for I in CartesianIndices(u)
        @test u[I]≈u_analytic[I] atol=1e-3
    end
end

@testset "Gradient Operation on a 3-dimensional function" begin

    # For testing the faster implementation for a 3-dim function
    s = x, y, z = (-5:1.25:5, -5:1.25:5, -5:1.25:5)
    dx = dy = dz = x[2] - x[1]

    f(x::T, y::T, z::T) where {T} = x^2 + y^2 + z^2

    u0 = [f(X, Y, Z) for X in x, Y in y, Z in z]

    # Analytic Gradient of the function is given by u_analytic = 2x ê₁ + 2y ê₂  + 2z ê₃

    u_analytic = zeros(Float64, (size(u0) .- 2)..., 3)

    for k in 1:(length(z) - 2), j in 1:(length(y) - 2), i in 1:(length(x) - 2)
        u_analytic[i, j, k, 1] = 2 * x[i + 1]
        u_analytic[i, j, k, 2] = 2 * y[j + 1]
        u_analytic[i, j, k, 3] = 2 * z[k + 1]
    end

    A = Gradient(4, (dx, dy, dz), size(u0) .- 2)

    u = A * u0

    for I in CartesianIndices(u)
        @test u[I]≈u_analytic[I] atol=1e-3
    end

    # check for multiplication with constant

    u1 = 2 * A * u0

    for I in CartesianIndices(u)
        @test u1[I]≈2 * u[I] atol=1e-3
    end

    # check for non-uniform grid
    dx = dy = dz = 1.25 * ones(length(s[1]) - 1)

    A = Gradient(4, (dx, dy, dz), size(u0) .- 2)

    u = A * u0

    for I in CartesianIndices(u)
        @test u[I]≈u_analytic[I] atol=1e-3
    end
end

@testset "Gradient Operation on a 4-dimensional function" begin

    # For Testing the fallback implementation

    s = x, y, z, w = (-5:1.25:5, -5:1.25:5, -5:1.25:5, -5:1.25:5)
    dx = dy = dz = dw = x[2] - x[1]

    f(x::T, y::T, z::T, w::T) where {T} = x^2 + y^2 + z^2 + w^2

    u0 = [f(X, Y, Z, W) for X in x, Y in y, Z in z, W in w]

    # Analytic Gradient of the function is given by u_analytic = 2x ê₁ + 2y ê₂  + 2z ê₃ + 2w ê₄

    u_analytic = zeros(Float64, (size(u0) .- 2)..., 4)

    for l in 1:(length(w) - 2), k in 1:(length(z) - 2), j in 1:(length(y) - 2),
        i in 1:(length(x) - 2)

        u_analytic[i, j, k, l, 1] = 2 * x[i + 1]
        u_analytic[i, j, k, l, 2] = 2 * y[j + 1]
        u_analytic[i, j, k, l, 3] = 2 * z[k + 1]
        u_analytic[i, j, k, l, 4] = 2 * w[l + 1]
    end

    A = Gradient(4, (dx, dy, dz, dw), size(u0) .- 2)

    u = A * u0

    for I in CartesianIndices(u)
        @test u[I]≈u_analytic[I] atol=1e-3
    end
end
