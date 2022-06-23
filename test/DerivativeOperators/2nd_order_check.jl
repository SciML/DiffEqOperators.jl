using DiffEqOperators, Test, LinearAlgebra

@testset "Second Order Value Check" begin
    ord = 2               # Order of approximation
    h = 0.025π
    N = Int(2 * (π / h))
    x = (-π):h:(π - h)
    L1 = CenteredDifference(1, ord, h, N)
    L2 = CenteredDifference(2, ord, h, N)

    Q = PeriodicBC(Float64)

    u0 = cos.(x)
    du_true = -cos.(x)

    # Explicit stencil for L2
    M = Matrix(Tridiagonal([1.0 for i in 1:(N - 1)],
                           [-2.0 for i in 1:N],
                           [1.0 for i in 1:(N - 1)]))

    # Insert periodic BC
    M[end, 1] = 1.0
    M[1, end] = 1.0
    M = M / (h^2)
    @test M * u0 ≈ L2 * Q * u0

    A = zeros(size(M))
    for i in 1:N, j in 1:N
        i == j && (A[i, j] = -0.5)
        abs(i - j) == 2 && (A[i, j] = 0.25)
    end
    A[end - 1, 1] = 0.25
    A[end, 2] = 0.25
    A[1, end - 1] = 0.25
    A[2, end] = 0.25
    A = A / (h^2)

    @test A * u0 ≈ L1 * Q * (L1 * Q * u0)

    κ(x) = 1.0
    tmp1 = similar(x)
    tmp2 = similar(x)
    function f(t, u, du)
        mul!(tmp1, L1, Q * u)
        @. tmp2 = κ(x) * tmp1
        mul!(du, L1, Q * tmp2)
    end

    du = similar(u0)
    f(0, u0, du)
    du ≈ A * u0

    error = du_true - du
    du2 = L2 * Q * u0
    error2 = du_true - du2

    @test maximum(error) < 0.004
    @test maximum(error2) < 0.001
end
