# These examples are documented by the LCP notebooks from
# https://github.com/QuantEcon/SimpleDifferentialOperators.jl
# In general, these are optimal stopping problems solved
# as linear complementarity problems.

using Parameters, DiffEqOperators, Random, Suppressor, Test, PATHSolver

# Seed
Random.seed!(1793)

function LCP_split(S_)
    # setup
    u1(x) = x^2
    ρ = 0.75
    μ1(x) = -x
    σ = 0.1
    M = 300
    x = range(-5.0, 5.0, length = M)

    # construct operator and non-operator LCP objects
    BC = RobinBC((0.0, 1.0, 0.0), (0.0, 1.0, 0.0), x[2] - x[1], 1)
    L1 = UpwindDifference(1, 1, x[2] - x[1], M, map(t -> μ1(t), x)) * BC
    L2 = σ^2 / 2 * CenteredDifference(2, 2, x[2] - x[1], M) * BC
    u_vec = u1.(x)

    return (L1 = Array(L1)[1], L2 = Array(L2)[1], S = S_ .* ones(M), u = u_vec, ρ = ρ)
end

@testset "Split Case, Nonzero S" begin
    # setup
    @unpack L1, L2, S, u, ρ = LCP_split(0.125)
    g = z -> ρ .* z - L1 * z - L2 * z -
             u + ρ .* S - L1 * S - L2 * S

    n = 300
    lb = zeros(n)
    ub = 300.0 .* ones(n) # a reasonable guess?
    options(convergence_tolerance = 1e-14, output = :no,
            time_limit = 600) # 10 minute budget
    exit_code, sol_z, sol_g = @suppress solveLCP(g, lb, ub)
    # tests
    @test exit_code == :StationaryPointFound
    @test sol_z[3]≈8.774937148286833 atol=1e-5
    @test sol_z[120]≈0.3021168981772892 atol=1e-5
    @test sol_z[270]≈5.729541001488482 atol=1e-5
    @test sol_g[end - 1]≈3.197442310920451e-14 atol=1e-5
end

@testset "Split Case, Zero S" begin
    # setup
    @unpack L1, L2, S, u, ρ = LCP_split(0.0)
    f = z -> ρ .* z - L1 * z - L2 * z -
             u + ρ .* S - L1 * S - L2 * S

    # @unpack L, B, S, q = LCP_split(0.)
    # f = z -> B*z + q
    n = 300
    lb = zeros(n)
    ub = 300 * ones(n) # a reasonable guess?
    options(convergence_tolerance = 1e-12, output = :no,
            time_limit = 600) # 10 minute budget
    exit_code, sol_z, sol_f = @suppress solveLCP(f, lb, ub)

    # tests
    @test exit_code == :StationaryPointFound
    @test sol_z[3]≈8.888461348456772 atol=1e-5
    @test sol_z[76]≈2.279767804635279 atol=1e-5
    @test sol_z[150]≈0.005770703117189083 atol=1e-5
    @test sol_z[269]≈5.744079221450249 atol=1e-5
    @test sol_f[123]≈1.7763568394002505e-15 atol=1e-5
end

nothing
