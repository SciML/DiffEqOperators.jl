# These examples are documented by the LCP notebooks from
# https://github.com/QuantEcon/SimpleDifferentialOperators.jl
# In general, these are optimal stopping problems solved
# as linear complementarity problems.

using Parameters, DiffEqOperators, Random, Suppressor, Test, PATHSolver

# Seed
Random.seed!(1793)

# Setup
StoppingProblem = @with_kw (μ_bar = -0.01, # 1D Brownian motion with drift
                            σ_bar = 0.01,
                            S_bar = 10.0,
                            γ = 0.5, # u(x) = x^γ
                            ρ = 0.05, # discount rate
                            x_min = 0.01,
                            x_max = 5.0,
                            M = 15) # num of grid points

function LCP_objects(sp)
    # setup
    @unpack μ_bar, σ_bar, S_bar, γ, ρ, x_min, x_max, M = sp
    grid = range(x_min, x_max, length = M)
    grid_extended = [grid[1] - diff(grid)[1]; grid; grid[end] + diff(grid)[end]]
    μ = μ_bar
    S(x) = S_bar
    u(x) = x^γ
    σ = σ_bar

    # construct operator and non-operator LCP objects
    BC1 = RobinBC((0., 1., 0.), (0., 1., 0.), grid[2] - grid[1], 1) # for first-derivative
    BC2 = RobinBC((0., 1., 0.), (0., 1., 0.), grid[2] - grid[1], 1) # for second-derivative
    L1 = UpwindDifference(1, 1, grid[2] - grid[1], M, 0, μ) * BC1
    L2 = σ^2/2 * CenteredDifference(2, 2, grid[2] - grid[1], M) * BC2
    S_vec = S.(grid)
    u_vec = u.(grid)

    return (L1 = Array(L1)[1], L2 = Array(L2)[1], S = S_vec, u = u_vec, ρ = ρ)
end


function LCPsolve(sp)
    @unpack L1, L2, S, u, ρ = LCP_objects(sp)
    f = z -> ρ .* z - L1 * z - L2 * z -
        u + ρ .* S - L1 * S - L2 * S
    n = sp.M
    lb = zeros(n)
    ub = 300 .* ones(n) # a reasonable guess
    options(convergence_tolerance = 1e-15, output = :no,
            time_limit = 600) # 10 minute budget
    exit_code, sol_z, sol_f = @suppress solveLCP(f, lb, ub)
end

@testset "Backward Case, Nonzero S" begin
  sp = StoppingProblem()
  code, sol, f = LCPsolve(sp)
  @test code == :Solved
  @test sol[1] ≈ 0.0 atol = 1e-5
  @test sol[5] ≈ 12.066957816758809 atol = 1e-5
  @test f[1] ≈ 0.39946441597137766 atol = 1e-5
end

@testset "Backward Case, Zero S" begin
  code, sol, f = LCPsolve(StoppingProblem(S_bar = 0.))
  @test code == :Solved
  @test sol[1] ≈ 2.050665004133949 atol = 1e-5
  @test sol[8] ≈ 30.258918534086924 atol = 1e-5
  @test f[1] ≈ 0.0 atol = 1e-2
end



nothing
