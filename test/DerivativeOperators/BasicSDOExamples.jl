using LinearAlgebra, DiffEqOperators, Test, Parameters

# These tests are based on examples from SimpleDifferentialOperators.jl

#parameters
params = @with_kw (
    μ = -0.1, # constant negative drift
    σ = 0.1,
    ρ = 0.05,
    M = 3, # size of grid (interior points)
    x̄ = range(0.0, 1.0, length = (M + 2)),
    x = x̄[2:end-1],
    S = 3.0,
)
p = params()

#----------------------------------
#Testing For Negative Drifts
#----------------------------------
# payoff function
pi_profit(x) = x^2
#= SimpleDifferentialOperators setup
function SDO_negative_drift(pi_profit, params)
    bc = (Reflecting(), Reflecting())
    Lₓ = params.μ*L₁₋bc(params.x̄, bc) + params.σ^2 / 2 * L₂bc(params.x̄, bc)
    L_bc = I * params.ρ - Lₓ
    v = L_bc \ pi_profit.(params.x);
    return (v = v, L_bc = L_bc, Lₓ = Lₓ, L₁₋bc = L₁₋bc(params.x̄, bc), L₂bc = L₂bc(params.x̄, bc))
end=#
# DiffEqOperators Setup
function DEO_negative_drift(pi_profit, params)
    dx = params.x[2] - params.x[1]
    # discretize L = ρ - μ D_x - σ^2 / 2 D_xx
    # subject to reflecting barriers at 0 and 1
    L1 = UpwindDifference(1, 1, dx, params.M, 1.0)
    L2 = CenteredDifference(2, 2, dx, params.M)
    Q = Neumann0BC(dx, 1)
    L₁₋bc = -1.0 .* Array(UpwindDifference(1, 1, dx, params.M, -1.0) * Q)[1]
    # Here Array(A::GhostDerivativeOperator) will return a tuple of the linear part
    # and the affine part of the operator A, hence we index Array(µ*L1*Q).
    # The operators in this example are purely linear, so we don't worry about Array(µ*L1*Q)[2]
    Lₓ = Array(params.μ * L1 * Q)[1] + Array(params.σ^2 / 2 * L2 * Q)[1] # when scalar coefficient is properly implemented
    # Lₓ = Array(L1*Q)[1] + Array(params.σ^2/2*L2*Q)[1] # when scalar coefficient is properly implemented
    L_bc = I * params.ρ - Lₓ

    # solve the value function
    v = L_bc \ pi_profit.(params.x)
    return (v = v, L_bc = L_bc, Lₓ = Lₓ, L₁₋bc = L₁₋bc, L₂bc = Array(L2 * Q)[1])
end

@testset "Constant Negative Drifts" begin
    @test DEO_negative_drift(pi_profit, p).v ≈
          [1.9182649086005406, 2.3359304764758773, 3.1768804315253277] # SDO_negative_drift(pi_profit, p).v
    @test DEO_negative_drift(pi_profit, p).L_bc ≈
          [0.13 -0.08 0.0; -0.48 0.61 -0.08; 0.0 -0.48 0.53] # SDO_negative_drift(pi_profit, p).L_bc
    @test DEO_negative_drift(pi_profit, p).Lₓ ≈
          [-0.08 0.08 0.0; 0.48 -0.56 0.08; 0.0 0.48 -0.48] # SDO_negative_drift(pi_profit, p).Lₓ
    @test DEO_negative_drift(pi_profit, p).L₁₋bc ≈ [0.0 0.0 0.0; -4.0 4.0 0.0; 0.0 -4.0 4.0] # SDO_negative_drift(pi_profit, p).L₁₋bc
    @test DEO_negative_drift(pi_profit, p).L₂bc ≈
          [-16.0 16.0 0.0; 16.0 -32.0 16.0; 0.0 16.0 -16.0] # SDO_negative_drift(pi_profit, p).L₂bc
end
p = params(μ = 0.1);
#----------------------------------
#Testing For Positive Drifts
#----------------------------------
# SimpleDifferentialOperators setup
#=function SDO_positive_drift(pi_profit, params)
    bc = (Reflecting(), Reflecting())
    Lₓ = params.μ*L₁₊bc(params.x̄, bc) + params.σ^2 / 2 * L₂bc(params.x̄, bc)
    L_bc = I * params.ρ - Lₓ
    v = L_bc \ pi_profit.(params.x);
    return (v = v, L_bc = L_bc, Lₓ = Lₓ, L₁₊bc = L₁₊bc(params.x̄, bc), L₂bc = L₂bc(params.x̄, bc))
end=#
# DiffEqOperators Setup
function DEO_positive_drift(pi_profit, params)
    dx = params.x[2] - params.x[1]
    # discretize L = ρ - μ D_x - σ^2 / 2 D_xx
    # subject to reflecting barriers at 0 and 1
    L1 = UpwindDifference(1, 1, dx, params.M, 1.0)
    L2 = CenteredDifference(2, 2, dx, params.M)
    Q = Neumann0BC(dx, 1)
    # Here Array(A::GhostDerivativeOperator) will return a tuple of the linear part
    # and the affine part of the operator A, hence we index Array(µ*L1*Q).
    # The operators in this example are purely linear, so we don't worry about Array(µ*L1*Q)[2]
    Lₓ = Array(params.µ * L1 * Q)[1] + Array(params.σ^2 / 2 * L2 * Q)[1]
    L_bc = I * params.ρ - Lₓ

    # solve the value function
    v = L_bc \ pi_profit.(params.x)
    return (v = v, L_bc = L_bc, Lₓ = Lₓ, L₁₊bc = Array(L1 * Q)[1], L₂bc = Array(L2 * Q)[1])
end

@testset "Constant Positive Drifts" begin
    @test DEO_positive_drift(pi_profit, p).v ≈
          [8.855633802816902, 9.647887323943664, 10.264084507042257] # SDO_positive_drift(pi_profit, p).v
    @test DEO_positive_drift(pi_profit, p).L_bc ≈
          [0.53 -0.48 0.0; -0.08 0.61 -0.48; 0.0 -0.08 0.13] # SDO_positive_drift(pi_profit, p).L_bc
    @test DEO_positive_drift(pi_profit, p).Lₓ ≈
          [-0.48 0.48 0.0; 0.08 -0.56 0.48; 0.0 0.08 -0.08] # SDO_positive_drift(pi_profit, p).Lₓ
    @test DEO_positive_drift(pi_profit, p).L₁₊bc ≈ [-4.0 4.0 0.0; 0.0 -4.0 4.0; 0.0 0.0 0.0] # SDO_positive_drift(pi_profit, p).L₁₊bc
    @test DEO_positive_drift(pi_profit, p).L₂bc ≈
          [-16.0 16.0 0.0; 16.0 -32.0 16.0; 0.0 16.0 -16.0] # SDO_positive_drift(pi_profit, p).L₂bc
end


#----------------------------------
#Testing For State Dependent Drifts
#----------------------------------
μ(x) = -x
# SimpleDifferentialOperators setup
#=function SDO_state_dependent_drift(pi_profit, μ, params)
    bc = (Reflecting(), Reflecting())
    L₁ = Diagonal(min.(μ.(params.x), 0.0)) * L₁₋bc(params.x̄, bc) + Diagonal(max.(μ.(params.x), 0.0)) * L₁₊bc(params.x̄, bc)
    Lₓ = L₁ - params.σ^2 / 2 * L₂bc(params.x̄, bc)
    L_bc = I * params.ρ - Lₓ
    v = L_bc \ pi_profit.(params.x)
    return (v = v, L_bc = L_bc,  Lₓ = Lₓ, L₁ = L₁, L₂bc = L₂bc(params.x̄, bc))
end=#
# DiffEqOperators Setup
function DEO_state_dependent_drift(pi_profit, μ, params)
    dx = params.x[2] - params.x[1]
    # discretize L = ρ - μ D_x - σ^2 / 2 D_xx
    # subject to reflecting barriers at 0 and 1
    drift = μ.(params.x)
    L1 = UpwindDifference(1, 1, dx, params.M, drift)
    L2 = CenteredDifference(2, 2, dx, params.M)
    Q = Neumann0BC(dx, 1)
    # Here Array(A::GhostDerivativeOperator) will return a tuple of the linear part
    # and the affine part of the operator A, hence we index Array(µ*L1*Q).
    # The operators in this example are purely linear, so we don't worry about Array(µ*L1*Q)[2]
    L₁ = Array(L1 * Q)[1]
    Lₓ = L₁ - Array(params.σ^2 / 2 * L2 * Q)[1]
    L_bc = I * params.ρ - Lₓ

    # solve the value function
    v = L_bc \ pi_profit.(params.x)
    return (v = v, L_bc = L_bc, Lₓ = Lₓ, L₁ = L₁, L₂bc = Array(L2 * Q)[1])
end

@testset "State Dependent Drifts" begin
    @test DEO_state_dependent_drift(pi_profit, μ, p).v ≈
          [1.1027342984846056, 1.194775361931727, 1.3640552379934825] # SDO_state_dependent_drift(pi_profit, μ, p).v
    @test DEO_state_dependent_drift(pi_profit, μ, p).L_bc ≈
          [-0.03 0.08 0.0; -1.92 1.89 0.08; 0.0 -2.92 2.97] # SDO_state_dependent_drift(pi_profit, μ, p).L_bc
    @test DEO_state_dependent_drift(pi_profit, μ, p).Lₓ ≈
          [0.08 -0.08 0.0; 1.92 -1.84 -0.08; 0.0 2.92 -2.92]# SDO_state_dependent_drift(pi_profit, μ, p).Lₓ
    @test DEO_state_dependent_drift(pi_profit, μ, p).L₁ ≈
          [0.0 0.0 0.0; 2.0 -2.0 0.0; 0.0 3.0 -3.0] # SDO_state_dependent_drift(pi_profit, μ, p).L₁
    @test DEO_state_dependent_drift(pi_profit, μ, p).L₂bc ≈
          [-16.0 16.0 0.0; 16.0 -32.0 16.0; 0.0 16.0 -16.0] # SDO_state_dependent_drift(pi_profit, μ, p).L₂bc
end



#----------------------------------
#Testing For Absorbing BC
#----------------------------------


# payoff function
pi_profit(x) = x^2
# SimpleDifferentialOperators setup
#=function SDO_absorbing_bc(pi_profit, params)
    bc = (NonhomogeneousAbsorbing(params.S), Reflecting())
    Lₓbc = params.μ*L₁₋bc(params.x̄, bc) + params.σ^2 / 2 * L₂bc(params.x̄ , bc)
    L_bc = I * params.ρ - Lₓbc

    # construct the RHS with affine boundary
    pi_profit_star = pi_profit.(params.x) + L₀affine(params.x̄, pi_profit.(params.x), bc)

    # solve the value function
    v = L_bc \ pi_profit_star

    return (v = v, L_bc = L_bc, Lₓbc = Lₓbc, L₁₋bc = L₁₋bc(params.x̄, bc), L₂bc = L₂bc(params.x̄, bc),
        pi_profit_star = pi_profit_star, pi_profit = pi_profit.(params.x))
end=#

# DiffEqOperators Setup
function DEO_absorbing_bc(pi_profit, params)
    dx = params.x[2] - params.x[1]

    L1 = UpwindDifference(1, 1, dx, params.M, params.μ)
    L2 = CenteredDifference(2, 2, dx, params.M)
    # RobinBC(l::NTuple{3,T}, r::NTuple{3,T}, dx::T, order = 1)
    # The variables in l are [αl, βl, γl], and correspond to a BC of the form αl*u(0) + βl*u'(0) = γl
    # imposed on the lower index boundary. The variables in r are [αr, βr, γr],
    # and correspond to an analagous boundary on the higher index end.
    l = (1.0, 0.0, p.S)
    r = (0.0, 1.0, 0.0)
    Q = RobinBC(l, r, dx, 1)

    Lₓbc = Array(L1 * Q)[1] + Array(params.σ^2 / 2 * L2 * Q)[1]
    L_bc = I * params.ρ - Lₓbc

    # solve the value function
    v = L_bc \ pi_profit.(params.x)
    return (
        v = v,
        L_bc = L_bc,
        Lₓbc = Lₓbc,
        L₁₋bc = Array(L1 * Q)[1] ./ params.μ,
        L₂bc = Array(L2 * Q)[1],
        pi_profit_star = pi_profit.(params.x),
        pi_profit = pi_profit.(params.x),
    )
end

p = params(x̄ = range(0.0, 1.0, length = (p.M + 2)))

@testset "Absorbing BC" begin
    @test DEO_absorbing_bc(pi_profit, p).v ≈
          [0.2085953844248779, 0.8092898062396943, 1.7942624660284021] # SDO_absorbing_bc(pi_profit, p).v
    @test DEO_absorbing_bc(pi_profit, p).L_bc ≈
          [0.61 -0.08 0.0; -0.48 0.61 -0.08; 0.0 -0.48 0.53] # SDO_absorbing_bc(pi_profit, p).L_bc
    @test DEO_absorbing_bc(pi_profit, p).Lₓbc ≈
          [-0.56 0.08 0.0; 0.48 -0.56 0.08; 0.0 0.48 -0.48] # SDO_absorbing_bc(pi_profit, p).Lₓbc
    @test DEO_absorbing_bc(pi_profit, p).L₁₋bc ≈ [4.0 0.0 0.0; -4.0 4.0 0.0; 0.0 -4.0 4.0] # SDO_absorbing_bc(pi_profit, p).L₁₋bc
    @test DEO_absorbing_bc(pi_profit, p).L₂bc ≈
          [-32.0 16.0 0.0; 16.0 -32.0 16.0; 0.0 16.0 -16.0] # SDO_absorbing_bc(pi_profit, p).L₂bc
    @test DEO_absorbing_bc(pi_profit, p).pi_profit_star ≈ [0.0625, 0.25, 0.5625] # SDO_absorbing_bc(pi_profit, p).pi_profit_star
    @test DEO_absorbing_bc(pi_profit, p).pi_profit ≈ [0.0625, 0.25, 0.5625] # SDO_absorbing_bc(pi_profit, p).pi_profit
end



#----------------------------------
#Testing For Solving KFE
#----------------------------------


#=function SDO_Solve_KFE(params)
    # ξ values for mixed boundary conditions
    ξ_lb = ξ_ub = -2 * params.μ / params.σ^2
    # define the corresponding mixed boundary conditions
    # note that the direction on the lower bound is backward (default is forward)
    # as the drift μ is negative.
    bc = (Mixed(ξ = ξ_lb, direction = :backward), Mixed(ξ = ξ_ub))

    # use SimpleDifferentialOperators.jl to construct the operator on the interior
    L_KFE_with_drift = Array(-params.μ*L₁₊bc(params.x̄, bc) + params.σ^2 / 2 * L₂bc(params.x̄, bc))
    L_KFE_without = params.σ^2 / 2 * L₂bc(params.x̄, bc)
    return (L_KFE_with_drift = L_KFE_with_drift, L_KFE_without = L_KFE_without)
end=#

#=function SDO_Solve_KFE_forward(params)
    # ξ values for mixed boundary conditions
    ξ_lb = ξ_ub = -2 * params.μ / params.σ^2
    # define the corresponding mixed boundary conditions
    # but with forward finite difference derivatives at boundaries
    bc = (Mixed(ξ = ξ_lb, direction = :forward), Mixed(ξ = ξ_ub))

    # use SimpleDifferentialOperators.jl to construct the operator on the interior
    L_KFE_with_drift = Array(-params.μ*L₁₊bc(params.x̄, bc) + params.σ^2 / 2 * L₂bc(params.x̄, bc))
    L_KFE_without = params.σ^2 / 2 * L₂bc(params.x̄, bc)
    return (L_KFE_with_drift = L_KFE_with_drift, L_KFE_without = L_KFE_without)
end=#


function DEO_Solve_KFE(params)
    dx = params.x[2] - params.x[1]

    L1 = UpwindDifference(1, 1, dx, params.M, -params.μ)
    # L2l = UpwindDifference(2,2,dx,params.M,
    #                          vcat(-1.,zeros(params.M-1)))
    # L2r = UpwindDifference(2,2,dx,params.M,
    #                          vcat(zeros(params.M-1),1.))
    L2 = CenteredDifference(2, 2, dx, params.M)

    ξ_lb = ξ_ub = -2 * params.μ / params.σ^2

    l = (ξ_lb, 1.0, 0.0)
    r = (ξ_ub, 1.0, 0.0)
    Q = RobinBC(l, r, dx, 1)

    # use SimpleDifferentialOperators.jl to construct the operator on the interior
    # Only difference is handling of L2, we may need to upwind for boundary
    L_KFE_with_drift = Array(L1 * Q)[1] + Array(params.σ^2 / 2 * L2 * Q)[1]
    L_KFE_without = (params.σ^2 / 2) .* Array(L2 * Q)[1]
    # L_KFE_without = (params.σ^2 / 2) .* ([0., 1., 0.] .* Array(L2 * Q)[1] +
    #     -(Array(L2l * Q)) + Array(L2r * Q))

    return (L_KFE_with_drift = L_KFE_with_drift, L_KFE_without = L_KFE_without)
end

p = params(x̄ = range(0.0, 1.0, length = (p.M + 2)))
@testset "Solving KFE" begin
    # @test_broken DEO_Solve_KFE(p).L_KFE_with_drift ≈ SDO_Solve_KFE(p).L_KFE_with_drift # concretization of Robin conditions appears broken
    # @test_broken DEO_Solve_KFE(p).L_KFE_without ≈ SDO_Solve_KFE(p).L_KFE_without # Hard to check what's wrong since KFE uses forward/backward at boundaries rather than central differences when w/mixed bcs
    @test DEO_Solve_KFE(p).L_KFE_with_drift ≈
          [-0.58 0.48 0.0; 0.08 -0.56 0.48; 0.0 0.08 -0.48] # SDO_Solve_KFE_forward(p).L_KFE_with_drift
    @test DEO_Solve_KFE(p).L_KFE_without ≈
          [-0.18 0.08 0.0; 0.08 -0.16 0.08; 0.0 0.08 -0.1466666666666667] # SDO_Solve_KFE_forward(p).L_KFE_without
end

nothing
