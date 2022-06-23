using Test, DiffEqOperators

# Set up coefficient functions to test with
vec_fcn = Vector{Function}(undef, 0)
f1(x::Float64) = sin(x)
f2(x::Vector{Float64}) = sin.(x)
push!(vec_fcn, f1)
push!(vec_fcn, f2)

@testset "Test coefficient functions when current_coeffs exists" begin

    vec_fcn_ans = Vector{Vector{Float64}}(undef, 0)
    for i = 1:2
        push!(vec_fcn_ans, [sin(1.1), sin(2.3), sin(4.5)])
    end

    current_coeffs = [1.1, 2.3, 4.5]

    # Test inplace versions
    current_coeffs1 = Vector{Float64}(undef, 3)
    @test DiffEqOperators.compute_coeffs!(2.54, current_coeffs1) ≈ 2.54 .* ones(3)
    current_coeffs1 = Vector{Float64}(undef, 3)
    @test DiffEqOperators.compute_coeffs!([1.1, 2.3, 4.5], current_coeffs1) ≈ current_coeffs
    @test DiffEqOperators.compute_coeffs!(2.54, current_coeffs1) ≈ [3.64, 4.84, 7.04]
    @test all(current_coeffs .!= current_coeffs1)
    current_coeffs1 = copy(current_coeffs)
    @test DiffEqOperators.compute_coeffs!([2.54, 1.03, 0.18], current_coeffs1) ≈
          [3.64, 3.33, 4.68]
    @test all(current_coeffs .!= current_coeffs1)
    for (fcn, fcn_ans) in zip(vec_fcn, vec_fcn_ans)
        current_coeffs1 = copy(current_coeffs)
        @test DiffEqOperators.compute_coeffs!(fcn, current_coeffs1) ≈ fcn_ans
        @test all(current_coeffs1 .!= current_coeffs)
    end

    # Test not in place versions
    current_coeffs1 = Vector{Float64}(undef, 3)
    @test DiffEqOperators.compute_coeffs(2.54, current_coeffs) ≈ [3.64, 4.84, 7.04]
    @test DiffEqOperators.compute_coeffs([2.54, 1.03, 0.18], current_coeffs) ≈
          [3.64, 3.33, 4.68]
    for (fcn, fcn_ans) in zip(vec_fcn, vec_fcn_ans)
        @test DiffEqOperators.compute_coeffs(fcn, current_coeffs) ≈ fcn_ans
    end
end

@testset "Check coefficients of DerivativeOperators are properly computed" begin

    # Set up operators to test construction with different coeff_func
    vec_op = Vector{DerivativeOperator}(undef, 0)
    dxv = [1.4, 1.1, 2.3, 3.6, 1.5]
    push!(vec_op, UpwindDifference(1, 1, 1.0, 3, 1.0))
    push!(vec_op, UpwindDifference(1, 1, 1.0, 3, [1.0, 2.25, 3.5]))
    push!(vec_op, UpwindDifference(1, 1, 1.0, 3, cos))
    push!(vec_op, UpwindDifference(1, 1, dxv, 3, 1.0))
    push!(vec_op, UpwindDifference(1, 1, dxv, 3, [1.0, 2.25, 3.5]))
    push!(vec_op, UpwindDifference(1, 1, dxv, 3, cos))


    vec_op_ans = Vector{Any}(undef, 0)
    push!(vec_op_ans, ones(3))
    push!(vec_op_ans, [1.0, 2.25, 3.5])
    push!(vec_op_ans, ones(3))
    push!(vec_op_ans, ones(3))
    push!(vec_op_ans, [1.0, 2.25, 3.5])
    push!(vec_op_ans, ones(3))

    # Check constructors
    @test UpwindDifference(1, 1, 1.0, 3, 0.0).coefficients + ones(3) - ones(3) ≈ zeros(3)
    @test CenteredDifference(2, 2, 1.0, 3, 0.0).coefficients + ones(3) - ones(3) ≈ zeros(3)
    for (L, op_ans) in zip(vec_op, vec_op_ans)
        @test L.coefficients ≈ op_ans
    end
    push!(vec_op, CenteredDifference(2, 2, 1.0, 3)) == nothing

    # Compute answers to coeff_func * operator
    func_mul_op_ans1 = Vector{Any}(undef, 0)
    for i = 1:2
        push!(func_mul_op_ans1, zeros(3))
    end
    func_mul_op_ans2 = Vector{Any}(undef, 0)
    for i = 1:2
        push!(func_mul_op_ans2, sin(1) .* ones(3))
    end
    func_mul_op_ans3 = Vector{Any}(undef, 0)
    for i = 1:2
        push!(func_mul_op_ans3, [0.0, sin(1.5), 0.0])
    end
end

nothing
