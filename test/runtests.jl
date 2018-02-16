using DiffEqOperators
using Base.Test
using SpecialMatrices

tic()
@time @testset "Derivative Operators Interface" begin include("derivative_operators_interface.jl") end
@time @testset "Dirichlet BCs" begin include("dirichlet.jl") end
@time @testset "Periodic BCs" begin include("periodic.jl") end
@time @testset "Neumann BCs" begin include("neumann.jl") end
@time @testset "2nd order check" begin include("2nd_order_check.jl") end
@time @testset "None BCs" begin include("none.jl") end
#@time @testset "KdV" begin include("KdV.jl") end # KdV times out
@time @testset "Heat Equation" begin include("heat_eqn.jl") end
toc()
