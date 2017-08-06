using DiffEqOperators
using Base.Test
using SpecialMatrices

@testset "Dirichlet BCs" begin include("dirichlet.jl") end
@testset "Periodic BCs" begin include("periodic.jl") end
@testset "Neumann BCs" begin include("neumann.jl") end
@testset "None BCs" begin include("none.jl") end
println("skipping KdV tests for now")
# include("KdV.jl")
@testset "Heat Equation" begin include("heat_eqn.jl") end
@testset "Derivative Operators Interface" begin include("derivative_operators_interface.jl") end
