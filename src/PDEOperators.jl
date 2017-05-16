__precompile__()

module PDEOperators

export PDEOperator
# package code goes here
include("fornberg.jl")
abstract FiniteDifferenceOperator

PDEOperator(args...) = FiniteDifferenceEvenGrid{Float64}(args...)

end # module
