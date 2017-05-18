__precompile__()

module PDEOperators

export PDEOperator
# package code goes here
include("operator.jl")

export operate
export FiniteDifferenceEvenGrid
end # module
