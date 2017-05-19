__precompile__()

module PDEOperators

export PDEOperator
include("operator.jl")

export operate, operate!
export AbstractLinearOperator, LinearOperator
end # module
