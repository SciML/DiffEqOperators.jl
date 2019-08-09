struct VectorCompositeDerivativeOperator{T, N, S, O} <: AbstractDiffEqCompositeOperator{T}
    ops::Array{O, N}
end
