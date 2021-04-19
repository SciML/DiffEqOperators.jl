# The object type for Vector calculus operators is CompositeVectorDerivativeOperator. This stores
# all the operators required during computations.

struct CompositeVectorDerivativeOperator{T,N,O} <:AbstractDiffEqCompositeOperator{T}
    ops :: O
end

# Operator for calculating gradients of a N-dimensional function

struct GradientOperator end

function GradientOperator(approximation_order::Int, Δ::Union{NTuple{N,AbstractVector{T}},NTuple{N,T}},
        s::NTuple{N,Int}, coeff_func=nothing) where {T<:Real, N }
    
    ops = permutedims([CenteredDifference{n}(1, approximation_order, Δ[n], s[n], coeff_func) for n in 1:N])

    CompositeVectorDerivativeOperator{T,N,typeof(ops)}(
        ops
        )
end

function *(A::CompositeVectorDerivativeOperator{T},M::AbstractArray{T,N}) where {T<:Real,N}
    
    size_x_temp = [size(M)...].-2

    x_temp = Array{Array{T,1},length(A.ops)}(undef,size_x_temp...)
    
    for I in CartesianIndices(x_temp)
        x_temp[I] = zeros(T,length(A.ops))
    end

    for L in A.ops
        mul!(x_temp, L, M)
    end

    return x_temp
end