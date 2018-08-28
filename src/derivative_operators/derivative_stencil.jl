struct UniformDerivativeStencil{T,S<:SVector} <: AbstractDiffEqLinearOperator{T}
    derivative_order    :: Int
    approximation_order :: Int
    dx                  :: T
    dimension           :: Int
    stencil_length      :: Int
    stencil_coefs       :: S
    function UniformDerivativeStencil(dorder,aorder,dx::T,dim) where {T}
        stencil_length = dorder + aorder - 1 + (dorder + aorder) % 2
        stl_2 = div(stencil_length, 2)
        stencil_coefs = convert(SVector{stencil_length,T}, calculate_weights(
            dorder, zero(T), one(T) .* collect(-stl_2 : 1 : stl_2)))
        new{T,typeof(stencil_coefs)}(dorder,aorder,dx,dim,stencil_length,stencil_coefs)
    end
end

struct IrregularDerivativeStencil{T,S<:SVector} <: AbstractDiffEqLinearOperator{T}
    derivative_order    :: Int
    approximation_order :: Int
    dx                  :: Vector{T}
    dimension           :: Int
    stencil_length      :: Int
    stencil_coefs       :: Vector{S}
    function IrregularDerivativeStencil(dorder,aorder,dx::Vector{T},dim) where {T}
        stencil_length = dorder + aorder - 1 + (dorder + aorder) % 2
        stl_2 = div(stencil_length, 2)
        any(x -> x < zero(T), dx) && error("All grid steps must be greater than 0.0!")
        x = [zero(T); cumsum(dx)]
        stencil_coefs = [convert(SVector{stencil_length, T}, calculate_weights(
            dorder, zero(T), x[i-stl_2 : i+stl_2] .- x[i])) for i in stl_2+1:dim-stl_2]
        new{T,eltype(stencil_coefs)}(dorder,aorder,dx,dim,stencil_length,stencil_coefs)
    end
end

DerivativeStencil = Union{UniformDerivativeStencil, IrregularDerivativeStencil}
