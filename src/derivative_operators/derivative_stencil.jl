struct UniformDerivativeStencil{T,S<:SVector} <: AbstractDiffEqLinearOperator{T}
    derivative_order    :: Int
    approximation_order :: Int
    dimension           :: Int
    stencil_length      :: Int
    stencil_coefs       :: S
    function UniformDerivativeStencil(dorder,aorder,dx::T,dim) where {T}
        stencil_length = dorder + aorder - 1 + (dorder + aorder) % 2
        stl_2 = div(stencil_length, 2)
        stencil_coefs = convert(SVector{stencil_length,T}, calculate_weights(
            dorder, zero(T), dx .* collect(-stl_2 : 1 : stl_2)))
        new{T,typeof(stencil_coefs)}(dorder,aorder,dim,stencil_length,stencil_coefs)
    end
    UniformDerivativeStencil(xgrid::AbstractRange{T},dorder,aorder) where {T} =
        UniformDerivativeStencil(dorder,aorder,step(xgrid),length(xgrid))
end

struct IrregularDerivativeStencil{T,S<:SVector} <: AbstractDiffEqLinearOperator{T}
    derivative_order    :: Int
    approximation_order :: Int
    dimension           :: Int
    stencil_length      :: Int
    stencil_coefs       :: Vector{S}
    function IrregularDerivativeStencil(xgrid::Vector{T},dorder,aorder) where {T}
        dim = length(xgrid)
        stencil_length = dorder + aorder - 1 + (dorder + aorder) % 2
        stl_2 = div(stencil_length, 2)
        stencil_coefs = [convert(SVector{stencil_length, T}, calculate_weights(
            dorder, zero(T), xgrid[i-stl_2 : i+stl_2] .- xgrid[i])) for i in stl_2+1:dim-stl_2]
        new{T,eltype(stencil_coefs)}(dorder,aorder,dim,stencil_length,stencil_coefs)
    end
end

DerivativeStencil = Union{UniformDerivativeStencil, IrregularDerivativeStencil}
