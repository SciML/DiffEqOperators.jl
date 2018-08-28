struct UniformDerivativeStencil{T,S<:SVector} <: AbstractDiffEqLinearOperator{T}
    derivative_order    :: Int
    approximation_order :: Int
    dimension           :: Tuple{Int,Int}
    stencil_length      :: Int
    stencil_coefs       :: S
    function UniformDerivativeStencil(dorder,aorder,dx::T,dim_extended) where {T}
        stencil_length = dorder + aorder - 1 + (dorder + aorder) % 2
        stl_2 = div(stencil_length, 2)
        dim = (dim_extended - stencil_length + 1, dim_extended)
        stencil_coefs = convert(SVector{stencil_length,T}, calculate_weights(
            dorder, zero(T), dx .* collect(-stl_2 : 1 : stl_2)))
        new{T,typeof(stencil_coefs)}(dorder,aorder,dim,stencil_length,stencil_coefs)
    end
    UniformDerivativeStencil(xgrid::AbstractRange{T},dorder,aorder) where {T} =
        UniformDerivativeStencil(dorder,aorder,step(xgrid),length(xgrid))
end
function mul!(y::AbstractVector{T}, L::UniformDerivativeStencil{T,S}, x::AbstractVector{T}) where {T,S}
    coeffs = L.stencil_coefs
    Threads.@threads for i in 1:length(y)
        ytemp = zero(T)
        @inbounds for idx in 1:L.stencil_length
            ytemp += coeffs[idx] * x[i + idx - 1]
        end
        y[i] = ytemp
    end
    return y
end
function convert(::Type{AbstractMatrix}, L::UniformDerivativeStencil)
    coeffs = L.stencil_coefs
    mat = spzeros(eltype(L), size(L,1), size(L,2))
    for i in 1:size(L,1)
        for idx in 1:L.stencil_length
            mat[i, i + idx - 1] = coeffs[idx]
        end
    end
    return mat
end

struct IrregularDerivativeStencil{T,S<:SVector} <: AbstractDiffEqLinearOperator{T}
    derivative_order    :: Int
    approximation_order :: Int
    dimension           :: Tuple{Int,Int}
    stencil_length      :: Int
    stencil_coefs       :: Vector{S}
    function IrregularDerivativeStencil(xgrid::Vector{T},dorder,aorder) where {T}
        dim_extended = length(xgrid)
        stencil_length = dorder + aorder - 1 + (dorder + aorder) % 2
        stl_2 = div(stencil_length, 2)
        dim = (dim_extended - stencil_length + 1, dim_extended)
        stencil_coefs = [convert(SVector{stencil_length, T}, calculate_weights(
            dorder, zero(T), xgrid[i-stl_2 : i+stl_2] .- xgrid[i])) for i in stl_2+1:dim_extended-stl_2]
        new{T,eltype(stencil_coefs)}(dorder,aorder,dim,stencil_length,stencil_coefs)
    end
end
function mul!(y::AbstractVector{T}, L::IrregularDerivativeStencil{T,S}, x::AbstractVector{T}) where {T,S}
    coeffs = L.stencil_coefs
    Threads.@threads for i in 1:length(y)
        ytemp = zero(T)
        @inbounds for idx in 1:L.stencil_length
            ytemp += coeffs[i][idx] * x[i + idx - 1]
        end
        y[i] = ytemp
    end
    return y
end
function convert(::Type{AbstractMatrix}, L::IrregularDerivativeStencil)
    coeffs = L.stencil_coefs
    mat = spzeros(eltype(L), size(L,1), size(L,2))
    for i in 1:size(L,1)
        for idx in 1:L.stencil_length
            mat[i, i + idx - 1] = coeffs[i][idx]
        end
    end
    return mat
end

DerivativeStencil = Union{UniformDerivativeStencil, IrregularDerivativeStencil}
size(L::DerivativeStencil) = L.dimension
size(L::DerivativeStencil, i::Int) = i <= 2 ? L.dimension[i] : 1
function *(L::DerivativeStencil, x::AbstractVector)
    y = zeros(promote_type(eltype(L), eltype(x)), size(L, 1))
    mul!(y, L, x)
end
