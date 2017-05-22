include("fornberg.jl")
import Base: *

*(A::AbstractLinearOperator,x::AbstractVector) = operate(A::AbstractLinearOperator, x::AbstractVector)
Base.length(fdg::AbstractLinearOperator) = fdg.stencil_length
Base.ndims(fdg::AbstractLinearOperator) = 1
# Base.eltype{L<:AbstractLinearOperator}(::Type{L})=eltype(supertype(L))

function convolve!{T<:AbstractFloat}(x_temp::Vector{T}, x::Vector{T}, coeffs::Array{T,1}, i::Int64, mid::Int64, wndw_low::Int64, wndw_high::Int64)
    x_temp[i] = sum(coeffs[wndw_low:wndw_high] .* x[(i-(mid-wndw_low)):(i+(wndw_high-mid))])
end

function operate!{T<:AbstractFloat}(x_temp::AbstractVector{T}, fdg::AbstractLinearOperator{T}, x::AbstractVector{T})
    coeffs = fdg.stencil_coefs
    stencil_length = length(coeffs)
    mid = Int(ceil(stencil_length/2))
    boundary_point_count = stencil_length - mid
    L = length(x)

    low(i) = mid + (i-1)*(1-mid)/boundary_point_count
    high(i) = stencil_length - (stencil_length-mid)*(i-L+boundary_point_count)/(boundary_point_count)

    for i in 1 : length(x)
        wndw_low = Int(max(1, low(i)))
        wndw_high = Int(min(stencil_length, high(i)))
        convolve!(x_temp, x, coeffs, i, mid, wndw_low, wndw_high)
    end
end

function operate{T<:AbstractFloat}(fdg::AbstractLinearOperator{T}, x::AbstractVector{T})
    x_temp = similar(x)
    operate!(x_temp, fdg, x)
    return x_temp
end