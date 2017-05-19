include("fornberg.jl")


function convolve!{T}(x_temp::SharedArray{T}, x::Vector{T}, coeffs::Array{T,1}, i::Int64, mid::Int64, wndw_low::Int64, wndw_high::Int64)
    wndw_low = Int(max(1, low(i)))
    wndw_high = Int(min(stencil_length, high(i)))
    x_temp[i] = sum(coeffs[wndw_low:wndw_high] .* x[(i-(mid-wndw_low)):(i+(wndw_high-mid))])
end

function operate{T <: AbstractFloat}(fdg::FiniteDifferenceEvenGrid{T}, x::Vector{T})
    coeffs = fdg.stencil_coefs
    stencil_length = length(coeffs)
    mid = Int(ceil(stencil_length/2))
    boundary_point_count = stencil_length - mid
    x_temp = SharedArray(T, length(x))
    L = length(x)

    low(i) = mid + (i-1)*(1-mid)/boundary_point_count
    high(i) = stencil_length - (stencil_length-mid)*(i-L+boundary_point_count)/(boundary_point_count)

    @parallel for i in 1 : length(x)
        wndw_low = Int(max(1, low(i)))
        wndw_high = Int(min(stencil_length, high(i)))
        convolve!(x_temp, x, coeffs, i, mid, wndw_low, wndw_high)
    end
    return x_temp
end
