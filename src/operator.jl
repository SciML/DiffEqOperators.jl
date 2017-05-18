include("fornberg.jl")


function operate{T <: AbstractFloat}(fdg::FiniteDifferenceEvenGrid{T}, x::Vector{T})
    coeffs = fdg.stencil_coefs
    stencil_length = length(coeffs)
    mid = Int(ceil(stencil_length/2))
    boundary_point_count = stencil_length - mid
    x_temp = zeros(x)
    L = length(x)

    low(i) = mid + (i-1)*(1-mid)/boundary_point_count
    high(i) = stencil_length - (stencil_length-mid)*(i-L+boundary_point_count)/(boundary_point_count)

    for i in 1 : length(x)
        wndw_low = Int(max(1, low(i)))
        wndw_high = Int(min(stencil_length, high(i)))
        x_temp[i] = sum(coeffs[wndw_low:wndw_high] .* x[(i-(mid-wndw_low)):(i+(wndw_high-mid))])
    end
    return x_temp
end
