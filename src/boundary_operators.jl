export dirichlet_0!, periodic!, low, high

#= Worker functions=#
low(i::Int, mid::Int, bpc::Int) = Int(mid + (i-1)*(1-mid)/bpc)
high(i::Int, mid::Int, bpc::Int, slen::Int, L::Int) = Int(slen - (slen-mid)*(i-L+bpc)/(bpc))

function rem1(x,y)
    r = x%y
    if r > 0
        return r
    else
        return r+y
    end
end


function dirichlet_0!{T<:Real}(x_temp::AbstractVector{T}, x::AbstractVector{T}, coeffs::SVector, i::Int)
    #=
        The high and low functions determine the starting and ending indices of the weight vector.
        As we move along the input vector to calculate the derivative at the point, the weights which
        are to be considered to calculate the derivative are to be chosen carefully. eg. at the boundaries,
        only half of the stencil is going to be used to calculate the derivative at that point.
        So, observing that the left index grows as:-
                  i ^
                    |       mid = ceil(stencil_length/2)
               mid--|       bpc = boundary_point_count
                    |\
                    | \
               0  <_|__\________>
                    |  bpc      i

        And the right index grows as:-
                  i ^       mid = ceil(stencil_length/2)
                    |       bpc = boundary_point_count
               mid--|_______
                    |       \
                    |        \
               0  <_|_________\___>
                    |        bpc  i
        The high and low functions are basically equations of these graphs which are used to calculate
        the left and right index of the stencil as a function of the index i (where we need to find the derivative).
    =#
    stencil_length = length(coeffs)
    mid = div(stencil_length, 2) + 1
    bpc = stencil_length - mid
    L = length(x)

    wndw_low = i>bpc ? 1:max(1, low(i, mid, bpc))
    wndw_high = i>L-bpc ? min(stencil_length, high(i, mid, bpc, stencil_length, L)):stencil_length

    #=
        Here we are taking the weighted sum of a window of the input vector to calculate the derivative
        at the middle point. This requires choosing the end points carefully which are being passed from above.
    =#
    xtempi = x_temp[i]
    @inbounds for idx in wndw_low:wndw_high
        xtempi += coeffs[idx] * x[i - (mid-idx)]
    end
    x_temp[i] = xtempi
end


function periodic!{T<:Real}(x_temp::AbstractVector{T}, x::AbstractVector{T}, coeffs::SVector, i::Int)
    stencil_length = length(coeffs)
    mid = div(stencil_length, 2) + 1
    bpc = stencil_length - mid

    wndw_low = 1
    wndw_high = length(coeffs)
    L = length(x)
    #=
        Here we are taking the weighted sum of a window of the input vector to calculate the derivative
        at the middle point. Instead of breaking the stencil we loop it over from the other side if it
        doesn't fit the column of the transformation matrix to simulate the periodic boundary condition.
    =#
    xtempi = x_temp[i]
    @inbounds for idx in wndw_low:wndw_high
        xtempi += coeffs[idx] * x[rem1(i - (mid-idx), L)]
    end
    x_temp[i] = xtempi
end

function neumann!{T<:Real}(x_temp::AbstractVector{T}, x::AbstractVector{T}, coeffs::SVector, i::Int)
    stencil_length = length(coeffs)
    mid = div(stencil_length, 2) + 1
    bpc = stencil_length - mid

    wndw_low = 1
    wndw_high = length(coeffs)
    L = length(x)
    #=
        Here we are taking the weighted sum of a window of the input vector to calculate the derivative
        at the middle point. Instead of breaking the stencil we loop it over from the other side if it
        doesn't fit the column of the transformation matrix to simulate the periodic boundary condition.
    =#
    xtempi = x_temp[i]
    @inbounds for idx in wndw_low:wndw_high
        xtempi += coeffs[idx] * x[rem1(i - (mid-idx), L)]
    end
    x_temp[i] = xtempi
end
