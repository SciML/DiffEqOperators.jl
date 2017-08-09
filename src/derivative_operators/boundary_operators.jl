#= Worker functions=#
low(i::Int, mid::Int, bpc::Int) = Int(mid + (i-1)*(1-mid)/bpc)
high(i::Int, mid::Int, bpc::Int, slen::Int, L::Int) = Int(slen - (slen-mid)*(i-L+bpc)/(bpc))
limit(i, N) = N>=i>=1 ? i : (i<1 ? 1 : N)

function reflect(idx, L)
    abs1 = abs(L-idx)
    if L - abs1 > 0
        return L-abs1
    else
        return abs(L-abs1)+2
    end
end

# gives the index for periodic BC
function rem1(idx,L)
    r = idx%L
    if r > 0
        return r
    else
        return r+L
    end
end


#= LEFT BOUNDARY CONDITIONS =#
function convolve_BC_left!{T<:Real,S<:SVector,RBC}(x_temp::AbstractVector{T}, x::AbstractVector{T}, A::DerivativeOperator{T,S,:Dirichlet0,RBC})
    mid = div(A.stencil_length,2) + 1
    bpc = A.stencil_length - mid
    x[1] = zero(T)

    for i in 1 : A.boundary_point_count[1]
        dirichlet_0!(x_temp, x, A.stencil_coefs, mid, bpc, i)
    end
end


function convolve_BC_left!{T<:Real,S<:SVector,RBC}(x_temp::AbstractVector{T}, x::AbstractVector{T}, A::UpwindOperator{T,S,:Dirichlet0,RBC})
    stencil_length = A.stencil_length
    x[1] = zero(T)
    stencil_rem = 1-stencil_length%2
    for i in 1 : A.boundary_point_count[1]
        A.directions[][i] ? start_idx = stencil_length-1 + (stencil_length)%2 : start_idx = 2 - stencil_length%2
        # we have to modify the number of boundary points to be considered as with upwind operators
        # the number of bpc is only 0 or 1 depending on the order
        A.directions[][i] ? bpc = A.boundary_point_count[1] : bpc = stencil_rem
        # println("*** i = $i, start_idx/mid = $start_idx, bpc = $bpc, stencil_length = $stencil_length ***")
        dirichlet_0!(x_temp, x, A.directions[][i] ? A.down_stencil_coefs : A.up_stencil_coefs, start_idx, bpc, i)
    end
end


function convolve_BC_left!{T<:Real,S<:SVector,RBC}(x_temp::AbstractVector{T}, x::AbstractVector{T}, A::DerivativeOperator{T,S,:Dirichlet,RBC})
    x[1] = A.boundary_condition[][1][3](A.t)
    mid = div(A.stencil_length,2)+1
    for i in 1 : A.boundary_point_count[1]
        dirichlet_1!(x_temp, x, A.stencil_coefs, mid, i)
    end
end


function convolve_BC_left!{T<:Real,S<:SVector,RBC}(x_temp::AbstractVector{T}, x::AbstractVector{T}, A::DerivativeOperator{T,S,:periodic,RBC})
    for i in 1 : A.boundary_point_count[1]
        periodic!(x_temp, x, A.stencil_coefs, i)
    end
end


function convolve_BC_left!{T<:Real,S<:SVector,RBC}(x_temp::AbstractVector{T}, x::AbstractVector{T}, A::DerivativeOperator{T,S,:Neumann0,RBC})
    for i in 1 : A.boundary_point_count[1]
        neumann0!(x_temp, x, A.stencil_coefs, i)
    end
end


function convolve_BC_left!{T<:Real,S<:SVector,RBC}(x_temp::AbstractVector{T}, x::AbstractVector{T}, A::DerivativeOperator{T,S,:Neumann,RBC})
    @inbounds for i in 1 : A.boundary_point_count[1]
        bc = A.low_boundary_coefs[][i]
        tmp = zero(T)
        @inbounds for j in 1 : length(bc)
            tmp += bc[j] * x[j]
        end
        x_temp[i] = tmp
    end
    x_temp[1] += A.boundary_condition[][1][3](A.t)
end


function convolve_BC_left!{T<:Real,S<:SVector,RBC}(x_temp::AbstractVector{T}, x::AbstractVector{T}, A::DerivativeOperator{T,S,:Robin,RBC})
    @inbounds for i in 1 : A.boundary_point_count[1]
        bc = A.low_boundary_coefs[][i]
        tmp = zero(T)
        @inbounds for j in 1 : length(bc)
            tmp += bc[j] * x[j]
        end
        x_temp[i] = tmp
    end
    x_temp[1] += A.boundary_condition[][1][3](A.t)
end


function convolve_BC_left!{T<:Real,S<:SVector,RBC}(x_temp::AbstractVector{T}, x::AbstractVector{T}, A::Union{DerivativeOperator{T,S,:None,RBC},UpwindOperator{T,S,:None,RBC}})
    halfstencil = div(A.stencil_length, 2)
    for i in 1 : A.boundary_point_count[1]
        @inbounds bc = A.low_boundary_coefs[][i]
        tmp = zero(T)
        startid = max(0,i-1-halfstencil)
        @inbounds for j in 1 : length(bc)
            tmp += bc[j] * x[startid+j]
        end
        @inbounds x_temp[i] = tmp
    end
end


function convolve_BC_left!{T<:Real,S<:SVector,RBC}(x_temp::AbstractVector{T}, x::AbstractVector{T}, A::UpwindOperator{T,S,:nothing,RBC})
    stencil_length = length(A.down_stencil_coefs)
    stencil_rem = 1 - stencil_length%2
    start_idx = 1

    # this case when our stencil uses 1 point at the left, so we cant use the upwind stencil
    # as it will spill over. So we use a special boundary stencil.

    if stencil_rem == 1
        x_temp[1] = sum(A.low_boundary_coefs[][1].*x[1:length(A.low_boundary_coefs[][1])])
        start_idx = 2
    end

    for i in start_idx : A.boundary_point_count[1]
        xtempi = zero(T)
        # startid = max(0,i-1-halfstencil)
        @inbounds for j in 1:length(A.up_stencil_coefs)
            xtempi += A.up_stencil_coefs[j] * x[i+j-1-stencil_rem]
        end
        @inbounds x_temp[end-i+1] = xtempi
    end
end


#= INTERIOR CONVOLUTION =#
function convolve_interior!{T<:Real,S<:SVector,LBC,RBC}(x_temp::AbstractVector{T}, x::AbstractVector{T}, A::DerivativeOperator{T,S,LBC,RBC})
    N = length(x)
    coeffs = A.stencil_coefs
    mid = div(A.stencil_length, 2) + 1

    Threads.@threads for i in A.boundary_point_count[1]+1 : N-A.boundary_point_count[2]
        # dirichlet_0!(x_temp,x,A.stencil_coefs, i)
        xtempi = zero(T)
        @inbounds for idx in 1:A.stencil_length
            xtempi += coeffs[idx] * x[i - (mid-idx)]
        end
        x_temp[i] = xtempi
    end
end


#= INTERIOR CONVOLUTION =#
function convolve_interior!{T<:Real,S<:SVector,LBC,RBC}(x_temp::AbstractVector{T}, x::AbstractVector{T}, A::UpwindOperator{T,S,LBC,RBC})
    N = length(x)
    stencil_length = length(A.up_stencil_coefs)
    stencil_rem = 1 - stencil_length%2
    Threads.@threads for i in A.boundary_point_count[1]+1 : N-A.boundary_point_count[2]
        xtempi = zero(T)
        if A.directions[][i] == false
            @inbounds for j in 1 : length(A.up_stencil_coefs)
                xtempi += A.up_stencil_coefs[j] * x[i+j-1-stencil_rem]
            end
        else
            @inbounds for j in -length(A.down_stencil_coefs)+1 : 0
                xtempi += A.down_stencil_coefs[j+stencil_length] * x[i+j+stencil_rem]
                # println("i = $i, j = $j, s_idx = $(j+stencil_length), x_idx = $(i+j+stencil_rem), $(A.down_stencil_coefs[j+stencil_length]) * $(x[i+j+stencil_rem]), xtempi = $xtempi")
            end
        end

        x_temp[i] = xtempi
    end
end


#= RIGHT BOUNDARY CONDITIONS =#
function convolve_BC_right!{T<:Real,S<:SVector,LBC}(x_temp::AbstractVector{T}, x::AbstractVector{T}, A::DerivativeOperator{T,S,LBC,:Dirichlet0})
    # Dirichlet 0 means that the value at the boundary is 0
    N = length(x)
    mid = div(A.stencil_length,2) + 1
    bpc = A.stencil_length - mid
    x[end] = zero(T)
    for i in 1 : A.boundary_point_count[2]
        dirichlet_0!(x_temp, x, A.stencil_coefs, mid, bpc, N - A.boundary_point_count[2] + i)
    end
end


function convolve_BC_right!{T<:Real,S<:SVector,LBC}(x_temp::AbstractVector{T}, x::AbstractVector{T}, A::UpwindOperator{T,S,LBC,:Dirichlet0})
    # Dirichlet 0 means that the value at the boundary is 0
    N = length(x)
    bpc = A.boundary_point_count[2]
    stencil_length = A.stencil_length
    x[end] = zero(T)

    for i in 1 : A.boundary_point_count[2]
        pt_idx = N - A.boundary_point_count[2] + i
        A.directions[][pt_idx] ? start_idx = stencil_length-1 + (stencil_length)%2 : start_idx = 2 - stencil_length%2
        dirichlet_0!(x_temp, x, A.directions[][pt_idx] ? A.down_stencil_coefs : A.up_stencil_coefs, start_idx, bpc, pt_idx)
    end
end


function convolve_BC_right!{T<:Real,S<:SVector,LBC}(x_temp::AbstractVector{T}, x::AbstractVector{T}, A::DerivativeOperator{T,S,LBC,:Dirichlet})
    N = length(x)
    mid = div(A.stencil_length,2) + 1
    x[end] = A.boundary_condition[][2][3](A.t)

    for i in 1 : A.boundary_point_count[2]
        dirichlet_1!(x_temp, x, A.stencil_coefs, mid, N - A.boundary_point_count[2] + i)
    end
end


function convolve_BC_right!{T<:Real,S<:SVector,LBC}(x_temp::AbstractVector{T}, x::AbstractVector{T}, A::DerivativeOperator{T,S,LBC,:periodic})
    N = length(x)
    for i in 1 : A.boundary_point_count[2]
        periodic!(x_temp, x, A.stencil_coefs, N - A.boundary_point_count[2] + i)
    end
end


function convolve_BC_right!{T<:Real,S<:SVector,LBC}(x_temp::AbstractVector{T}, x::AbstractVector{T}, A::DerivativeOperator{T,S,LBC,:Neumann0})
    N = length(x)
    for i in 1 : A.boundary_point_count[2]
        neumann0!(x_temp, x, A.stencil_coefs, N - A.boundary_point_count[2] + i)
    end
end


function convolve_BC_right!{T<:Real,S<:SVector,LBC}(x_temp::AbstractVector{T}, x::AbstractVector{T}, A::DerivativeOperator{T,S,LBC,:Neumann})
    N = length(x)
    @inbounds for i in 1 : A.boundary_point_count[2]
        bc = A.high_boundary_coefs[][A.boundary_point_count[2] - i + 1]
        tmp = zero(T)
        @inbounds for j in 1 : length(bc)
            # our coefficients and points are aligned so as we have not reversed anything in the stencil
            tmp += bc[j] * x[N-j+1]
        end
        x_temp[N-i+1] = tmp
    end
    x_temp[end] += A.boundary_condition[][2][3](A.t)
end


function convolve_BC_right!{T<:Real,S<:SVector,LBC}(x_temp::AbstractVector{T}, x::AbstractVector{T}, A::DerivativeOperator{T,S,LBC,:Robin})
    N = length(x)
    @inbounds for i in 1 : A.boundary_point_count[2]
        bc = A.high_boundary_coefs[][A.boundary_point_count[2] - i + 1]
        tmp = zero(T)
        @inbounds for j in 1 : length(bc)
            tmp += bc[j] * x[N-j+1]
        end
        x_temp[N-i+1] = tmp
    end
    x_temp[end] += A.boundary_condition[][2][3](A.t)
end


function convolve_BC_right!{T<:Real,S<:SVector,LBC}(x_temp::AbstractVector{T}, x::AbstractVector{T}, A::Union{DerivativeOperator{T,S,LBC,:None},UpwindOperator{T,S,LBC,:None}})
    # halfstencil = div(A.stencil_length, 2)
    for i in 1 : A.boundary_point_count[2] # the first stencil is for the last point ie. in reverse order
        @inbounds bc = A.high_boundary_coefs[][i]
        tmp = zero(T)
        # startid = max(0,i-1-halfstencil)
        @inbounds for j in 1 : length(bc)
            tmp += bc[j] * x[end-length(bc)+j]
        end
        @inbounds x_temp[end-i+1] = tmp
    end
end


function convolve_BC_right!{T<:Real,S<:SVector,LBC}(x_temp::AbstractVector{T}, x::AbstractVector{T}, A::UpwindOperator{T,S,LBC,:nothing})
    stencil_length = length(A.down_stencil_coefs)
    stencil_rem = 1 - stencil_length%2
    start_idx = 1

    # this case when our stencil uses 1 point at the right, so we cant use the downwind stencil
    # as it will spill over. So we use a special boundary stencil.
    if stencil_rem == 1
        x_temp[end] = sum(A.high_boundary_coefs[][1].*x[end-length(A.high_boundary_coefs[][1])+1:end])
        start_idx = 2
    # else
    #     x_temp[end] = sum(A.down_stencil_coefs.*x[end-length(A.down_stencil_coefs)+1:end])
    end

    # println(start_idx)
    for i in start_idx : A.boundary_point_count[2] # the first stencil is for the last point ie. in reverse order
        xtempi = zero(T)
        # startid = max(0,i-1-halfstencil)
        @inbounds for j in -length(A.down_stencil_coefs)+1 : 0
            # println("$(A.down_stencil_coefs[j+stencil_length]) * $(x[end+1-i+j+stencil_rem])")
            xtempi += A.down_stencil_coefs[j+stencil_length] * x[end+1-i+j+stencil_rem]
        end
        @inbounds x_temp[end-i+1] = xtempi
    end
end


#= DIFFERENT BOUNDARIES =#
function dirichlet_0!{T<:Real}(x_temp::AbstractVector{T}, x::AbstractVector{T}, coeffs::SVector, mid::Int, bpc::Int, i::Int)
    #=
        The high and low functions determine the starting and ending indices of the weight vector.
        As we move along the input vector to calculate the derivative at the pointhe weights which
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
    # mid = div(stencil_length, 2) + 1 # generalizing to any mid for upwind operators
    N = length(x)
    wndw_low = i>bpc ? 1:max(1, low(i, mid, bpc))
    wndw_high = i>N-bpc ? min(stencil_length, high(i, mid, bpc, stencil_length, N)):stencil_length

    # println(wndw_low," ",wndw_high, " mid = ", mid)
    # println("#####")

    #=
        Here we are taking the weighted sum of a window of the input vector to calculate the derivative
        at the middle point. This requires choosing the end points carefully which are being passed from above.
    =#
    xtempi = zero(T)
    @inbounds for idx in wndw_low:wndw_high
        xtempi += coeffs[idx] * x[(i - (mid-idx))]
        println("i = $i, idx = $((i - (mid-idx))), $(coeffs[idx]) * $(x[(i - (mid-idx))]), xtempi = $xtempi")
    end
    x_temp[i] = xtempi
end


function dirichlet_1!{T<:Real}(x_temp::AbstractVector{T}, x::AbstractVector{T}, coeffs::SVector, mid::Int, i::Int)
    stencil_length = length(coeffs)
    N = length(x)
    #=
        Here we are taking the weighted sum of a window of the input vector to calculate the derivative
        at the middle point. Once the stencil goes out of the edge, we assume that it's has a constant
        value outside for all points.
    =#
    xtempi = zero(T)
    @inbounds for idx in 1:stencil_length
        xtempi += coeffs[idx] * x[limit(i - (mid-idx), N)]
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
    xtempi = zero(T)
    @inbounds for idx in wndw_low:wndw_high
        xtempi += coeffs[idx] * x[rem1(i - (mid-idx), L)]
    end
    x_temp[i] = xtempi
end


function neumann0!{T<:Real}(x_temp::AbstractVector{T}, x::AbstractVector{T}, coeffs::SVector, i::Int)
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
    xtempi = zero(T)
    @inbounds for idx in wndw_low:wndw_high
        xtempi += coeffs[idx] * x[reflect(i - (mid-idx), L)]
    end
    x_temp[i] = xtempi
end
