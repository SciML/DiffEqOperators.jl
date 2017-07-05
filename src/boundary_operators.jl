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

function rem1(idx,L)
    r = idx%L
    if r > 0
        return r
    else
        return r+L
    end
end


#= LEFT BOUNDARY CONDITIONS =#
function convolve_BC_left!{T<:Real,S<:SVector,RBC}(x_temp::AbstractVector{T}, x::AbstractVector{T}, A::LinearOperator{T,S,:Dirichlet0,RBC})
    Threads.@threads for i in 1 : A.boundary_point_count
        dirichlet_0!(x_temp, x, A.stencil_coefs, i)
    end
end


function convolve_BC_left!{T<:Real,S<:SVector,RBC}(x_temp::AbstractVector{T}, x::AbstractVector{T}, A::LinearOperator{T,S,:Dirichlet,RBC})
    x[1] = A.boundary_fn[1][3]
    Threads.@threads for i in 1 : A.boundary_point_count
        dirichlet_1!(x_temp, x, A.stencil_coefs, i)
    end
end


function convolve_BC_left!{T<:Real,S<:SVector,RBC}(x_temp::AbstractVector{T}, x::AbstractVector{T}, A::LinearOperator{T,S,:periodic,RBC})
    Threads.@threads for i in 1 : A.boundary_point_count
        periodic!(x_temp, x, A.stencil_coefs, i)
    end
end


function convolve_BC_left!{T<:Real,S<:SVector,RBC}(x_temp::AbstractVector{T}, x::AbstractVector{T}, A::LinearOperator{T,S,:Neumann0,RBC})
    Threads.@threads for i in 1 : A.boundary_point_count
        neumann0!(x_temp, x, A.stencil_coefs, i)
    end
end


function convolve_BC_left!{T<:Real,S<:SVector,RBC}(x_temp::AbstractVector{T}, x::AbstractVector{T}, A::LinearOperator{T,S,:Neumann,RBC})
    @inbounds for i in 1 : A.boundary_point_count
        bc = A.low_boundary_coefs[i]
        tmp = zero(T)
        @inbounds for j in 1 : length(bc)
            tmp += bc[j] * x[j]
        end
        x_temp[i] = tmp
    end
    x_temp[1] += A.boundary_fn[1][3]
end


function convolve_BC_left!{T<:Real,S<:SVector,RBC}(x_temp::AbstractVector{T}, x::AbstractVector{T}, A::LinearOperator{T,S,:Robin,RBC})
    @inbounds for i in 1 : A.boundary_point_count
        bc = A.low_boundary_coefs[i]
        tmp = zero(T)
        @inbounds for j in 1 : length(bc)
            tmp += bc[j] * x[j]
        end
        x_temp[i] = tmp
    end
    x_temp[1] += A.boundary_fn[1][3]
end


function convolve_BC_left!{T<:Real,S<:SVector,RBC}(x_temp::AbstractVector{T}, x::AbstractVector{T}, A::LinearOperator{T,S,:None,RBC})
    halfstencil = div(A.stencil_length, 2)
    Threads.@threads for i in 1 : A.boundary_point_count
        @inbounds bc = A.low_boundary_coefs[i]
        tmp = zero(T)
        startid = max(0,i-1-halfstencil)
        @inbounds for j in 1 : length(bc)
            tmp += bc[j] * x[startid+j]
        end
        @inbounds x_temp[i] = tmp
    end
end


#= INTERIOR CONVOLUTION =#
function convolve_interior!{T<:Real,S<:SVector,LBC,RBC}(x_temp::AbstractVector{T}, x::AbstractVector{T}, A::LinearOperator{T,S,LBC,RBC})
    N = length(x)
    coeffs = A.stencil_coefs
    mid = div(A.stencil_length, 2) + 1

    Threads.@threads for i in A.boundary_point_count+1 : N-A.boundary_point_count
        # dirichlet_0!(x_temp,x,A.stencil_coefs, i)
        xtempi = zero(T)
        @inbounds for idx in 1:A.stencil_length
            xtempi += coeffs[idx] * x[i - (mid-idx)]
        end
        x_temp[i] = xtempi
    end
end


#= RIGHT BOUNDARY CONDITIONS =#
function convolve_BC_right!{T<:Real,S<:SVector,LBC}(x_temp::AbstractVector{T}, x::AbstractVector{T}, A::LinearOperator{T,S,LBC,:Dirichlet0})
    # Dirichlet 0 means that the value at the boundary is 0
    N = length(x)
    Threads.@threads for i in 1 : A.boundary_point_count
        dirichlet_0!(x_temp, x, A.stencil_coefs, N - A.boundary_point_count + i)
    end
end


function convolve_BC_right!{T<:Real,S<:SVector,LBC}(x_temp::AbstractVector{T}, x::AbstractVector{T}, A::LinearOperator{T,S,LBC,:Dirichlet})
    N = length(x)
    Threads.@threads for i in 1 : A.boundary_point_count
        dirichlet_1!(x_temp, x, A.stencil_coefs, N - A.boundary_point_count + i)
    end
    x[end] = A.boundary_fn[2][3]
end


function convolve_BC_right!{T<:Real,S<:SVector,LBC}(x_temp::AbstractVector{T}, x::AbstractVector{T}, A::LinearOperator{T,S,LBC,:periodic})
    N = length(x)
    Threads.@threads for i in 1 : A.boundary_point_count
        periodic!(x_temp, x, A.stencil_coefs, N - A.boundary_point_count + i)
    end
end


function convolve_BC_right!{T<:Real,S<:SVector,LBC}(x_temp::AbstractVector{T}, x::AbstractVector{T}, A::LinearOperator{T,S,LBC,:Neumann0})
    N = length(x)
    Threads.@threads for i in 1 : A.boundary_point_count
        neumann0!(x_temp, x, A.stencil_coefs, N - A.boundary_point_count + i)
    end
end


function convolve_BC_right!{T<:Real,S<:SVector,LBC}(x_temp::AbstractVector{T}, x::AbstractVector{T}, A::LinearOperator{T,S,LBC,:Neumann})
    N = length(x)
    @inbounds for i in 1 : A.boundary_point_count
        bc = A.high_boundary_coefs[A.boundary_point_count - i + 1]
        tmp = zero(T)
        @inbounds for j in 1 : length(bc)
            tmp += bc[j] * x[N-j+1]
        end
        x_temp[N-i+1] = tmp
    end
    x_temp[end] += A.boundary_fn[2][3]
end


function convolve_BC_right!{T<:Real,S<:SVector,LBC}(x_temp::AbstractVector{T}, x::AbstractVector{T}, A::LinearOperator{T,S,LBC,:Robin})
    N = length(x)
    @inbounds for i in 1 : A.boundary_point_count
        bc = A.high_boundary_coefs[A.boundary_point_count - i + 1]
        tmp = zero(T)
        @inbounds for j in 1 : length(bc)
            tmp += bc[j] * x[N-j+1]
        end
        x_temp[N-i+1] = tmp
    end
    x_temp[end] += A.boundary_fn[2][3]
end


function convolve_BC_right!{T<:Real,S<:SVector,LBC}(x_temp::AbstractVector{T}, x::AbstractVector{T}, A::LinearOperator{T,S,LBC,:None})
    halfstencil = div(A.stencil_length, 2)
    Threads.@threads for i in 1 : A.boundary_point_count
        @inbounds bc = A.high_boundary_coefs[i]
        tmp = zero(T)
        startid = max(0,i-1-halfstencil)
        @inbounds for j in 1 : length(bc)
            tmp += bc[j] * x[end-length(bc)-startid+j]
        end
        @inbounds x_temp[end-i+1] = tmp
    end
end


#= DIFFERENT BOUNDARIES =#
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
    N = length(x)
    wndw_low = i>bpc ? 1:max(1, low(i, mid, bpc))
    wndw_high = i>N-bpc ? min(stencil_length, high(i, mid, bpc, stencil_length, N)):stencil_length

    #=
        Here we are taking the weighted sum of a window of the input vector to calculate the derivative
        at the middle point. This requires choosing the end points carefully which are being passed from above.
    =#
    xtempi = zero(T)
    @inbounds for idx in wndw_low:wndw_high
        xtempi += coeffs[idx] * x[(i - (mid-idx))]
    end
    x_temp[i] = xtempi
end


function dirichlet_1!{T<:Real}(x_temp::AbstractVector{T}, x::AbstractVector{T}, coeffs::SVector, i::Int)
    stencil_length = length(coeffs)
    mid = div(stencil_length, 2) + 1
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
