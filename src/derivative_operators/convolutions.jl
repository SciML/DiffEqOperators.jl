# mul! done by convolutions
function LinearAlgebra.mul!(x_temp::AbstractVector{T}, A::DerivativeOperator, x::AbstractVector{T}) where T<:Real
    convolve_BC_left!(x_temp, x, A)
    convolve_interior!(x_temp, x, A)
    convolve_BC_right!(x_temp, x, A)
    if typeof(A.dx) <: T
        rmul!(x_temp, @.(1/(A.dx^A.derivative_order)))
    else
        nothing # what to do when dx is not uniform???
    end
    rmul!(x_temp, @.(1/(A.dx^A.derivative_order)))
end

################################################

# Against a standard vector, assume already padded and just apply the stencil
function convolve_interior!(x_temp::AbstractVector{T}, x::AbstractVector{T}, A::DerivativeOperator) where {T<:Real}
    @assert length(x_temp)+2 == length(x)
    stencil = A.stencil_coefs
    coeff   = A.coefficients
    # Upwind operators have a non-centred stencil
    if use_winding(A)
        mid = 1 + A.stencil_length%2
    else
        mid = div(A.stencil_length,2)
    end
    for i in (1+A.boundary_point_count) : (length(x_temp)-A.boundary_point_count)
        xtempi = zero(T)
        cur_stencil = eltype(stencil) <: AbstractVector ? stencil[i] : stencil
        cur_coeff   = typeof(coeff)   <: AbstractVector ? coeff[i] : true
        cur_stencil = use_winding(A) && cur_coeff < 0 ? reverse(cur_stencil) : cur_stencil
        for idx in 1:A.stencil_length
            xtempi += cur_coeff * cur_stencil[idx] * x[i - mid + idx]
        end
        x_temp[i] = xtempi
    end
end

function convolve_BC_left!(x_temp::AbstractVector{T}, x::AbstractVector{T}, A::DerivativeOperator) where {T<:Real}
    stencil = A.low_boundary_coefs
    coeff   = A.coefficients
    for i in 1 : A.boundary_point_count
        xtempi = stencil[i][1]*x[1]
        cur_stencil = stencil[i]
        cur_coeff   = typeof(coeff)   <: AbstractVector ? coeff[i] : true
        cur_stencil = use_winding(A) && cur_coeff < 0 ? reverse(cur_stencil) : cur_stencil
        for idx in 2:A.boundary_stencil_length
            xtempi += cur_coeff * cur_stencil[idx] * x[idx]
        end
        x_temp[i] = xtempi
    end
end

function convolve_BC_right!(x_temp::AbstractVector{T}, x::AbstractVector{T}, A::DerivativeOperator) where {T<:Real}
    stencil = A.high_boundary_coefs
    coeff   = A.coefficients
    for i in 1 : A.boundary_point_count
        xtempi = stencil[i][end]*x[end]
        cur_stencil = stencil[i]
        cur_coeff   = typeof(coeff)   <: AbstractVector ? coeff[i] : true
        cur_stencil = use_winding(A) && cur_coeff < 0 ? reverse(cur_stencil) : cur_stencil
        for idx in (A.boundary_stencil_length-1):-1:1
            xtempi += cur_coeff * cur_stencil[end-idx] * x[end-idx]
        end
        x_temp[end-A.boundary_point_count+i] = xtempi
    end
end

###########################################

# Against A BC-padded vector, specialize the computation to explicitly use the left, right, and middle parts
function convolve_interior!(x_temp::AbstractVector{T}, _x::BoundaryPaddedVector, A::DerivativeOperator) where {T<:Real}
    stencil = A.stencil_coefs
    coeff   = A.coefficients
    x = _x.u
    # Upwind operators have a non-centred stencil
    if use_winding(A)
        mid = 1 + A.stencil_length%2
    else
        mid = div(A.stencil_length,2)
    end
    # Just do the middle parts
    for i in (2+A.boundary_point_count) : (length(x_temp)-A.boundary_point_count)-1
        xtempi = zero(T)
        cur_stencil = eltype(stencil) <: AbstractVector ? stencil[i-A.boundary_point_count] : stencil
        cur_coeff   = typeof(coeff)   <: AbstractVector ? coeff[i-A.boundary_point_count] : true
        cur_stencil = use_winding(A) && cur_coeff < 0 ? reverse(cur_stencil) : cur_stencil
        @inbounds for idx in 1:A.stencil_length
            xtempi += cur_coeff * cur_stencil[idx] * x[(i-1) - (mid-idx) + 1]
        end
        x_temp[i] = xtempi
    end
end

function convolve_BC_left!(x_temp::AbstractVector{T}, _x::BoundaryPaddedVector, A::DerivativeOperator) where {T<:Real}
    stencil = A.low_boundary_coefs
    coeff   = A.coefficients
    for i in 1 : A.boundary_point_count
        cur_stencil = stencil[i]
        cur_coeff   = typeof(coeff)   <: AbstractVector ? coeff[i] : true
        # need to account for x.l in first interior
        xtempi = cur_coeff*cur_stencil[1]*_x.l
        cur_stencil = use_winding(A) && cur_coeff < 0 ? reverse(cur_stencil) : cur_stencil
        @inbounds for idx in 2:A.boundary_stencil_length
            xtempi += cur_coeff * cur_stencil[idx] * _x.u[idx-1]
        end
        x_temp[i] = xtempi
    end
end

function convolve_BC_right!(x_temp::AbstractVector{T}, _x::BoundaryPaddedVector, A::DerivativeOperator) where {T<:Real}
    stencil = A.high_boundary_coefs
    coeff   = A.coefficients
    bc_start = length(_x.u) - A.boundary_point_count
    # need to account for _x.r in last interior convolution

    for i in 1 : A.boundary_point_count
        cur_stencil = stencil[i]
        cur_coeff   = typeof(coeff)   <: AbstractVector ? coeff[bc_start + i] : true
        xtempi = cur_coeff*cur_stencil[end]*_x.r
        cur_stencil = use_winding(A) && cur_coeff < 0 ? reverse(cur_stencil) : cur_stencil
        @inbounds for idx in A.stencil_length:-1:1
            xtempi += cur_coeff * cur_stencil[end-idx] * _x.u[end-idx+1]
        end
        x_temp[bc_start + i] = xtempi
    end
end
