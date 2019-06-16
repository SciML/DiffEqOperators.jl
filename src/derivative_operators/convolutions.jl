# mul! done by convolutions
function LinearAlgebra.mul!(x_temp::AbstractVector{T}, A::DerivativeOperator, x::AbstractVector{T}) where T<:Real
    convolve_BC_left!(x_temp, x, A)
    convolve_interior!(x_temp, x, A)
    convolve_BC_right!(x_temp, x, A)
    rmul!(x_temp, @.(1/(A.dx^A.derivative_order)))
end

################################################

# Against a standard vector, assume already padded and just apply the stencil
function convolve_interior!(x_temp::AbstractVector{T}, x::AbstractVector{T}, A::DerivativeOperator) where {T<:Real}
    @assert length(x_temp)+2 == length(x)
    stencil = A.stencil_coefs
    coeff   = A.coefficients
    mid = div(A.stencil_length,2)
    for i in (1+A.boundary_point_count) : (length(x_temp)-A.boundary_point_count)
        xtempi = zero(T)
        cur_stencil = eltype(stencil) <: AbstractVector ? stencil[i] : stencil
        cur_coeff   = eltype(coeff)   <: AbstractVector ? coeff[i] : true
        cur_stencil = cur_coeff < 0 && A.use_winding ? reverse(cur_stencil) : cur_stencil
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
        cur_coeff   = eltype(coeff)   <: AbstractVector ? coeff[i] : true
        cur_stencil = cur_coeff < 0 && A.use_winding ? reverse(cur_stencil) : cur_stencil
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
        cur_coeff   = eltype(coeff)   <: AbstractVector ? coeff[i] : true
        cur_stencil = cur_coeff < 0 && A.use_winding ? reverse(cur_stencil) : cur_stencil
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
    mid = div(A.stencil_length,2) + 1
    # Just do the middle parts
    for i in (1+A.boundary_length) : (length(x_temp)-A.boundary_length)
        xtempi = zero(T)
        cur_stencil = eltype(stencil) <: AbstractVector ? stencil[i-A.boundary_length] : stencil
        cur_coeff   = eltype(coeff)   <: AbstractVector ? coeff[i-A.boundary_length] : true
        cur_stencil = cur_coeff < 0 && A.use_winding ? reverse(cur_stencil) : cur_stencil
        @inbounds for idx in 1:A.stencil_length
            xtempi += cur_coeff * cur_stencil[idx] * x[i - (mid-idx) + 1]
        end
        x_temp[i] = xtempi
    end
end

function convolve_BC_left!(x_temp::AbstractVector{T}, _x::BoundaryPaddedVector, A::DerivativeOperator) where {T<:Real}
    stencil = A.low_boundary_coefs
    coeff   = A.coefficients
    for i in 1 : A.boundary_length
        xtempi = stencil[i][1]*x.l
        cur_stencil = stencil[i]
        cur_coeff   = eltype(coeff)   <: AbstractVector ? coeff[i-A.boundary_length] : true
        cur_stencil = cur_coeff < 0 && A.use_winding ? reverse(cur_stencil) : cur_stencil
        @inbounds for idx in 2:A.stencil_length
            xtempi += cur_coeff * cur_stencil[idx] * x.u[idx-1]
        end
        x_temp[i] = xtempi
    end
end

function convolve_BC_right!(x_temp::AbstractVector{T}, _x::BoundaryPaddedVector, A::DerivativeOperator) where {T<:Real}
    stencil = A.low_boundary_coefs
    coeff   = A.coefficients
    bc_start = length(x.u) - A.stencil_length
    for i in 1 : A.boundary_length
        xtempi = stencil[i][end]*x.r
        cur_stencil = stencil[i]
        cur_coeff   = eltype(coeff)   <: AbstractVector ? coeff[i-A.boundary_length] : true
        cur_stencil = cur_coeff < 0 && A.use_winding ? reverse(cur_stencil) : cur_stencil
        @inbounds for idx in A.stencil_length:-1:2
            xtempi += cur_coeff * cur_stencil[end-idx] * x.u[end-idx+1]
        end
        x_temp[i] = xtempi
    end
end
