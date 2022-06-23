#
# Convolutions that cannot be done by NNlib.conv!
#
# There are three convolution routines. They compute the derivative
# on the left margin, interior, and right margin of the grid. They
# are used by the mul! methods defined in derivative_operator_functions.jl
# to compute derivatives. When possible, the mul! methods use
# NNlib.conv! for the interior.
#
# These methods are defined for AbstractVector as a fallback. However,
# derivative operators normally act on a BoundaryPaddedVector returned
# by a boundary condition operator, and there are specialized methods
# to avoid allocation in this case.
#

# mul! done by convolutions
function LinearAlgebra.mul!(
    x_temp::AbstractVector,
    A::DerivativeOperator,
    x::AbstractVector;
    overwrite = true,
)
    convolve_BC_left!(x_temp, x, A, overwrite = overwrite)
    convolve_interior!(x_temp, x, A, overwrite = overwrite)
    convolve_BC_right!(x_temp, x, A, overwrite = overwrite)
end

################################################

# Against a standard vector, assume already padded and just apply the stencil
function convolve_interior!(
    x_temp::AbstractVector{T1},
    x::AbstractVector{T2},
    A::DerivativeOperator{T3,N,false};
    overwrite = true,
    add_range = false,
    offset::Int = 0,
) where {T1,T2,T3,N}
    T = promote_type(T1, T2, T3)
    @assert length(x_temp) + 2 == length(x)
    stencil = A.stencil_coefs

    # Initialize cur_stencil so that LoopVectorization.check_args(curr_stencil) doesn't throw undef variable for cur_stencil
    cur_stencil = 0

    coeff = A.coefficients
    len = length(x_temp)
    mid = div(A.stencil_length, 2)
    if !add_range
        if eltype(stencil) <: AbstractVector
            @turbo for i = (1+A.boundary_point_count):(len-A.boundary_point_count)
                xtempi = zero(T)
                cur_stencil = stencil[i-A.boundary_point_count]
                cur_coeff = coeff[i]
                for idx = 1:A.stencil_length
                    xtempi += cur_coeff * cur_stencil[idx] * x[i-mid+idx]
                end
                x_temp[i] = xtempi + !overwrite * x_temp[i]
            end
        else
            @turbo for i = (1+A.boundary_point_count):(len-A.boundary_point_count)
                xtempi = zero(T)
                cur_coeff = coeff[i]
                for idx = 1:A.stencil_length
                    xtempi += cur_coeff * stencil[idx] * x[i-mid+idx]
                end
                x_temp[i] = xtempi + !overwrite * x_temp[i]
            end
        end
    else
        for i in [
            (1+A.boundary_point_count):(A.boundary_point_count+offset)
            (len-A.boundary_point_count-offset+1):(len-A.boundary_point_count)
        ]
            xtempi = zero(T)
            cur_stencil =
                eltype(stencil) <: AbstractVector ? stencil[i-A.boundary_point_count] :
                stencil
            cur_coeff = coeff[i]
            for idx = 1:A.stencil_length
                xtempi += cur_coeff * cur_stencil[idx] * x[i-mid+idx]
            end
            x_temp[i] = xtempi + !overwrite * x_temp[i]
        end
    end
end

function convolve_BC_left!(
    x_temp::AbstractVector{T1},
    x::AbstractVector{T2},
    A::DerivativeOperator{T3,N,false};
    overwrite = true,
) where {T1,T2,T3,N}
    T = promote_type(T1, T2, T3)
    stencil = A.low_boundary_coefs

    # Initialize cur_stencil so that LoopVectorization.check_args(curr_stencil) doesn't throw undef variable for cur_stencil
    cur_stencil = 0

    coeff = A.coefficients
    @turbo for i = 1:A.boundary_point_count
        cur_stencil = stencil[i]
        cur_coeff = coeff[i]
        xtempi = zero(T)
        for idx = 1:A.boundary_stencil_length
            xtempi += cur_coeff * cur_stencil[idx] * x[idx]
        end
        x_temp[i] = xtempi + !overwrite * x_temp[i]
    end
end

function convolve_BC_right!(
    x_temp::AbstractVector{T1},
    x::AbstractVector{T2},
    A::DerivativeOperator{T3,N,false};
    overwrite = true,
) where {T1,T2,T3,N}
    T = promote_type(T1, T2, T3)
    stencil = A.high_boundary_coefs

    # Initialize cur_stencil so that LoopVectorization.check_args(curr_stencil) doesn't throw undef variable for cur_stencil
    cur_stencil = 0

    coeff = A.coefficients
    @turbo for i = 1:A.boundary_point_count
        cur_stencil = stencil[i]
        cur_coeff = coeff[i]
        xtempi = zero(T)
        for idx = (A.boundary_stencil_length-1):-1:0
            xtempi += cur_coeff * cur_stencil[end-idx] * x[end-idx]
        end
        x_temp[end-A.boundary_point_count+i] =
            xtempi + !overwrite * x_temp[end-A.boundary_point_count+i]
    end
end

################################################################################
# Uniform grid Upwind convolutions
################################################################################
function convolve_interior!(
    x_temp::AbstractVector{T},
    x::AbstractVector{T},
    A::DerivativeOperator{T,N,true};
    overwrite = true,
) where {T<:Real,N}
    @assert length(x_temp) + 2 == length(x)
    stencil = A.stencil_coefs
    coeff = A.coefficients

    for i = (1+A.boundary_point_count):(length(x_temp)-A.boundary_point_count-A.offside)
        xtempi = zero(T)
        cur_stencil = stencil
        cur_coeff = coeff[i]
        cur_stencil =
            cur_coeff >= 0 ? cur_stencil :
            A.derivative_order % 2 == 0 ? reverse(cur_stencil) : -1 * reverse(cur_stencil)
        for idx = 1:A.stencil_length
            x_idx =
                cur_coeff < 0 ? x[i-A.stencil_length+1+idx+A.offside] : x[i+idx-A.offside]
            xtempi += cur_coeff * cur_stencil[idx] * x_idx
        end
        x_temp[i] = xtempi + !overwrite * x_temp[i]
    end
end

function convolve_BC_left!(
    x_temp::AbstractVector{T},
    x::AbstractVector{T},
    A::DerivativeOperator{T,N,true};
    overwrite = true,
) where {T<:Real,N}
    coeff = A.coefficients
    upwind_stencils = A.stencil_coefs
    downwind_stencils = A.low_boundary_coefs
    for i = 1:A.boundary_point_count
        cur_coeff = coeff[i]
        xtempi = 0.0
        if cur_coeff >= 0 && i + A.stencil_length <= length(x) && i >= A.offside
            cur_stencil = upwind_stencils
            for idx = 1:A.stencil_length
                xtempi += cur_coeff * cur_stencil[idx] * x[i+idx-A.offside]
            end
        else
            cur_stencil = downwind_stencils[i]
            for idx = 1:A.boundary_stencil_length
                xtempi += cur_coeff * cur_stencil[idx] * x[idx]
            end
        end
        x_temp[i] = xtempi + !overwrite * x_temp[i]
    end
end

function convolve_BC_right!(
    x_temp::AbstractVector{T},
    x::AbstractVector{T},
    A::DerivativeOperator{T,N,true};
    overwrite = true,
) where {T<:Real,N}
    coeff = A.coefficients
    upwind_stencils = A.high_boundary_coefs
    downwind_stencils = A.stencil_coefs
    x_temp_len = length(x_temp)
    _x_len = length(x)
    for i = 1:A.boundary_point_count+A.offside
        cur_coeff = coeff[x_temp_len-A.boundary_point_count+i-A.offside]
        xtempi = 0.0
        if cur_coeff < 0 &&
           _x_len - A.stencil_length - A.boundary_point_count + i >= 1 &&
           i <= A.boundary_point_count + 1
            cur_stencil = downwind_stencils
            cur_stencil = ((-1)^A.derivative_order) * reverse(cur_stencil)
            for idx = 1:A.stencil_length
                xtempi +=
                    cur_coeff *
                    cur_stencil[idx] *
                    x[_x_len-A.stencil_length+idx-A.boundary_point_count+i-1]
            end
        elseif cur_coeff < 0 &&
               _x_len - A.stencil_length - A.boundary_point_count + i >= 1 &&
               i > A.boundary_point_count + 1
            cur_stencil = upwind_stencils[A.boundary_point_count+A.offside+1-i]
            cur_stencil = ((-1)^A.derivative_order) * reverse(cur_stencil)
            for idx = 1:A.boundary_stencil_length
                x_idx = x[_x_len-A.boundary_stencil_length+idx]
                xtempi += cur_coeff * cur_stencil[idx] * x_idx
            end
        elseif cur_coeff >= 0 && i < A.offside + 1
            cur_stencil = downwind_stencils
            for idx = 1:A.stencil_length
                xtempi +=
                    cur_coeff *
                    cur_stencil[idx] *
                    x[_x_len-A.stencil_length+i+idx-A.offside]
            end
        else
            cur_stencil = upwind_stencils[i]
            for idx = 1:A.boundary_stencil_length
                xtempi +=
                    cur_coeff * cur_stencil[idx] * x[_x_len-A.boundary_stencil_length+idx]
            end
        end
        x_temp[x_temp_len-A.boundary_point_count+i-A.offside] =
            xtempi + !overwrite * x_temp[x_temp_len-A.boundary_point_count+i-A.offside]
    end
end

################################################################################
# Non-uniform grid Upwind convolutions
################################################################################

function convolve_interior!(
    x_temp::AbstractVector{T},
    x::AbstractVector{T},
    A::DerivativeOperator{T,N,true,M};
    overwrite = true,
) where {T<:Real,N,M<:AbstractArray{T}}
    @assert length(x_temp) + 2 == length(x)

    len = A.len
    bpc = A.boundary_point_count
    stl = A.stencil_length
    coeff = A.coefficients

    # Initialize cur_stencil so that LoopVectorization.check_args(curr_stencil) doesn't throw undef variable for cur_stencil
    cur_stencil = 0

    if A.coeff_func isa Number && A.coeff_func >= 0
        @turbo for i = bpc+1:len-bpc-A.offside
            cur_coeff = coeff[i]
            xtempi = zero(T)
            cur_stencil = A.stencil_coefs[1, i-bpc]
            for idx = 1:stl
                xtempi += cur_coeff * cur_stencil[idx] * x[i+idx-A.offside]
            end
            x_temp[i] = xtempi + !overwrite * x_temp[i]
        end
    elseif A.coeff_func isa Number && A.coeff_func < 0
        @turbo for i = bpc+1:len-bpc-A.offside
            cur_coeff = coeff[i]
            xtempi = zero(T)
            cur_stencil = A.stencil_coefs[2, i-bpc]
            for idx = 1:stl
                xtempi += cur_coeff * cur_stencil[idx] * x[i-stl+1+idx+A.offside]
            end
            x_temp[i] = xtempi + !overwrite * x_temp[i]
        end
    else
        for i = bpc+1:len-bpc-A.offside
            cur_coeff = coeff[i]
            if cur_coeff >= 0
                xtempi = zero(T)
                cur_stencil = A.stencil_coefs[1, i-bpc]
                for idx = 1:stl
                    xtempi += cur_coeff * cur_stencil[idx] * x[i+idx-A.offside]
                end
                x_temp[i] = xtempi + !overwrite * x_temp[i]
            else
                xtempi = zero(T)
                cur_stencil = A.stencil_coefs[2, i-bpc]
                for idx = 1:stl
                    xtempi += cur_coeff * cur_stencil[idx] * x[i-stl+1+idx+A.offside]
                end
                x_temp[i] = xtempi + !overwrite * x_temp[i]
            end
        end
    end
end

function convolve_BC_left!(
    x_temp::AbstractVector{T},
    x::AbstractVector{T},
    A::DerivativeOperator{T,N,true,M};
    overwrite = true,
) where {T<:Real,N,M<:AbstractArray{T}}

    bpc = A.boundary_point_count
    stl = A.stencil_length
    bstl = A.boundary_stencil_length
    coeff = A.coefficients

    for i = 1:bpc
        cur_coeff = coeff[i]
        if cur_coeff >= 0 && A.offside == 0
            xtempi = zero(T)
            cur_stencil = A.low_boundary_coefs[1, i]
            for idx = 1:stl
                xtempi += cur_coeff * cur_stencil[idx] * x[i+idx]
            end
            x_temp[i] = xtempi + !overwrite * x_temp[i]
        elseif cur_coeff >= 0 && i < A.offside
            xtempi = zero(T)
            cur_stencil = A.low_boundary_coefs[1, i]
            for idx = 1:stl
                xtempi += cur_coeff * cur_stencil[idx] * x[idx]
            end
            x_temp[i] = xtempi + !overwrite * x_temp[i]
        elseif cur_coeff >= 0 && i >= A.offside
            xtempi = zero(T)
            cur_stencil = A.low_boundary_coefs[1, i]
            for idx = 1:stl
                xtempi += cur_coeff * cur_stencil[idx] * x[i+idx-A.offside]
            end
            x_temp[i] = xtempi + !overwrite * x_temp[i]
        else
            xtempi = zero(T)
            cur_stencil = A.low_boundary_coefs[2, i]
            for idx = 1:bstl
                xtempi += cur_coeff * cur_stencil[idx] * x[idx]
            end
            x_temp[i] = xtempi + !overwrite * x_temp[i]
        end
    end
end

function convolve_BC_right!(
    x_temp::AbstractVector{T},
    x::AbstractVector{T},
    A::DerivativeOperator{T,N,true,M};
    overwrite = true,
) where {T<:Real,N,M<:AbstractArray{T}}

    len = A.len
    bpc = A.boundary_point_count
    stl = A.stencil_length
    bstl = A.boundary_stencil_length
    coeff = A.coefficients
    off = A.offside

    for i = len-bpc+1-off:len
        cur_coeff = coeff[i]
        if cur_coeff < 0
            xtempi = zero(T)
            cur_stencil = A.high_boundary_coefs[2, i-len+bpc+off]
            if i <= len - off
                for idx = 1:stl
                    xtempi += cur_coeff * cur_stencil[idx] * x[i-stl+1+idx+off]
                end
                x_temp[i] = xtempi + !overwrite * x_temp[i]
            else
                for idx = 1:stl
                    xtempi += cur_coeff * cur_stencil[idx] * x[len-stl+2+idx]
                end
                x_temp[i] = xtempi + !overwrite * x_temp[i]
            end
        else
            xtempi = zero(T)
            if i <= len - bpc
                cur_stencil = A.stencil_coefs[1, i-bpc]
                for idx = 1:stl
                    xtempi += cur_coeff * cur_stencil[idx] * x[i-stl+1+idx+off]
                end
                x_temp[i] = xtempi + !overwrite * x_temp[i]
            else
                cur_stencil = A.high_boundary_coefs[1, i-len+bpc+off]
                for idx = 1:bstl
                    xtempi += cur_coeff * cur_stencil[idx] * x[len-bstl+2+idx]
                end
                x_temp[i] = xtempi + !overwrite * x_temp[i]
            end
        end
    end
end

###########################################

# Against A BC-padded vector, specialize the computation to explicitly use the left, right, and middle parts
function convolve_interior!(
    x_temp::AbstractVector{T1},
    _x::BoundaryPaddedVector,
    A::DerivativeOperator{T2,N,false};
    overwrite = true,
) where {T1,T2,N}
    T = promote_type(T1, T2)
    stencil = A.stencil_coefs

    # Initialize cur_stencil so that LoopVectorization.check_args(curr_stencil) doesn't throw undef variable for cur_stencil
    cur_stencil = 0

    coeff = A.coefficients
    x = _x.u
    mid = div(A.stencil_length, 2) + 1
    # Just do the middle parts
    if eltype(stencil) <: AbstractVector
        @turbo for i = (2+A.boundary_point_count):(length(x_temp)-A.boundary_point_count)-1
            xtempi = zero(T)
            cur_stencil = stencil[i-A.boundary_point_count]
            cur_coeff = coeff[i-A.boundary_point_count]
            for idx = 1:A.stencil_length
                xtempi += cur_coeff * cur_stencil[idx] * x[(i-1)-(mid-idx)+1]
            end
            x_temp[i] = xtempi + !overwrite * x_temp[i]
        end
    else
        @turbo for i = (2+A.boundary_point_count):(length(x_temp)-A.boundary_point_count)-1
            xtempi = zero(T)
            cur_coeff = coeff[i-A.boundary_point_count]
            for idx = 1:A.stencil_length
                xtempi += cur_coeff * stencil[idx] * x[(i-1)-(mid-idx)+1]
            end
            x_temp[i] = xtempi + !overwrite * x_temp[i]
        end
    end
end

function convolve_BC_left!(
    x_temp::AbstractVector{T1},
    _x::BoundaryPaddedVector,
    A::DerivativeOperator{T2,N,false};
    overwrite = true,
) where {T1,T2,N}
    T = promote_type(T1, T2)
    stencil = A.low_boundary_coefs
    coeff = A.coefficients

    # Initialize cur_stencil so that LoopVectorization.check_args(curr_stencil) doesn't throw undef variable for cur_stencil
    cur_stencil = 0

    @turbo for i = 1:A.boundary_point_count
        cur_stencil = stencil[i]
        cur_coeff = coeff[i]
        xtempi = cur_coeff * cur_stencil[1] * _x.l
        for idx = 2:A.boundary_stencil_length
            xtempi += cur_coeff * cur_stencil[idx] * _x.u[idx-1]
        end
        x_temp[i] = xtempi + !overwrite * x_temp[i]
    end
    # need to account for x.l in first interior
    mid = div(A.stencil_length, 2) + 1
    x = _x.u
    i = 1 + A.boundary_point_count
    xtempi = zero(T)
    cur_stencil =
        eltype(A.stencil_coefs) <: AbstractVector ?
        A.stencil_coefs[i-A.boundary_point_count] : A.stencil_coefs
    cur_coeff = coeff[i-A.boundary_point_count]
    xtempi = cur_coeff * cur_stencil[1] * _x.l
    @turbo for idx = 2:A.stencil_length
        xtempi += cur_coeff * cur_stencil[idx] * x[(i-1)-(mid-idx)+1]
    end
    x_temp[i] = xtempi + !overwrite * x_temp[i]
end

function convolve_BC_right!(
    x_temp::AbstractVector{T1},
    _x::BoundaryPaddedVector,
    A::DerivativeOperator{T2,N,false};
    overwrite = true,
) where {T1,T2,N}
    T = promote_type(T1, T2)
    stencil = A.high_boundary_coefs

    # Initialize cur_stencil so that LoopVectorization.check_args(curr_stencil) doesn't throw undef variable for cur_stencil
    cur_stencil = 0

    coeff = A.coefficients
    bc_start = length(_x.u) - A.boundary_point_count
    # need to account for _x.r in last interior convolution
    mid = div(A.stencil_length, 2) + 1
    x = _x.u
    i = length(x_temp) - A.boundary_point_count
    xtempi = zero(T)
    cur_stencil =
        eltype(A.stencil_coefs) <: AbstractVector ?
        A.stencil_coefs[i-A.boundary_point_count] : A.stencil_coefs
    cur_coeff = coeff[i-A.boundary_point_count]
    xtempi = cur_coeff * cur_stencil[end] * _x.r
    @turbo for idx = 1:A.stencil_length-1
        xtempi += cur_coeff * cur_stencil[idx] * x[(i-1)-(mid-idx)+1]
    end
    x_temp[i] = xtempi + !overwrite * x_temp[i]
    @turbo for i = 1:A.boundary_point_count
        cur_stencil = stencil[i]
        cur_coeff = coeff[bc_start+i]
        xtempi = cur_coeff * cur_stencil[end] * _x.r
        for idx = (A.boundary_stencil_length-1):-1:1
            xtempi += cur_coeff * cur_stencil[end-idx] * _x.u[end-idx+1]
        end
        x_temp[bc_start+i] = xtempi + !overwrite * x_temp[bc_start+i]
    end
end

################################################################################
# Uniform grid Upwind convolutions
# Against A BC-padded vector, specialize the computation to explicitly use the left, right, and middle parts
################################################################################
################################################################################
# Uniform grid Upwind convolutions
# Against A BC-padded vector, specialize the computation to explicitly use the left, right, and middle parts
################################################################################
function convolve_interior!(
    x_temp::AbstractVector{T},
    _x::BoundaryPaddedVector,
    A::DerivativeOperator{T,N,true};
    overwrite = true,
) where {T<:Real,N}
    @assert length(x_temp) + 2 == length(_x)
    stencil = A.stencil_coefs
    coeff = A.coefficients
    x = _x.u
    _x_len = length(_x)

    for i = (1+A.boundary_point_count):(length(x_temp)-A.boundary_point_count-A.offside)
        xtempi = zero(T)
        cur_stencil = stencil
        cur_coeff = coeff[i]
        cur_stencil =
            cur_coeff >= 0 ? cur_stencil :
            A.derivative_order % 2 == 0 ? reverse(cur_stencil) : -1 * reverse(cur_stencil)
        for idx = 1:A.stencil_length
            index =
                cur_coeff < 0 ? i - A.stencil_length + 1 + idx + A.offside :
                i + idx - A.offside
            if index == _x_len
                x_idx = _x.r
            elseif index == 1
                x_idx = _x.l
            else
                x_idx = x[index-1]
            end
            xtempi += cur_coeff * cur_stencil[idx] * x_idx
        end
        x_temp[i] = xtempi + !overwrite * x_temp[i]
    end
end

function convolve_BC_left!(
    x_temp::AbstractVector{T},
    _x::BoundaryPaddedVector,
    A::DerivativeOperator{T,N,true};
    overwrite = true,
) where {T<:Real,N}
    coeff = A.coefficients
    upwind_stencils = A.stencil_coefs
    downwind_stencils = A.low_boundary_coefs
    x = _x.u

    for i = 1:A.boundary_point_count
        cur_coeff = coeff[i]
        xtempi = 0.0
        if cur_coeff >= 0 && i + A.stencil_length <= length(_x) && i >= A.offside
            cur_stencil = upwind_stencils
            for idx = 1:A.stencil_length
                x_idx = i + idx - A.offside == 1 ? _x.l : x[i+idx-1-A.offside]
                xtempi += cur_coeff * cur_stencil[idx] * x_idx
            end
        else
            cur_stencil = downwind_stencils[i]
            for idx = 1:A.boundary_stencil_length
                x_idx = idx == 1 ? _x.l : x[idx-1]
                xtempi += cur_coeff * cur_stencil[idx] * x_idx
            end
        end
        x_temp[i] = xtempi + !overwrite * x_temp[i]
    end
end

function convolve_BC_right!(
    x_temp::AbstractVector{T},
    _x::BoundaryPaddedVector,
    A::DerivativeOperator{T,N,true};
    overwrite = true,
) where {T<:Real,N}
    coeff = A.coefficients
    upwind_stencils = A.high_boundary_coefs
    downwind_stencils = A.stencil_coefs
    x_temp_len = length(x_temp)
    x = _x.u
    _x_len = length(_x)

    for i = 1:A.boundary_point_count+A.offside
        cur_coeff = coeff[x_temp_len-A.boundary_point_count+i-A.offside]
        xtempi = 0.0
        if cur_coeff < 0 &&
           _x_len - A.stencil_length - A.boundary_point_count + i >= 1 &&
           i <= A.boundary_point_count + 1
            cur_stencil = downwind_stencils
            cur_stencil = ((-1)^A.derivative_order) * reverse(cur_stencil)
            for idx = 1:A.stencil_length
                x_idx =
                    _x_len - A.stencil_length + idx - A.boundary_point_count + i - 1 ==
                    _x_len ? _x.r :
                    x[_x_len-A.stencil_length+idx-A.boundary_point_count+i-2]
                xtempi += cur_coeff * cur_stencil[idx] * x_idx
            end
        elseif cur_coeff < 0 &&
               _x_len - A.stencil_length - A.boundary_point_count + i >= 1 &&
               i > A.boundary_point_count + 1
            cur_stencil = upwind_stencils[A.boundary_point_count+A.offside+1-i]
            cur_stencil = ((-1)^A.derivative_order) * reverse(cur_stencil)
            for idx = 1:A.boundary_stencil_length
                x_idx =
                    _x_len - A.boundary_stencil_length + idx == _x_len ? _x.r :
                    x[_x_len-A.boundary_stencil_length+idx-1]
                xtempi += cur_coeff * cur_stencil[idx] * x_idx
            end
        elseif cur_coeff >= 0 && i < A.offside + 1
            cur_stencil = downwind_stencils
            for idx = 1:A.stencil_length
                x_idx =
                    _x_len - A.stencil_length + idx - A.offside + i == _x_len ? _x.r :
                    x[_x_len-A.stencil_length+idx-A.offside+i-1]
                xtempi += cur_coeff * cur_stencil[idx] * x_idx
            end
        else
            cur_stencil = upwind_stencils[i]
            for idx = 1:A.boundary_stencil_length
                x_idx =
                    _x_len - A.boundary_stencil_length + idx == _x_len ? _x.r :
                    x[_x_len-A.boundary_stencil_length+idx-1]
                xtempi += cur_coeff * cur_stencil[idx] * x_idx
            end
        end
        x_temp[x_temp_len-A.boundary_point_count+i-A.offside] =
            xtempi + !overwrite * x_temp[x_temp_len-A.boundary_point_count+i-A.offside]
    end
end

################################################################################
# Non-uniform grid Upwind convolutions
# Against A BC-padded vector, specialize the computation to explicitly use the left, right, and middle parts
################################################################################
function convolve_interior!(
    x_temp::AbstractVector{T},
    _x::BoundaryPaddedVector,
    A::DerivativeOperator{T,N,true,M};
    overwrite = true,
) where {T<:Real,N,M<:AbstractArray{T}}
    @assert length(x_temp) + 2 == length(_x)

    len = A.len
    bpc = A.boundary_point_count
    stl = A.stencil_length
    coeff = A.coefficients
    x = _x.u
    _x_len = length(_x)


    for i = bpc+1:len-bpc-A.offside
        cur_coeff = coeff[i]
        if cur_coeff >= 0
            xtempi = zero(T)
            cur_stencil = A.stencil_coefs[1, i-bpc]
            for idx = 1:stl
                x_idx =
                    i + idx - A.offside == _x_len ? _x.r :
                    i + idx - A.offside == 1 ? _x.l : x[i+idx-1-A.offside]
                xtempi += cur_coeff * cur_stencil[idx] * x_idx
            end
            x_temp[i] = xtempi + !overwrite * x_temp[i]
        else
            xtempi = zero(T)
            cur_stencil = A.stencil_coefs[2, i-bpc]
            for idx = 1:stl
                x_idx = i - stl + 1 + idx + A.offside > 1 ? x[i-stl+idx+A.offside] : _x.l
                xtempi += cur_coeff * cur_stencil[idx] * x_idx
            end
            x_temp[i] = xtempi + !overwrite * x_temp[i]
        end
    end
end

function convolve_BC_left!(
    x_temp::AbstractVector{T},
    _x::BoundaryPaddedVector,
    A::DerivativeOperator{T,N,true,M};
    overwrite = true,
) where {T<:Real,N,M<:AbstractArray{T}}

    bpc = A.boundary_point_count
    stl = A.stencil_length
    bstl = A.boundary_stencil_length
    coeff = A.coefficients
    x = _x.u
    _x_len = length(_x)

    for i = 1:bpc
        cur_coeff = coeff[i]
        if cur_coeff >= 0 && A.offside == 0
            xtempi = zero(T)
            cur_stencil = A.low_boundary_coefs[1, i]
            for idx = 1:stl
                x_idx = i + idx < _x_len ? x[i+idx-1] : _x.r
                xtempi += cur_coeff * cur_stencil[idx] * x_idx
            end
            x_temp[i] = xtempi + !overwrite * x_temp[i]
        elseif cur_coeff >= 0 && i < A.offside
            xtempi = zero(T)
            cur_stencil = A.low_boundary_coefs[1, i]
            for idx = 1:stl
                x_idx = idx == 1 ? _x.l : x[idx-1]
                xtempi += cur_coeff * cur_stencil[idx] * x_idx
            end
            x_temp[i] = xtempi + !overwrite * x_temp[i]
        elseif cur_coeff >= 0 && i >= A.offside
            xtempi = zero(T)
            cur_stencil = A.low_boundary_coefs[1, i]
            for idx = 1:stl
                x_idx = i + idx - A.offside == 1 ? _x.l : x[i+idx-A.offside-1]
                xtempi += cur_coeff * cur_stencil[idx] * x_idx
            end
            x_temp[i] = xtempi + !overwrite * x_temp[i]
        else
            xtempi = zero(T)
            cur_stencil = A.low_boundary_coefs[2, i]
            for idx = 1:bstl
                x_idx = idx > 1 ? x[idx-1] : _x.l
                xtempi += cur_coeff * cur_stencil[idx] * x_idx
            end
            x_temp[i] = xtempi + !overwrite * x_temp[i]
        end
    end
end

function convolve_BC_right!(
    x_temp::AbstractVector{T},
    _x::BoundaryPaddedVector,
    A::DerivativeOperator{T,N,true,M};
    overwrite = true,
) where {T<:Real,N,M<:AbstractArray{T}}

    len = A.len
    bpc = A.boundary_point_count
    stl = A.stencil_length
    bstl = A.boundary_stencil_length
    coeff = A.coefficients
    off = A.offside
    x = _x.u
    _x_len = length(_x)

    for i = len-bpc+1-off:len
        cur_coeff = coeff[i]
        if cur_coeff < 0
            xtempi = zero(T)
            cur_stencil = A.high_boundary_coefs[2, i-len+bpc+off]
            if i <= len - off
                for idx = 1:stl
                    x_idx = i - stl + 1 + idx + off > 1 ? x[i-stl+idx+off] : _x.l
                    xtempi += cur_coeff * cur_stencil[idx] * x_idx
                end
                x_temp[i] = xtempi + !overwrite * x_temp[i]
            else
                for idx = 1:stl
                    x_idx = len - stl + 2 + idx < _x_len ? x[len-stl+1+idx] : _x.r
                    xtempi += cur_coeff * cur_stencil[idx] * x_idx
                end
                x_temp[i] = xtempi + !overwrite * x_temp[i]
            end
        else
            xtempi = zero(T)
            if i <= len - bpc
                cur_stencil = A.stencil_coefs[1, i-bpc]
                for idx = 1:stl
                    x_idx = i - stl + 1 + idx + off < _x_len ? x[i-stl+idx+off] : _x.r
                    xtempi += cur_coeff * cur_stencil[idx] * x_idx
                end
                x_temp[i] = xtempi + !overwrite * x_temp[i]
            else
                cur_stencil = A.high_boundary_coefs[1, i-len+bpc+off]
                for idx = 1:bstl
                    x_idx = len - bstl + 2 + idx < _x_len ? x[len-bstl+1+idx] : _x.r
                    xtempi += cur_coeff * cur_stencil[idx] * x_idx
                end
                x_temp[i] = xtempi + !overwrite * x_temp[i]
            end
        end
    end
end
