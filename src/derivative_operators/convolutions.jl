# mul! done by convolutions
function LinearAlgebra.mul!(x_temp::AbstractVector{T}, A::DerivativeOperator, x::AbstractVector{T}; overwrite = true) where T<:Real
    convolve_BC_left!(x_temp, x, A, overwrite = overwrite)
    convolve_interior!(x_temp, x, A, overwrite = overwrite)
    convolve_BC_right!(x_temp, x, A, overwrite = overwrite)
end

################################################

# Against a standard vector, assume already padded and just apply the stencil
function convolve_interior!(x_temp::AbstractVector{T}, x::AbstractVector{T}, A::DerivativeOperator{T,N,false}; overwrite = true, add_range = false, offset::Int = 0) where {T<:Real, N}
    @assert length(x_temp)+2 == length(x)
    stencil = A.stencil_coefs
    coeff   = A.coefficients
    mid = div(A.stencil_length,2)
    if !add_range
        for i in (1+A.boundary_point_count) : (length(x_temp)-A.boundary_point_count)
            xtempi = zero(T)
            cur_stencil = eltype(stencil) <: AbstractVector ? stencil[i-A.boundary_point_count] : stencil
            cur_coeff   = typeof(coeff)   <: AbstractVector ? coeff[i] : coeff isa Number ? coeff : true
            for idx in 1:A.stencil_length
                xtempi += cur_coeff * cur_stencil[idx] * x[i - mid + idx]
            end
            x_temp[i] = xtempi + !overwrite*x_temp[i]
        end
    else
        for i in [(1+A.boundary_point_count):(A.boundary_point_count+offset); (length(x_temp)-A.boundary_point_count-offset+1):(length(x_temp)-A.boundary_point_count)]
            xtempi = zero(T)
            cur_stencil = eltype(stencil) <: AbstractVector ? stencil[i-A.boundary_point_count] : stencil
            cur_coeff   = typeof(coeff)   <: AbstractVector ? coeff[i] : coeff isa Number ? coeff : true
            for idx in 1:A.stencil_length
                xtempi += cur_coeff * cur_stencil[idx] * x[i - mid + idx]
            end
            x_temp[i] = xtempi + !overwrite*x_temp[i]
        end
    end
end

function convolve_BC_left!(x_temp::AbstractVector{T}, x::AbstractVector{T}, A::DerivativeOperator{T,N,false}; overwrite = true) where {T<:Real, N}
    stencil = A.low_boundary_coefs
    coeff   = A.coefficients
    for i in 1 : A.boundary_point_count
        cur_stencil = stencil[i]
        cur_coeff   = typeof(coeff)   <: AbstractVector ? coeff[i] : coeff isa Number ? coeff : true
        xtempi = cur_coeff*stencil[i][1]*x[1]
        for idx in 2:A.boundary_stencil_length
            xtempi += cur_coeff * cur_stencil[idx] * x[idx]
        end
        x_temp[i] = xtempi + !overwrite*x_temp[i]
    end
end

function convolve_BC_right!(x_temp::AbstractVector{T}, x::AbstractVector{T}, A::DerivativeOperator{T,N,false}; overwrite = true) where {T<:Real, N}
    stencil = A.high_boundary_coefs
    coeff   = A.coefficients
    for i in 1 : A.boundary_point_count
        cur_stencil = stencil[i]
        cur_coeff   = typeof(coeff)   <: AbstractVector ? coeff[i] : coeff isa Number ? coeff : true
        xtempi = cur_coeff*stencil[i][end]*x[end]
        for idx in (A.boundary_stencil_length-1):-1:1
            xtempi += cur_coeff * cur_stencil[end-idx] * x[end-idx]
        end
        x_temp[end-A.boundary_point_count+i] = xtempi + !overwrite*x_temp[end-A.boundary_point_count+i]
    end
end

# Against a standard vector, assume already padded and just apply the stencil
function convolve_interior!(x_temp::AbstractVector{T}, x::AbstractVector{T}, A::DerivativeOperator{T,N,true}; overwrite = true) where {T<:Real, N}
    @assert length(x_temp)+2 == length(x)
    stencil = A.stencil_coefs
    coeff   = A.coefficients

    for i in (1+A.boundary_point_count) : (length(x_temp)-A.boundary_point_count)
        xtempi = zero(T)
        cur_stencil = eltype(stencil) <: AbstractVector ? stencil[i-A.boundary_point_count] : stencil
        cur_coeff   = typeof(coeff)   <: AbstractVector ? coeff[i] : coeff isa Number ? coeff : true
        cur_stencil = cur_coeff >= 0 ? cur_stencil : A.derivative_order % 2 == 0 ? reverse(cur_stencil) : -1*reverse(cur_stencil)
        for idx in 1:A.stencil_length
            x_idx = cur_coeff < 0 ? x[i - A.stencil_length + 1 + idx] : x[i + idx]
            xtempi += cur_coeff * cur_stencil[idx] * x_idx
        end
        x_temp[i] = xtempi + !overwrite*x_temp[i]
    end
end

function convolve_BC_left!(x_temp::AbstractVector{T}, x::AbstractVector{T}, A::DerivativeOperator{T,N,true}; overwrite = true) where {T<:Real, N}
    coeff = A.coefficients
    upwind_stencils = A.stencil_coefs
    downwind_stencils = A.low_boundary_coefs
    for i in 1:A.boundary_point_count
        cur_coeff = coeff[i]
        xtempi = 0.0
        if cur_coeff >= 0 && i+A.stencil_length <= length(x)
            cur_stencil = eltype(upwind_stencils) <: AbstractVector ? upwind_stencils[i] : upwind_stencils
            for idx in 1:A.stencil_length
                xtempi += cur_coeff*cur_stencil[idx]*x[i+idx]
            end
        else
            cur_stencil = downwind_stencils[i]
            for idx in 1:A.boundary_stencil_length
                xtempi += cur_coeff*cur_stencil[idx]*x[idx]
            end
        end
        x_temp[i] = xtempi + !overwrite*x_temp[i]
    end
end

function convolve_BC_right!(x_temp::AbstractVector{T}, x::AbstractVector{T}, A::DerivativeOperator{T,N,true}; overwrite = true) where {T<:Real, N}
    coeff = A.coefficients
    upwind_stencils = A.high_boundary_coefs
    downwind_stencils = A.stencil_coefs
    x_temp_len = length(x_temp)
    x_len = length(x)
    for i in 1:A.boundary_point_count
        cur_coeff = coeff[x_temp_len-A.boundary_point_count+i]
        xtempi = 0.0
        if cur_coeff < 0 && x_len-A.stencil_length - A.boundary_point_count + i >= 1
            cur_stencil = eltype(downwind_stencils) <: AbstractVector ? downwind_stencils[i] : downwind_stencils
            cur_stencil = ((-1)^A.derivative_order)*reverse(cur_stencil)
            for idx in 1:A.stencil_length
                xtempi += cur_coeff*cur_stencil[idx]*x[x_len-A.stencil_length + idx - A.boundary_point_count + i - 1]
            end
        else
            cur_stencil = upwind_stencils[i]
            for idx in 1:A.boundary_stencil_length
                xtempi += cur_coeff*cur_stencil[idx]*x[x_len-A.boundary_stencil_length+idx]
            end
        end
        x_temp[x_temp_len-A.boundary_point_count+i] = xtempi + !overwrite*x_temp[x_temp_len-A.boundary_point_count+i]
    end
end

################################################################################
# Non-uniform grid Upwind convolutions
################################################################################

function convolve_interior!(x_temp::AbstractVector{T}, x::AbstractVector{T}, A::DerivativeOperator{T,N,true,M}; overwrite = true) where {T<:Real,N,M<:AbstractArray{T}}
    @assert length(x_temp)+2 == length(x)

    len = A.len                         #
    bpc = A.boundary_point_count        #
    stl = A.stencil_length              #
    bstl = A.boundary_stencil_length
    coeff   = A.coefficients            #

    for i in bpc+1:len-bpc
        cur_coeff   = coeff[i]
        if cur_coeff >= 0
            xtempi = zero(T)
            cur_stencil = A.stencil_coefs[1,i-bpc]
            for idx in 1:stl
                xtempi += cur_coeff * cur_stencil[idx]*x[i+idx]
            end
            x_temp[i] = xtempi + !overwrite*x_temp[i]
        else
            xtempi = zero(T)
            cur_stencil = A.stencil_coefs[2,i-bpc]
            for idx in 1:stl
                xtempi += cur_coeff * cur_stencil[idx]*x[i-stl+1+idx]
            end
            x_temp[i] = xtempi + !overwrite*x_temp[i]
        end
    end
end

###########################################

# Against A BC-padded vector, specialize the computation to explicitly use the left, right, and middle parts
function convolve_interior!(x_temp::AbstractVector{T}, _x::BoundaryPaddedVector, A::DerivativeOperator{T,N,false}; overwrite = true) where {T<:Real, N}
    stencil = A.stencil_coefs
    coeff   = A.coefficients
    x = _x.u
    mid = div(A.stencil_length,2) + 1
    # Just do the middle parts
    for i in (2+A.boundary_point_count) : (length(x_temp)-A.boundary_point_count)-1
        xtempi = zero(T)
        cur_stencil = eltype(stencil) <: AbstractVector ? stencil[i-A.boundary_point_count] : stencil
        cur_coeff   = typeof(coeff)   <: AbstractVector ? coeff[i-A.boundary_point_count] : coeff isa Number ? coeff : true
        @inbounds for idx in 1:A.stencil_length
            xtempi += cur_coeff * cur_stencil[idx] * x[(i-1) - (mid-idx) + 1]
        end
        x_temp[i] = xtempi + !overwrite*x_temp[i]
    end
end

function convolve_BC_left!(x_temp::AbstractVector{T}, _x::BoundaryPaddedVector, A::DerivativeOperator{T,N,false}; overwrite = true) where {T<:Real, N}
    stencil = A.low_boundary_coefs
    coeff   = A.coefficients
    for i in 1 : A.boundary_point_count
        cur_stencil = stencil[i]
        cur_coeff   = typeof(coeff)   <: AbstractVector ? coeff[i] : coeff isa Number ? coeff : true
        xtempi = cur_coeff*cur_stencil[1]*_x.l
        @inbounds for idx in 2:A.boundary_stencil_length
            xtempi += cur_coeff * cur_stencil[idx] * _x.u[idx-1]
        end
        x_temp[i] = xtempi + !overwrite*x_temp[i]
    end
    # need to account for x.l in first interior
    mid = div(A.stencil_length,2) + 1
    x = _x.u
    i = 1 + A.boundary_point_count
    xtempi = zero(T)
    cur_stencil = eltype(A.stencil_coefs) <: AbstractVector ? A.stencil_coefs[i-A.boundary_point_count] : A.stencil_coefs
    cur_coeff   = typeof(coeff)   <: AbstractVector ? coeff[i-A.boundary_point_count] : coeff isa Number ? coeff : true
    xtempi = cur_coeff*cur_stencil[1]*_x.l
    @inbounds for idx in 2:A.stencil_length
        xtempi += cur_coeff * cur_stencil[idx] * x[(i-1) - (mid-idx) + 1]
    end
    x_temp[i] = xtempi + !overwrite*x_temp[i]
end

function convolve_BC_right!(x_temp::AbstractVector{T}, _x::BoundaryPaddedVector, A::DerivativeOperator{T,N,false}; overwrite = true) where {T<:Real, N}
    stencil = A.high_boundary_coefs
    coeff   = A.coefficients
    bc_start = length(_x.u) - A.boundary_point_count
    # need to account for _x.r in last interior convolution
    mid = div(A.stencil_length,2) + 1
    x = _x.u
    i = length(x_temp)-A.boundary_point_count
    xtempi = zero(T)
    cur_stencil = eltype(A.stencil_coefs) <: AbstractVector ? A.stencil_coefs[i-A.boundary_point_count] : A.stencil_coefs
    cur_coeff   = typeof(coeff)   <: AbstractVector ? coeff[i-A.boundary_point_count] : coeff isa Number ? coeff : true
    xtempi = cur_coeff*cur_stencil[end]*_x.r
    @inbounds for idx in 1:A.stencil_length-1
        xtempi += cur_coeff * cur_stencil[idx] * x[(i-1) - (mid-idx) + 1]
    end
    x_temp[i] = xtempi + !overwrite*x_temp[i]
    for i in 1 : A.boundary_point_count
        cur_stencil = stencil[i]
        cur_coeff   = typeof(coeff)   <: AbstractVector ? coeff[bc_start + i] : coeff isa Number ? coeff : true
        xtempi = cur_coeff*cur_stencil[end]*_x.r
        @inbounds for idx in A.stencil_length:-1:1
            xtempi += cur_coeff * cur_stencil[end-idx] * _x.u[end-idx+1]
        end
        x_temp[bc_start + i] = xtempi + !overwrite*x_temp[bc_start + i]
    end
end


# Against A BC-padded vector, specialize the computation to explicitly use the left, right, and middle parts
function convolve_interior!(x_temp::AbstractVector{T}, _x::BoundaryPaddedVector, A::DerivativeOperator{T,N,true}; overwrite = true) where {T<:Real,N}
    @assert length(x_temp) == length(_x.u)
    stencil = A.stencil_coefs
    coeff   = A.coefficients
    x_len = length(x_temp) + 2
    # Upwind operators have a non-centred stencil
    mid = 1 + A.stencil_length%2
    x = _x.u
    # Just do the middle parts
    for i in (1+A.boundary_point_count) : (length(x_temp)-A.boundary_point_count)
        xtempi = zero(T)
        cur_stencil = eltype(stencil) <: AbstractVector ? stencil[i-A.boundary_point_count] : stencil
        cur_coeff   = typeof(coeff)   <: AbstractVector ? coeff[i-A.boundary_point_count] : coeff isa Number ? coeff : true
        cur_stencil = cur_coeff < 0 ? reverse(cur_stencil) : cur_stencil
        @inbounds for idx in 1:A.stencil_length
            if i-mid+idx == 1
                x_idx = _x.l
            elseif i-mid+idx == x_len
                x_idx = _x.r
            else
                x_idx = cur_coeff < 0 ? x[i + mid - idx - 1] : x[i - mid + idx - 1]
            end
            xtempi += cur_coeff * cur_stencil[idx] * x_idx
        end
        x_temp[i] = xtempi
    end
end


function convolve_BC_left!(x_temp::AbstractVector{T}, _x::BoundaryPaddedVector, A::DerivativeOperator{T,N,true}) where {T<:Real,N}
    stencil = A.low_boundary_coefs
    coeff   = A.coefficients

    _bpc = A.boundary_point_count
    use_interior_stencil = false

    if isempty(stencil)
        _bpc = 1
        use_interior_stencil = true
    end

    for i in 1 : _bpc
        if use_interior_stencil == true
            cur_stencil = A.stencil_coefs
            slen = length(A.stencil_coefs)
        else
            cur_stencil = stencil[i]
            slen = length(cur_stencil)
        end

        cur_coeff   = typeof(coeff)   <: AbstractVector ? coeff[i] : coeff isa Number ? coeff : true
        cur_stencil = cur_coeff < 0 ? reverse(cur_stencil) : cur_stencil

        # need to account for x.l in first interior
        xtempi = cur_coeff*cur_stencil[1]*_x.l
        @inbounds for idx in 2:slen
            xtempi += cur_coeff * cur_stencil[idx] * _x.u[idx-1]
        end
        x_temp[i] = xtempi + !overwrite*x_temp[i]
    end
end

function convolve_BC_right!(x_temp::AbstractVector{T}, _x::BoundaryPaddedVector, A::DerivativeOperator{T,N,true}; overwrite = true) where {T<:Real,N}
    stencil = A.high_boundary_coefs
    coeff   = A.coefficients
    u_len = length(_x.u)
    bpc = A.boundary_point_count

    # need to account for _x.r in last interior convolution
    x = _x.u
    i = length(x_temp)-A.boundary_point_count
    L = A.boundary_stencil_length

    _bpc = A.boundary_point_count
    use_interior_stencil = false

    if isempty(stencil)
        _bpc = 1
        L = A.stencil_length
        use_interior_stencil = true
    end

    for i in 1 : _bpc
        if use_interior_stencil == true
            cur_stencil = A.stencil_coefs
            slen = length(A.stencil_coefs)
        else
            cur_stencil = stencil[i]
            slen = length(cur_stencil)
        end

        cur_coeff   = typeof(coeff)   <: AbstractVector ? coeff[end-_bpc+i] : coeff isa Number ? coeff : true
        xtempi = cur_coeff*cur_stencil[end]*_x.r
        cur_stencil = cur_coeff < 0 ? reverse(cur_stencil) : cur_stencil

        @inbounds for idx in slen-1:-1:1
            xtempi += cur_coeff * cur_stencil[end-idx] * _x.u[end-idx+1]
        end
        x_temp[end-_bpc+i] = xtempi + !overwrite*x_temp[end-_bpc+i]
    end
end
