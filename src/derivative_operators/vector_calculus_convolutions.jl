# mul! implementations for cases when output/input arrays represent vector elements 

function LinearAlgebra.mul!(x_temp::AbstractArray{T1,N3}, has_vector::Bool, A::DerivativeOperator{T2,N}, M::AbstractArray{T1,N2}; overwrite = false) where {T1,T2,N,N2,N3}
    
    # transformation for convenience of computing derivates along particular axis
    if has_vector

        # For Divergence, input would be a physical vector and output a scalar matrix with one less dimension than input 
        # Check that axis of differentiation is in the dimensions of M and x_temp
        ndims_M = ndims(M)
        @assert N < ndims_M

        alldims = [1:ndims(x_temp);]
        otherdims = setdiff(alldims, N)

        idx = Any[first(ind) for ind in axes(x_temp)]
        nidx = length(otherdims)

        dims_M = [axes(M)[1:end-1]...]
        dims_x_temp = [axes(x_temp)...]
        minimal_padding_indices = map(enumerate(dims_x_temp)) do (dim, val)
            if dim == N || length(dims_x_temp[dim]) == length(dims_M[dim])
                Colon()
            else
                dims_M[dim][begin+1:end-1]
            end
        end
        minimally_padded_M = view(M, minimal_padding_indices...,N)

        itershape = tuple(dims_x_temp[otherdims]...)
        indices = Iterators.drop(CartesianIndices(itershape), 0)

        setindex!(idx, :, N)
        for I in indices
            # replace all elements of idx with corresponding elt of I, except at index N
            Base.replace_tuples!(nidx, idx, idx, otherdims, I)
            mul!(view(x_temp, idx...), true, A, view(minimally_padded_M, idx...), overwrite = false)
        end
    else
        # For Gradient, input would be a scalar matrix and output a phycial vector with an extra dimension      
        # Check that axis of differentiation is in the dimensions of M and x_temp
        ndims_M = ndims(M)
        @assert N <= ndims_M

        alldims = [1:ndims(M);]
        otherdims = setdiff(alldims, N)

        idx = Any[first(ind) for ind in axes(M)]
        nidx = length(otherdims)

        dims_M = [axes(M)...]
        dims_x_temp = [axes(x_temp)[1:end-1]...]
        minimal_padding_indices = map(enumerate(dims_x_temp)) do (dim, val)
            if dim == N || length(dims_x_temp[dim]) == length(dims_M[dim])
                Colon()
            else
                dims_M[dim][begin+1:end-1]
            end
        end
        minimally_padded_M = view(M, minimal_padding_indices...)

        itershape = tuple(dims_x_temp[otherdims]...)
        indices = Iterators.drop(CartesianIndices(itershape), 0)

        setindex!(idx, :, N)
        for I in indices
            # replace all elements of idx with corresponding elt of I, except at index N
            Base.replace_tuples!(nidx, idx, idx, otherdims, I)
            mul!(view(x_temp, idx...,N), false, A, view(minimally_padded_M, idx...), overwrite = false)
        end
    end
end

##################################################################################
# Divergence and Gradient convolutions
##################################################################################

function LinearAlgebra.mul!(x_temp::AbstractVector, has_vector::Bool, A::DerivativeOperator,x::AbstractVector; overwrite = false)
    convolve_BC_left!(x_temp, A, x, overwrite = overwrite)
    convolve_interior!(x_temp, A, x, overwrite = overwrite)
    convolve_BC_right!(x_temp, A, x, overwrite = overwrite)
end

function convolve_interior!(x_temp::AbstractVector{T1},  A::DerivativeOperator{T2}, x::AbstractVector{T1}; overwrite = true) where {T1, T2}
    
    T = promote_type(T1,T2)
    @assert length(x_temp)+2 == length(x)
    stencil = A.stencil_coefs
    coeff   = A.coefficients
    mid = div(A.stencil_length,2)
    for i in (1+A.boundary_point_count) : (length(x_temp)-A.boundary_point_count)
        xtempi = zero(T)
        cur_stencil = eltype(stencil) <: AbstractVector ? stencil[i-A.boundary_point_count] : stencil
        cur_coeff   = typeof(coeff)   <: AbstractVector ? coeff[i] : coeff isa Number ? coeff : true
        for idx in 1:A.stencil_length
            xtempi += cur_coeff * cur_stencil[idx] * x[i - mid + idx]
        end
        x_temp[i] = xtempi + !overwrite*x_temp[i]
    end
end

function convolve_BC_left!(x_temp::AbstractVector{T1}, A::DerivativeOperator{T2}, x::AbstractVector{T1}; overwrite = true) where {T1,T2}

    T = promote_type(T1,T2)
    stencil = A.low_boundary_coefs
    coeff   = A.coefficients
    for i in 1 : A.boundary_point_count
        cur_stencil = stencil[i]
        cur_coeff   = typeof(coeff)   <: AbstractVector ? coeff[i] : coeff isa Number ? coeff : true
        xtempi = zero(T)
        for idx in 1:A.boundary_stencil_length
            xtempi += cur_coeff * cur_stencil[idx] * x[idx]
        end
        x_temp[i] = xtempi + !overwrite*x_temp[i]
    end
end

function convolve_BC_right!(x_temp::AbstractVector{T1}, A::DerivativeOperator{T2}, x::AbstractVector{T1}; overwrite = true) where {T1,T2}

    T = promote_type(T1,T2)
    stencil = A.high_boundary_coefs
    coeff   = A.coefficients
    for i in 1 : A.boundary_point_count
        cur_stencil = stencil[i]
        cur_coeff   = typeof(coeff)   <: AbstractVector ? coeff[i] : coeff isa Number ? coeff : true
        xtempi = zero(T)
        for idx in (A.boundary_stencil_length-1):-1:0
            xtempi += cur_coeff * cur_stencil[end-idx] * x[end-idx]
        end
        x_temp[end-A.boundary_point_count+i] = xtempi + !overwrite*x_temp[end-A.boundary_point_count+i]
    end
end

##################################################################################
# Curl convolutions
##################################################################################

# Against a standard vector, assume already padded and just apply the stencil

function LinearAlgebra.mul!(x_temp::AbstractArray{T,4}, A::CurlOperator, x::AbstractArray{T,4}; overwrite = false) where {T<:Number}
    convolve_BC_left!(x_temp, x, A, overwrite = overwrite)
    convolve_interior!(x_temp, x, A, overwrite = overwrite)
    convolve_BC_right!(x_temp, x, A, overwrite = overwrite)
end

function convolve_interior!(x_temp::AbstractArray{T,4}, u::AbstractArray{T,4}, A::CurlOperator; overwrite = false) where {T<:Number}
    
    s = size(x_temp)
    stencil_1 = A.ops[1].stencil_coefs
    stencil_2 = A.ops[2].stencil_coefs
    stencil_3 = A.ops[3].stencil_coefs

    coeff_1   = A.ops[1].coefficients
    coeff_2   = A.ops[2].coefficients
    coeff_3   = A.ops[3].coefficients

    bpc = A.ops[1].boundary_point_count
    mid = div(A.ops[1].stencil_length,2)
      
    # Compute derivatives in particular dimensions and aggregate the outputs

    # Along 1st axis
    for i in 1+bpc:s[1]-bpc, j in 1:s[2], k in 1:s[3] 
        cur_stencil_1 = eltype(stencil_1) <: AbstractArray ? stencil_1[i-bpc] : stencil_1
        cur_coeff_1   = typeof(coeff_1)   <: AbstractArray ? coeff_1[i] : coeff_1 isa Number ? coeff_1 : true
        for idx in 1:(A.ops[1].stencil_length)
            x_temp[i,j,k,2] += -cur_coeff_1*cur_stencil_1[idx]*u[i+idx-mid,j,k,3]
            x_temp[i,j,k,3] += cur_coeff_1*cur_stencil_1[idx]*u[i+idx-mid,j,k,2]
        end
    end

    # Along 2nd axis
    for i in 1:s[1] , j in 1+bpc : s[2]-bpc , k in 1:s[3]
        cur_stencil_2 = eltype(stencil_2) <: AbstractArray ? stencil_2[j-bpc] : stencil_2
        cur_coeff_2   = typeof(coeff_2)   <: AbstractArray ? coeff_2[j] : coeff_2 isa Number ? coeff_2 : true
        for idx in 1:(A.ops[2].stencil_length)
            x_temp[i,j,k,1] += cur_coeff_2*cur_stencil_2[idx]*u[i,j+idx-mid,k,3]
            x_temp[i,j,k,3] += -cur_coeff_2*cur_stencil_2[idx]*u[i,j+idx-mid,k,1]
        end
    end

    # Along 3rd axis
    for i in 1 : s[1] , j in 1 : s[2] , k in 1+bpc : s[3]-bpc
        cur_stencil_3 = eltype(stencil_3) <: AbstractArray ? stencil_3[k-bpc] : stencil_3
        cur_coeff_3   = typeof(coeff_3)   <: AbstractArray ? coeff_3[k] : coeff_3 isa Number ? coeff_3 : true
        for idx in 1:(A.ops[3].stencil_length)
            x_temp[i,j,k,1] += -cur_coeff_3*cur_stencil_3[idx]*u[i,j,k+idx-mid,2]
            x_temp[i,j,k,2] += cur_coeff_3*cur_stencil_3[idx]*u[i,j,k+idx-mid,1]
        end
    end
end

function convolve_BC_left!(x_temp::AbstractArray{T,4}, u::AbstractArray{T,4}, A::CurlOperator; overwrite = false) where {T<:Number}
    
    s = size(x_temp)
    stencil_1 = A.ops[1].low_boundary_coefs
    stencil_2 = A.ops[2].low_boundary_coefs
    stencil_3 = A.ops[3].low_boundary_coefs

    coeff_1   = A.ops[1].coefficients
    coeff_2   = A.ops[2].coefficients
    coeff_3   = A.ops[3].coefficients

    bpc = A.ops[1].boundary_point_count

    # Compute derivatives in particular dimensions and aggregate the outputs
    for i in 1 : bpc , j in 1:s[2] , k in 1:s[3]
        cur_stencil_1 = stencil_1[i]
        cur_coeff_1   = typeof(coeff_1)   <: AbstractArray ? coeff_1[i] : coeff_1 isa Number ? coeff_1 : true
        for idx in 1:(A.ops[1].stencil_length)
            x_temp[i,j,k,2] += -cur_coeff_1*cur_stencil_1[idx]*u[idx,j,k,3]
            x_temp[i,j,k,3] += cur_coeff_1*cur_stencil_1[idx]*u[idx,j,k,2]
        end
    end

    for i in 1:s[1] , j in 1:bpc , k in 1:s[3]
        cur_stencil_2 = stencil_2[j]
        cur_coeff_2   = typeof(coeff_2)   <: AbstractArray ? coeff_2[j] : coeff_2 isa Number ? coeff_2 : true
        for idx in 1:(A.ops[2].stencil_length)
            x_temp[i,j,k,1] += cur_coeff_2*cur_stencil_2[idx]*u[i,idx,k,3]
            x_temp[i,j,k,3] += -cur_coeff_2*cur_stencil_2[idx]*u[i,idx,k,1]
        end
    end

    for i in 1:s[1] , j in 1:s[2] , k in 1:bpc
        cur_stencil_3 = stencil_3[k]
        cur_coeff_3   = typeof(coeff_3)   <: AbstractArray ? coeff_3[k] : coeff_3 isa Number ? coeff_3 : true
        for idx in 1:(A.ops[1].stencil_length)
            x_temp[i,j,k,1] += -cur_coeff_3*cur_stencil_3[idx]*u[i,j,idx,2]
            x_temp[i,j,k,2] += cur_coeff_3*cur_stencil_3[idx]*u[i,j,idx,1]
        end
    end
end

function convolve_BC_right!(x_temp::AbstractArray{T,4}, u::AbstractArray{T,4}, A::CurlOperator; overwrite = false) where {T<:Number}
    
    s = size(x_temp)
    stencil_1 = A.ops[1].high_boundary_coefs
    stencil_2 = A.ops[2].high_boundary_coefs
    stencil_3 = A.ops[3].high_boundary_coefs

    coeff_1   = A.ops[1].coefficients
    coeff_2   = A.ops[2].coefficients
    coeff_3   = A.ops[3].coefficients

    bpc = A.ops[1].boundary_point_count
    bstl = A.ops[1].boundary_stencil_length

    # Compute derivatives in particular dimensions and aggregate the outputs
    for i in s[1]-bpc+1 : s[1] , j in 1 : s[2] , k in 1 : s[3]
        cur_stencil_1 = stencil_1[i - s[1] + bpc]
        cur_coeff_1   = typeof(coeff_1)   <: AbstractArray ? coeff_1[i] : coeff_1 isa Number ? coeff_1 : true
        for idx in 1:bstl
            x_temp[i,j,k,2] += -cur_coeff_1*cur_stencil_1[idx]*u[s[1]-bstl+idx+2,j,k,3]
            x_temp[i,j,k,3] += cur_coeff_1*cur_stencil_1[idx]*u[s[1]-bstl+idx+2,j,k,2]
        end
    end

    for i in 1 : s[1] , j in s[2]-bpc+1 : s[2] , k in 1 : s[3]
        cur_stencil_2 = stencil_2[j - s[2] + bpc]
        cur_coeff_2   = typeof(coeff_2)   <: AbstractArray ? coeff_2[j] : coeff_2 isa Number ? coeff_2 : true
        for idx in 1:bstl
            x_temp[i,j,k,1] += cur_coeff_2*cur_stencil_2[idx]*u[i,s[2]-bstl+idx+2,k,3]
            x_temp[i,j,k,3] += -cur_coeff_2*cur_stencil_2[idx]*u[i,s[2]-bstl+idx+2,k,1]
        end
    end

    for i in 1 : s[1] , j in 1 : s[2] , k in s[3]-bpc+1 : s[3]
        cur_stencil_3 = stencil_3[k - s[3] + bpc]
        cur_coeff_3   = typeof(coeff_3)   <: AbstractArray ? coeff_3[k] : coeff_3 isa Number ? coeff_3 : true
        for idx in 1:bstl
            x_temp[i,j,k,1] += -cur_coeff_3*cur_stencil_3[idx]*u[i,j,s[3]-bstl+idx+2,2]
            x_temp[i,j,k,2] += cur_coeff_3*cur_stencil_3[idx]*u[i,j,s[3]-bstl+idx+2,1]
        end
    end
end