
# mul! implementations for cases when output/input arrays represent vector elements 
function LinearAlgebra.mul!(x_temp::AbstractArray, has_vector::Bool, A::Union{GradientOperator, DivergenceOperator}, M::AbstractArray; overwrite = false)
    for L in A.ops
        mul!(x_temp, has_vector, L, M)
    end
end

function LinearAlgebra.mul!(x_temp::AbstractArray, has_vector::Bool, A::DerivativeOperator{T,N}, M::AbstractArray; overwrite = false) where {T,N}
    
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
# Divergence and Gradient convolutions for general N-dim functions/vectors 
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
    coeff = A.coefficients

    # Initialize cur_stencil so that LoopVectorization.check_args(curr_stencil) doesn't throw undef variable for cur_stencil
    cur_stencil = 0

    mid = div(A.stencil_length,2)
    if eltype(stencil) <: AbstractVector
        @turbo for i in (1+A.boundary_point_count) : (length(x_temp)-A.boundary_point_count)
            xtempi = zero(T)
            cur_stencil = stencil[i-A.boundary_point_count]
            cur_coeff   = coeff[i]
            for idx in 1:A.stencil_length
                xtempi += cur_coeff * cur_stencil[idx] * x[i - mid + idx]
            end
            x_temp[i] = xtempi + !overwrite*x_temp[i]
        end
    else
        @turbo for i in (1+A.boundary_point_count) : (length(x_temp)-A.boundary_point_count)
            xtempi = zero(T)
            cur_coeff   = coeff[i]
            for idx in 1:A.stencil_length
                xtempi += cur_coeff * stencil[idx] * x[i - mid + idx]
            end
            x_temp[i] = xtempi + !overwrite*x_temp[i]
        end
    end
end

function convolve_BC_left!(x_temp::AbstractVector{T1}, A::DerivativeOperator{T2}, x::AbstractVector{T1}; overwrite = true) where {T1,T2}

    T = promote_type(T1,T2)
    stencil = A.low_boundary_coefs
    coeff   = A.coefficients

    # Initialize cur_stencil so that LoopVectorization.check_args(curr_stencil) doesn't throw undef variable for cur_stencil
    cur_stencil = 0

    @turbo for i in 1 : A.boundary_point_count
        cur_stencil = stencil[i]
        cur_coeff   = coeff[i]
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

    # Initialize cur_stencil so that LoopVectorization.check_args(curr_stencil) doesn't throw undef variable for cur_stencil
    cur_stencil = 0

    @turbo for i in 1 : A.boundary_point_count
        cur_stencil = stencil[i]
        cur_coeff   = coeff[i]
        xtempi = zero(T)
        for idx in (A.boundary_stencil_length-1):-1:0
            xtempi += cur_coeff * cur_stencil[end-idx] * x[end-idx]
        end
        x_temp[end-A.boundary_point_count+i] = xtempi + !overwrite*x_temp[end-A.boundary_point_count+i]
    end
end

#########################################################################################
# More efficient Divergence and Gradient convolutions for 2D & 3D functions/vectors 
#########################################################################################

for MT in [2,3]
    @eval begin
        function LinearAlgebra.mul!(x_temp::AbstractArray, has_vector::Bool, A::Union{GradientOperator{T,$MT}, DivergenceOperator{T,$MT}}, M::AbstractArray; overwrite = false) where {T}
            convolve_BC_left!(x_temp, A, M, overwrite = overwrite)
            convolve_interior!(x_temp, A, M, overwrite = overwrite)
            convolve_BC_right!(x_temp, A, M, overwrite = overwrite)
        end
    end
end

####### Gradient Convolutions for  2D functions #######
function convolve_interior!(x_temp::AbstractArray{T1}, A::GradientOperator{T2,2}, x::AbstractArray{T3}; overwrite = false) where {T1,T2,T3}
    
    T = promote_type(T1,T2,T3)
    s = size(x_temp)
    bpc = A.ops[1].boundary_point_count
    mid = div(A.ops[1].stencil_length,2)

    stencil_1 = A.ops[1].stencil_coefs
    stencil_2 = A.ops[2].stencil_coefs

    # Initialize cur_stencil so that LoopVectorization.check_args(curr_stencil) doesn't throw undef variable for cur_stencil
    cur_stencil_1 = 0 
    cur_stencil_2 = 0

    coeff_1 = A.ops[1].coefficients
    coeff_2 = A.ops[2].coefficients
      
    # Compute derivatives along particular axis and aggregate the outputs

    if eltype(stencil_1) <: AbstractArray
        # Along 1st axis
        @turbo for j in 1:s[2], i in 1+bpc:s[1]-bpc 
            cur_stencil_1 = stencil_1[i-bpc]
            cur_coeff_1   = coeff_1[i]
            x_temp1 = zero(T)
                for idx in 1:(A.ops[1].stencil_length)
                    x_temp1 += cur_coeff_1 * cur_stencil_1[idx] * x[i+idx-mid,j]
                end
                x_temp[i,j,1] += x_temp1
        end
        # Along 2nd axis
        @turbo for j in 1+bpc : s[2]-bpc , i in 1:s[1]  
            cur_stencil_2 = stencil_2[j-bpc]
            cur_coeff_2   = coeff_2[j]
            x_temp2 = zero(T)
            for idx in 1:(A.ops[2].stencil_length)
                x_temp2 += cur_coeff_2 * cur_stencil_2[idx] * x[i,j+idx-mid]
            end
            x_temp[i,j,2] += x_temp2
        end
    else
        # Along 1st axis
        @turbo for j in 1:s[2], i in 1+bpc:s[1]-bpc 
            cur_coeff_1   = coeff_1[i]
            x_temp1 = zero(T)
                for idx in 1:(A.ops[1].stencil_length)
                    x_temp1 += cur_coeff_1 * stencil_1[idx] * x[i+idx-mid,j]
                end
            x_temp[i,j,1] += x_temp1

        end
        # Along 2nd axis
        @turbo for j in 1+bpc : s[2]-bpc , i in 1:s[1]  
            cur_coeff_2   = coeff_2[j]
            x_temp2 = zero(T)
                for idx in 1:(A.ops[2].stencil_length)
                    x_temp2 += cur_coeff_2 * stencil_2[idx] * x[i,j+idx-mid]
                end
                x_temp[i,j,2] += x_temp2
        end
    end
end

function convolve_BC_left!(x_temp::AbstractArray{T1}, A::GradientOperator{T2,2}, x::AbstractArray{T3}; overwrite = false) where {T1,T2,T3}

    T = promote_type(T1,T2,T3)
    s = size(x_temp)
    stencil_1 = A.ops[1].low_boundary_coefs
    stencil_2 = A.ops[2].low_boundary_coefs

    # Initialize cur_stencil so that LoopVectorization.check_args(curr_stencil) doesn't throw undef variable for cur_stencil
    cur_stencil_1 = 0 
    cur_stencil_2 = 0

    coeff_1 = A.ops[1].coefficients
    coeff_2 = A.ops[2].coefficients

    bpc = A.ops[1].boundary_point_count

    # Compute derivatives along particular axis and aggregate the outputs

    # Along 1st axis
    @turbo for j in 1:s[2], i in 1:bpc
        cur_stencil_1 = stencil_1[i]
        cur_coeff_1   = coeff_1[i]
        x_temp1 = zero(T)
        for idx in 1:(A.ops[1].boundary_stencil_length)
            x_temp1 += cur_coeff_1 * cur_stencil_1[idx] * x[idx,j]
        end
        x_temp[i,j,1] += x_temp1
    end
    # Along 2nd axis
    @turbo for j in 1:bpc , i in 1:s[1]  
        cur_stencil_2 = stencil_2[j]
        cur_coeff_2   = coeff_2[j]
        x_temp2 = zero(T)
        for idx in 1:(A.ops[2].boundary_stencil_length)
            x_temp2 += cur_coeff_2 * cur_stencil_2[idx] * x[i,idx]
        end
        x_temp[i,j,2] += x_temp2
    end
end

function convolve_BC_right!(x_temp::AbstractArray{T1}, A::GradientOperator{T2,2}, x::AbstractArray{T3}; overwrite = false) where {T1,T2,T3}

    T = promote_type(T1,T2,T3)
    s = size(x_temp)
    stencil_1 = A.ops[1].high_boundary_coefs
    stencil_2 = A.ops[2].high_boundary_coefs

    # Initialize cur_stencil so that LoopVectorization.check_args(curr_stencil) doesn't throw undef variable for cur_stencil
    cur_stencil_1 = 0 
    cur_stencil_2 = 0

    coeff_1 = A.ops[1].coefficients
    coeff_2 = A.ops[2].coefficients

    bpc = A.ops[1].boundary_point_count
    bstl = A.ops[1].boundary_stencil_length

    # Compute derivatives along particular axis and aggregate the outputs
    # Along 1st axis
    @turbo for j in 1:s[2], i in s[1]-bpc+1 : s[1]
        cur_stencil_1 = stencil_1[i - s[1] + bpc]
        cur_coeff_1   = coeff_1[i]
        x_temp1 = zero(T)
        for idx in 1:bstl
            x_temp1 += cur_coeff_1 * cur_stencil_1[idx] * x[s[1]-bstl+idx+2,j]
        end
        x_temp[i,j,1] += x_temp1
    end
    # Along 2nd axis
    @turbo for j in s[2]-bpc+1 : s[2] , i in 1:s[1]  
        cur_stencil_2 = stencil_2[j - s[2] + bpc]
        cur_coeff_2   = coeff_2[j]
        x_temp2 = zero(T)
        for idx in 1:bstl
            x_temp2 += cur_coeff_2 * cur_stencil_2[idx] * x[i,s[2]-bstl+idx+2]
        end
        x_temp[i,j,2] += x_temp2
    end
end

####### Divergence Convolutions for  2D Vectors #######
function convolve_interior!(x_temp::AbstractArray{T1}, A::DivergenceOperator{T2,2}, x::AbstractArray{T3}; overwrite = false) where {T1,T2,T3}
    
    T = promote_type(T1,T2,T3)
    s = size(x_temp)
    bpc = A.ops[1].boundary_point_count
    mid = div(A.ops[1].stencil_length,2)

    stencil_1 = A.ops[1].stencil_coefs
    stencil_2 = A.ops[2].stencil_coefs
    cur_stencil_1 = 0 
    cur_stencil_2 = 0

    coeff_1 = A.ops[1].coefficients
    coeff_2 = A.ops[2].coefficients
      
    # Compute derivatives along particular axis and aggregate the outputs

    if eltype(stencil_1) <: AbstractArray
        
        # Along 1st axis
        @turbo for j in 1:s[2], i in 1+bpc:s[1]-bpc 
            cur_stencil_1 = stencil_1[i-bpc]
            cur_coeff_1   = coeff_1[i]
            x_temp1 = zero(T)
            for idx in 1:(A.ops[1].stencil_length)
                x_temp1 += cur_coeff_1 * cur_stencil_1[idx] * x[i+idx-mid,j,1]
            end
            x_temp[i,j] += x_temp1
        end
        # Along 2nd axis
        @turbo for j in 1+bpc : s[2]-bpc , i in 1:s[1]  
            cur_stencil_2 = stencil_2[j-bpc]
            cur_coeff_2   = coeff_2[j]
            x_temp2 = zero(T)
            for idx in 1:(A.ops[2].stencil_length)
                x_temp2 += cur_coeff_2 * cur_stencil_2[idx] * x[i,j+idx-mid,2]
            end
            x_temp[i,j] += x_temp2
        end
    else
        # Along 1st axis
        @turbo for j in 1:s[2], i in 1+bpc:s[1]-bpc 
            cur_coeff_1   = coeff_1[i]
            x_temp1 = zero(T)
                for idx in 1:(A.ops[1].stencil_length)
                    x_temp1 += cur_coeff_1 * stencil_1[idx] * x[i+idx-mid,j,1]
                end
            x_temp[i,j] += x_temp1

        end
        # Along 2nd axis
        @turbo for j in 1+bpc : s[2]-bpc , i in 1:s[1]  
            cur_coeff_2   = coeff_2[j]
            x_temp2 = zero(T)
                for idx in 1:(A.ops[2].stencil_length)
                    x_temp2 += cur_coeff_2 * stencil_2[idx] * x[i,j+idx-mid,2]
                end
                x_temp[i,j] += x_temp2
        end
    end
end

function convolve_BC_left!(x_temp::AbstractArray{T1}, A::DivergenceOperator{T2,2}, x::AbstractArray{T3}; overwrite = false) where {T1,T2,T3}

    T = promote_type(T1,T2,T3)
    s = size(x_temp)
    stencil_1 = A.ops[1].low_boundary_coefs
    stencil_2 = A.ops[2].low_boundary_coefs
    cur_stencil_1 = 0 
    cur_stencil_2 = 0

    coeff_1 = A.ops[1].coefficients
    coeff_2 = A.ops[2].coefficients

    bpc = A.ops[1].boundary_point_count

    # Compute derivatives along particular axis and aggregate the outputs
    # Along 1st axis
    @turbo for j in 1:s[2], i in 1:bpc 
        cur_stencil_1 = stencil_1[i]
        cur_coeff_1   = coeff_1[i]
        x_temp1 = zero(T)
        for idx in 1:(A.ops[1].boundary_stencil_length)
            x_temp1 += cur_coeff_1 * cur_stencil_1[idx] * x[idx,j,1]
        end
        x_temp[i,j] += x_temp1
    end
    # Along 2nd axis
    @turbo for j in 1:bpc , i in 1:s[1]  
        cur_stencil_2 = stencil_2[j]
        cur_coeff_2   = coeff_2[j]
        x_temp2 = zero(T)
        for idx in 1:(A.ops[2].boundary_stencil_length)
            x_temp2 += cur_coeff_2 * cur_stencil_2[idx] * x[i,idx,2]
        end
        x_temp[i,j] += x_temp2
    end
end

function convolve_BC_right!(x_temp::AbstractArray{T1}, A::DivergenceOperator{T2,2}, x::AbstractArray{T3}; overwrite = false) where {T1,T2,T3}

    T = promote_type(T1,T2,T3)
    s = size(x_temp)
    stencil_1 = A.ops[1].high_boundary_coefs
    stencil_2 = A.ops[2].high_boundary_coefs
    cur_stencil_1 = 0 
    cur_stencil_2 = 0

    coeff_1 = A.ops[1].coefficients
    coeff_2 = A.ops[2].coefficients

    bpc = A.ops[1].boundary_point_count
    bstl = A.ops[1].boundary_stencil_length

    # Compute derivatives along particular axis and aggregate the outputs

    # Along 1st axis
    @turbo for j in 1:s[2], i in s[1]-bpc+1 : s[1]
        cur_stencil_1 = stencil_1[i - s[1] + bpc]
        cur_coeff_1   = coeff_1[i]
        x_temp1 = zero(T)
        for idx in 1:bstl
            x_temp1 += cur_coeff_1 * cur_stencil_1[idx] * x[s[1]-bstl+idx+2,j,1]
        end
        x_temp[i,j] += x_temp1
    end
    # Along 2nd axis
    @turbo for j in s[2]-bpc+1 : s[2] , i in 1:s[1]  
        cur_stencil_2 = stencil_2[j - s[2] + bpc]
        cur_coeff_2   = coeff_2[j]
        x_temp2 = zero(T)
        for idx in 1:bstl
            x_temp2 += cur_coeff_2 * cur_stencil_2[idx] * x[i,s[2]-bstl+idx+2,2]
        end
        x_temp[i,j] += x_temp2
    end
end


####### Gradient Convolutions for  3D functions #######
function convolve_interior!(x_temp::AbstractArray{T1}, A::GradientOperator{T2,3}, x::AbstractArray{T3}; overwrite = false) where {T1,T2,T3}
    
    T = promote_type(T1,T2,T3)
    s = size(x_temp)
    bpc = A.ops[1].boundary_point_count
    mid = div(A.ops[1].stencil_length,2)

    stencil_1 = A.ops[1].stencil_coefs
    stencil_2 = A.ops[2].stencil_coefs
    stencil_3 = A.ops[3].stencil_coefs
    cur_stencil_1 = 0 
    cur_stencil_2 = 0
    cur_stencil_3 = 0

    coeff_1 = A.ops[1].coefficients
    coeff_2 = A.ops[2].coefficients
    coeff_3 = A.ops[3].coefficients

    # Compute derivatives along particular axis and aggregate the outputs

    if eltype(stencil_1) <: AbstractArray
        # Along 1st axis
        @turbo for k in 1:s[3], j in 1:s[2], i in 1+bpc:s[1]-bpc 
            cur_stencil_1 = stencil_1[i-bpc]
            cur_coeff_1   = coeff_1[i]
            x_temp1 = zero(T)
            for idx in 1:(A.ops[1].stencil_length)
                x_temp1 += cur_coeff_1 * cur_stencil_1[idx] * x[i+idx-mid,j,k]
            end
            x_temp[i,j,k,1] += x_temp1
        end
        # Along 2nd axis
        @turbo for k in 1:s[3], j in 1+bpc : s[2]-bpc , i in 1:s[1]  
            cur_stencil_2 = stencil_2[j-bpc]
            cur_coeff_2   = coeff_2[j]
            x_temp2 = zero(T)
            for idx in 1:(A.ops[2].stencil_length)
                x_temp2 += cur_coeff_2 * cur_stencil_2[idx] * x[i,j+idx-mid,k]
            end
            x_temp[i,j,k,2] += x_temp2
        end
        # Along 3rd axis
        @turbo for k in 1+bpc : s[3]-bpc, j in 1:s[2] , i in 1:s[1]  
            cur_stencil_3 = stencil_3[k-bpc]
            cur_coeff_3   = coeff_3[k]
            x_temp3 = zero(T)
            for idx in 1:(A.ops[3].stencil_length)
                x_temp3 += cur_coeff_3 * cur_stencil_3[idx] * x[i,j,k+idx-mid]
            end
            x_temp[i,j,k,3] += x_temp3
        end
    else
        # Along 1st axis
        @turbo for k in 1:s[3], j in 1:s[2], i in 1+bpc:s[1]-bpc 
            cur_coeff_1   = coeff_1[i]
            x_temp1 = zero(T)
            for idx in 1:(A.ops[1].stencil_length)
                x_temp1 += cur_coeff_1 * stencil_1[idx] * x[i+idx-mid,j,k]
            end
            x_temp[i,j,k,1] += x_temp1
        end
        # Along 2nd axis
        @turbo for k in 1:s[3], j in 1+bpc : s[2]-bpc , i in 1:s[1]  
            cur_coeff_2   = coeff_2[j]
            x_temp2 = zero(T)
            for idx in 1:(A.ops[2].stencil_length)
                x_temp2 += cur_coeff_2 * stencil_2[idx] * x[i,j+idx-mid,k]
            end
            x_temp[i,j,k,2] += x_temp2
        end
        # Along 3rd axis
        @turbo for k in 1+bpc : s[3]-bpc, j in 1:s[2] , i in 1:s[1]  
            cur_coeff_3   = coeff_3[k]
            x_temp3 = zero(T)
            for idx in 1:(A.ops[3].stencil_length)
                x_temp3 += cur_coeff_3 * stencil_3[idx] * x[i,j,k+idx-mid]
            end
            x_temp[i,j,k,3] += x_temp3
        end
    end
end

function convolve_BC_left!(x_temp::AbstractArray{T1}, A::GradientOperator{T2,3}, x::AbstractArray{T3}; overwrite = false) where {T1,T2,T3}

    T = promote_type(T1,T2,T3)
    s = size(x_temp)
    stencil_1 = A.ops[1].low_boundary_coefs
    stencil_2 = A.ops[2].low_boundary_coefs
    stencil_3 = A.ops[3].low_boundary_coefs
    cur_stencil_1 = 0 
    cur_stencil_2 = 0
    cur_stencil_3 = 0

    coeff_1 = A.ops[1].coefficients
    coeff_2 = A.ops[2].coefficients
    coeff_3 = A.ops[3].coefficients

    bpc = A.ops[1].boundary_point_count

    # Compute derivatives along particular axis and aggregate the outputs

    # Along 1st axis
    @turbo for k in 1:s[3], j in 1:s[2], i in 1:bpc 
        cur_stencil_1 = stencil_1[i]
        cur_coeff_1   = coeff_1[i]
        x_temp1 = zero(T)
        for idx in 1:(A.ops[1].boundary_stencil_length)
            x_temp1 += cur_coeff_1 * cur_stencil_1[idx] * x[idx,j,k]
        end
        x_temp[i,j,k,1] += x_temp1
    end
    # Along 2nd axis
    @turbo for k in 1:s[3], j in 1:bpc , i in 1:s[1]  
        cur_stencil_2 = stencil_2[j]
        cur_coeff_2   = coeff_2[j]
        x_temp2 = zero(T)
        for idx in 1:(A.ops[2].boundary_stencil_length)
            x_temp2 += cur_coeff_2 * cur_stencil_2[idx] * x[i,idx,k]
        end
        x_temp[i,j,k,2] += x_temp2
    end
    # Along 3rd axis
    @turbo for k in 1:bpc, j in 1:s[2] , i in 1:s[1]  
        cur_stencil_3 = stencil_3[k]
        cur_coeff_3   = coeff_3[k]
        x_temp3 = zero(T)
        for idx in 1:(A.ops[3].boundary_stencil_length)
            x_temp3 += cur_coeff_3 * cur_stencil_3[idx] * x[i,j,idx]
        end
        x_temp[i,j,k,3] += x_temp3
    end
end

function convolve_BC_right!(x_temp::AbstractArray{T1}, A::GradientOperator{T2,3}, x::AbstractArray{T3}; overwrite = false) where {T1,T2,T3}

    T = promote_type(T1,T2,T3)
    s = size(x_temp)
    stencil_1 = A.ops[1].high_boundary_coefs
    stencil_2 = A.ops[2].high_boundary_coefs
    stencil_3 = A.ops[3].high_boundary_coefs
    cur_stencil_1 = 0 
    cur_stencil_2 = 0
    cur_stencil_3 = 0

    coeff_1 = A.ops[1].coefficients
    coeff_2 = A.ops[2].coefficients
    coeff_3 = A.ops[3].coefficients

    bpc = A.ops[1].boundary_point_count
    bstl = A.ops[1].boundary_stencil_length

    # Compute derivatives along particular axis and aggregate the outputs
    # Along 1st axis
    @turbo for k in 1:s[3], j in 1:s[2], i in s[1]-bpc+1 : s[1]
        cur_stencil_1 = stencil_1[i - s[1] + bpc]
        cur_coeff_1   = coeff_1[i]
        x_temp1 = zero(T)
        for idx in 1:bstl
            x_temp1 += cur_coeff_1 * cur_stencil_1[idx] * x[s[1]-bstl+idx+2,j,k]
        end
        x_temp[i,j,k,1] += x_temp1
    end
    # Along 2nd axis
    @turbo for k in 1:s[3], j in s[2]-bpc+1 : s[2] , i in 1:s[1]  
        cur_stencil_2 = stencil_2[j - s[2] + bpc]
        cur_coeff_2   = coeff_2[j]
        x_temp2 = zero(T)
        for idx in 1:bstl
            x_temp2 += cur_coeff_2 * cur_stencil_2[idx] * x[i,s[2]-bstl+idx+2,k]
        end
        x_temp[i,j,k,2] += x_temp2
    end
    # Along 3rd axis
    @turbo for k in s[3]-bpc+1 : s[3], j in 1:s[2] , i in 1:s[1]  
        cur_stencil_3 = stencil_3[k - s[3] + bpc]
        cur_coeff_3   = coeff_3[k]
        x_temp3 = zero(T)
        for idx in 1:bstl
            x_temp3 += cur_coeff_3 * cur_stencil_3[idx] * x[i,j,s[3]-bstl+idx+2]
        end
        x_temp[i,j,k,3] += x_temp3
    end
end

####### Divergence Convolutions for  3D Vectors #######
function convolve_interior!(x_temp::AbstractArray{T1}, A::DivergenceOperator{T2,3}, x::AbstractArray{T3}; overwrite = false) where {T1,T2,T3}
    
    T = promote_type(T1,T2,T3)
    s = size(x_temp)
    bpc = A.ops[1].boundary_point_count
    mid = div(A.ops[1].stencil_length,2)

    stencil_1 = A.ops[1].stencil_coefs
    stencil_2 = A.ops[2].stencil_coefs
    stencil_3 = A.ops[3].stencil_coefs
    cur_stencil_1 = 0 
    cur_stencil_2 = 0
    cur_stencil_3 = 0

    coeff_1 = A.ops[1].coefficients
    coeff_2 = A.ops[2].coefficients
    coeff_3 = A.ops[3].coefficients

    # Compute derivatives along particular axis and aggregate the outputs

    if eltype(stencil_1) <: AbstractArray
        # Along 1st axis
        @turbo for k in 1:s[3], j in 1:s[2], i in 1+bpc:s[1]-bpc 
            cur_stencil_1 = stencil_1[i-bpc]
            cur_coeff_1   = coeff_1[i]
            x_temp1 = zero(T)
                for idx in 1:(A.ops[1].stencil_length)
                    x_temp1 += cur_coeff_1 * cur_stencil_1[idx] * x[i+idx-mid,j,k,1]
                end
                x_temp[i,j,k] += x_temp1
        end
        # Along 2nd axis
        @turbo for k in 1:s[3], j in 1+bpc : s[2]-bpc , i in 1:s[1]  
            cur_stencil_2 = stencil_2[j-bpc]
            cur_coeff_2   = coeff_2[j]
            x_temp2 = zero(T)
            for idx in 1:(A.ops[2].stencil_length)
                x_temp2 += cur_coeff_2 * cur_stencil_2[idx] * x[i,j+idx-mid,k,2]
            end
            x_temp[i,j,k] += x_temp2
        end
        # Along 3rd axis
        @turbo for k in 1+bpc : s[3]-bpc, j in 1:s[2] , i in 1:s[1]  
            cur_stencil_3 = stencil_3[k-bpc]
            cur_coeff_3   = coeff_3[k]
            x_temp3 = zero(T)
            for idx in 1:(A.ops[3].stencil_length)
                x_temp3 += cur_coeff_3 * cur_stencil_3[idx] * x[i,j,k+idx-mid,3]
            end
            x_temp[i,j,k] += x_temp3
        end
    else
        # Along 1st axis
        @turbo for k in 1:s[3], j in 1:s[2], i in 1+bpc:s[1]-bpc 
            cur_coeff_1   = coeff_1[i]
            x_temp1 = zero(T)
                for idx in 1:(A.ops[1].stencil_length)
                    x_temp1 += cur_coeff_1 * stencil_1[idx] * x[i+idx-mid,j,k,1]
                end
                x_temp[i,j,k] += x_temp1
        end
        # Along 2nd axis
        @turbo for k in 1:s[3], j in 1+bpc : s[2]-bpc , i in 1:s[1]  
            cur_coeff_2   = coeff_2[j]
            x_temp2 = zero(T)
            for idx in 1:(A.ops[2].stencil_length)
                x_temp2 += cur_coeff_2 * stencil_2[idx] * x[i,j+idx-mid,k,2]
            end
            x_temp[i,j,k] += x_temp2
        end
        # Along 3rd axis
        @turbo for k in 1+bpc : s[3]-bpc, j in 1:s[2] , i in 1:s[1]  
            cur_coeff_3   = coeff_3[k]
            x_temp3 = zero(T)
            for idx in 1:(A.ops[3].stencil_length)
                x_temp3 += cur_coeff_3 * stencil_3[idx] * x[i,j,k+idx-mid,3]
            end
            x_temp[i,j,k] += x_temp3
        end
    end
end

function convolve_BC_left!(x_temp::AbstractArray{T1}, A::DivergenceOperator{T2,3}, x::AbstractArray{T3}; overwrite = false) where {T1,T2,T3}

    T = promote_type(T1,T2,T3)
    s = size(x_temp)
    stencil_1 = A.ops[1].low_boundary_coefs
    stencil_2 = A.ops[2].low_boundary_coefs
    stencil_3 = A.ops[3].low_boundary_coefs
    cur_stencil_1 = 0 
    cur_stencil_2 = 0
    cur_stencil_3 = 0

    coeff_1 = A.ops[1].coefficients
    coeff_2 = A.ops[2].coefficients
    coeff_3 = A.ops[3].coefficients

    bpc = A.ops[1].boundary_point_count

    # Compute derivatives along particular axis and aggregate the outputs

    # Along 1st axis
    @turbo for k in 1:s[3], j in 1:s[2], i in 1:bpc 
        cur_stencil_1 = stencil_1[i]
        cur_coeff_1   = coeff_1[i]
        x_temp1 = zero(T)
        for idx in 1:(A.ops[1].boundary_stencil_length)
            x_temp1 += cur_coeff_1 * cur_stencil_1[idx] * x[idx,j,k,1]
        end
        x_temp[i,j,k] += x_temp1
    end
    # Along 2nd axis
    @turbo for k in 1:s[3], j in 1:bpc , i in 1:s[1]  
        cur_stencil_2 = stencil_2[j]
        cur_coeff_2   = coeff_2[j]
        x_temp2 = zero(T)
        for idx in 1:(A.ops[2].boundary_stencil_length)
            x_temp2 += cur_coeff_2 * cur_stencil_2[idx] * x[i,idx,k,2]
        end
        x_temp[i,j,k] += x_temp2
    end
    # Along 3rd axis
    @turbo for k in 1:bpc, j in 1:s[2] , i in 1:s[1]  
        cur_stencil_3 = stencil_3[k]
        cur_coeff_3   = coeff_3[k]
        x_temp3 = zero(T)
        for idx in 1:(A.ops[3].boundary_stencil_length)
            x_temp3 += cur_coeff_3 * cur_stencil_3[idx] * x[i,j,idx,3]
        end
        x_temp[i,j,k] += x_temp3
    end
end

function convolve_BC_right!(x_temp::AbstractArray{T1}, A::DivergenceOperator{T2,3}, x::AbstractArray{T3}; overwrite = false) where {T1,T2,T3}

    T = promote_type(T1,T2,T3)
    s = size(x_temp)
    stencil_1 = A.ops[1].high_boundary_coefs
    stencil_2 = A.ops[2].high_boundary_coefs
    stencil_3 = A.ops[3].high_boundary_coefs
    cur_stencil_1 = 0 
    cur_stencil_2 = 0
    cur_stencil_3 = 0

    coeff_1 = A.ops[1].coefficients
    coeff_2 = A.ops[2].coefficients
    coeff_3 = A.ops[3].coefficients

    bpc = A.ops[1].boundary_point_count
    bstl = A.ops[1].boundary_stencil_length

    # Compute derivatives along particular axis and aggregate the outputs
    # Along 1st axis
    @turbo for k in 1:s[3], j in 1:s[2], i in s[1]-bpc+1 : s[1]
        cur_stencil_1 = stencil_1[i - s[1] + bpc]
        cur_coeff_1   = coeff_1[i]
        x_temp1 = zero(T)
        for idx in 1:bstl
            x_temp1 += cur_coeff_1 * cur_stencil_1[idx] * x[s[1]-bstl+idx+2,j,k,1]
        end
        x_temp[i,j,k] += x_temp1
    end
    # Along 2nd axis
    @turbo for k in 1:s[3], j in s[2]-bpc+1 : s[2] , i in 1:s[1]  
        cur_stencil_2 = stencil_2[j - s[2] + bpc]
        cur_coeff_2   = coeff_2[j]
        x_temp2 = zero(T)
        for idx in 1:bstl
            x_temp2 += cur_coeff_2 * cur_stencil_2[idx] * x[i,s[2]-bstl+idx+2,k,2]
        end
        x_temp[i,j,k] += x_temp2
    end
    # Along 3rd axis
    @turbo for k in s[3]-bpc+1 : s[3], j in 1:s[2] , i in 1:s[1]  
        cur_stencil_3 = stencil_3[k - s[3] + bpc]
        cur_coeff_3   = coeff_3[k]
        x_temp3 = zero(T)
        for idx in 1:bstl
            x_temp3 += cur_coeff_3 * cur_stencil_3[idx] * x[i,j,s[3]-bstl+idx+2,3]
        end
        x_temp[i,j,k] += x_temp3
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
    cur_stencil_1 = 0 
    cur_stencil_2 = 0
    cur_stencil_3 = 0

    coeff_1 = A.ops[1].coefficients
    coeff_2 = A.ops[2].coefficients
    coeff_3 = A.ops[3].coefficients

    bpc = A.ops[1].boundary_point_count
    mid = div(A.ops[1].stencil_length,2)
      
    # Compute derivatives in particular dimensions and aggregate the outputs

    if eltype(stencil_1) <: AbstractArray
        # Along 1st axis
        @turbo for k in 1:s[3], j in 1:s[2], i in 1+bpc:s[1]-bpc 
            cur_stencil_1 = stencil_1[i-bpc]
            cur_coeff_1   = coeff_1[i]
            x_temp2 = zero(T)
            x_temp3 = zero(T)
            for idx in 1:(A.ops[1].stencil_length)
                x_temp2 += -cur_coeff_1*cur_stencil_1[idx]*u[i+idx-mid,j,k,3]
                x_temp3 += cur_coeff_1*cur_stencil_1[idx]*u[i+idx-mid,j,k,2]
            end
            x_temp[i,j,k,3] += x_temp3
            x_temp[i,j,k,2] += x_temp2
        end

        # Along 2nd axis
        @turbo for k in 1:s[3], j in 1+bpc : s[2]-bpc , i in 1:s[1]  
            cur_stencil_2 = stencil_2[j-bpc]
            cur_coeff_2   = coeff_2[j]
            x_temp1 = zero(T)
            x_temp3 = zero(T)
            for idx in 1:(A.ops[2].stencil_length)
                x_temp1 += cur_coeff_2*cur_stencil_2[idx]*u[i,j+idx-mid,k,3]
                x_temp3 += -cur_coeff_2*cur_stencil_2[idx]*u[i,j+idx-mid,k,1]
            end
            x_temp[i,j,k,3] += x_temp3
            x_temp[i,j,k,1] += x_temp1
        end

        # Along 3rd axis
        @turbo for k in 1+bpc : s[3]-bpc, j in 1 : s[2] , i in 1 : s[1]  
            cur_stencil_3 = stencil_3[k-bpc]
            cur_coeff_3   = coeff_3[k]
            x_temp2 = zero(T)
            x_temp1 = zero(T)
            for idx in 1:(A.ops[3].stencil_length)
                x_temp1 += -cur_coeff_3*cur_stencil_3[idx]*u[i,j,k+idx-mid,2]
                x_temp2 += cur_coeff_3*cur_stencil_3[idx]*u[i,j,k+idx-mid,1]
            end
            x_temp[i,j,k,1] += x_temp1
            x_temp[i,j,k,2] += x_temp2
        end
    else
        # Along 1st axis
        @turbo for k in 1:s[3], j in 1:s[2], i in 1+bpc:s[1]-bpc 
            cur_coeff_1   = coeff_1[i]
            x_temp2 = zero(T)
            x_temp3 = zero(T)
            for idx in 1:(A.ops[1].stencil_length)
                x_temp2 += -cur_coeff_1*stencil_1[idx]*u[i+idx-mid,j,k,3]
                x_temp3 += cur_coeff_1*stencil_1[idx]*u[i+idx-mid,j,k,2]
            end
            x_temp[i,j,k,3] += x_temp3
            x_temp[i,j,k,2] += x_temp2
        end

        # Along 2nd axis
        @turbo for k in 1:s[3], j in 1+bpc : s[2]-bpc , i in 1:s[1]  
            cur_coeff_2   = coeff_2[j]
            x_temp1 = zero(T)
            x_temp3 = zero(T)
            for idx in 1:(A.ops[2].stencil_length)
                x_temp1 += cur_coeff_2*stencil_2[idx]*u[i,j+idx-mid,k,3]
                x_temp3 += -cur_coeff_2*stencil_2[idx]*u[i,j+idx-mid,k,1]
            end
            x_temp[i,j,k,3] += x_temp3
            x_temp[i,j,k,1] += x_temp1
        end

        # Along 3rd axis
        @turbo for k in 1+bpc : s[3]-bpc, j in 1 : s[2] , i in 1 : s[1]  
            cur_coeff_3   = coeff_3[k]
            x_temp2 = zero(T)
            x_temp1 = zero(T)
            for idx in 1:(A.ops[3].stencil_length)
                x_temp1 += -cur_coeff_3*stencil_3[idx]*u[i,j,k+idx-mid,2]
                x_temp2 += cur_coeff_3*stencil_3[idx]*u[i,j,k+idx-mid,1]
            end
            x_temp[i,j,k,1] += x_temp1
            x_temp[i,j,k,2] += x_temp2
        end
    end
end

function convolve_BC_left!(x_temp::AbstractArray{T,4}, u::AbstractArray{T,4}, A::CurlOperator; overwrite = false) where {T<:Number}
    
    s = size(x_temp)
    stencil_1 = A.ops[1].low_boundary_coefs
    stencil_2 = A.ops[2].low_boundary_coefs
    stencil_3 = A.ops[3].low_boundary_coefs
    cur_stencil_1 = 0 
    cur_stencil_2 = 0
    cur_stencil_3 = 0

    coeff_1 = A.ops[1].coefficients
    coeff_2 = A.ops[2].coefficients
    coeff_3 = A.ops[3].coefficients

    bpc = A.ops[1].boundary_point_count

    # Compute derivatives in particular dimensions and aggregate the outputs
    @turbo for  k in 1:s[3], j in 1:s[2] , i in 1 : bpc
        cur_stencil_1 = stencil_1[i]
        cur_coeff_1   = coeff_1[i]
        x_temp2 = zero(T)
        x_temp3 = zero(T)
        for idx in 1:(A.ops[1].boundary_stencil_length)
            x_temp2 += -cur_coeff_1*cur_stencil_1[idx]*u[idx,j,k,3]
            x_temp3 += cur_coeff_1*cur_stencil_1[idx]*u[idx,j,k,2]
        end
        x_temp[i,j,k,3] += x_temp3
        x_temp[i,j,k,2] += x_temp2
    end

    @turbo for k in 1:s[3], j in 1:bpc , i in 1:s[1]
        cur_stencil_2 = stencil_2[j]
        cur_coeff_2   = coeff_2[j]
        x_temp1 = zero(T)
        x_temp3 = zero(T)
        for idx in 1:(A.ops[2].boundary_stencil_length)
            x_temp1 += cur_coeff_2*cur_stencil_2[idx]*u[i,idx,k,3]
            x_temp3 += -cur_coeff_2*cur_stencil_2[idx]*u[i,idx,k,1]
        end
        x_temp[i,j,k,1] += x_temp1
        x_temp[i,j,k,3] += x_temp3
    end

    @turbo for k in 1:bpc, j in 1:s[2] , i in 1:s[1]
        cur_stencil_3 = stencil_3[k]
        cur_coeff_3   = coeff_3[k]
        x_temp2 = zero(T)
        x_temp1 = zero(T)
        for idx in 1:(A.ops[1].boundary_stencil_length)
            x_temp1 += -cur_coeff_3*cur_stencil_3[idx]*u[i,j,idx,2]
            x_temp2 += cur_coeff_3*cur_stencil_3[idx]*u[i,j,idx,1]
        end
        x_temp[i,j,k,1] += x_temp1
        x_temp[i,j,k,2] += x_temp2
    end
end

function convolve_BC_right!(x_temp::AbstractArray{T,4}, u::AbstractArray{T,4}, A::CurlOperator; overwrite = false) where {T<:Number}
    
    s = size(x_temp)
    stencil_1 = A.ops[1].high_boundary_coefs
    stencil_2 = A.ops[2].high_boundary_coefs
    stencil_3 = A.ops[3].high_boundary_coefs
    cur_stencil_1 = 0 
    cur_stencil_2 = 0
    cur_stencil_3 = 0

    coeff_1 = A.ops[1].coefficients
    coeff_2 = A.ops[2].coefficients
    coeff_3 = A.ops[3].coefficients

    bpc = A.ops[1].boundary_point_count
    bstl = A.ops[1].boundary_stencil_length

    # Compute derivatives in particular dimensions and aggregate the outputs
    @turbo for k in 1 : s[3], j in 1 : s[2] ,i in s[1]-bpc+1 : s[1]
        cur_stencil_1 = stencil_1[i - s[1] + bpc]
        cur_coeff_1   = coeff_1[i]
        x_temp2 = zero(T)
        x_temp3 = zero(T)
        for idx in 1:bstl
            x_temp2 += -cur_coeff_1*cur_stencil_1[idx]*u[s[1]-bstl+idx+2,j,k,3]
            x_temp3 += cur_coeff_1*cur_stencil_1[idx]*u[s[1]-bstl+idx+2,j,k,2]
        end
        x_temp[i,j,k,3] += x_temp3
        x_temp[i,j,k,2] += x_temp2
    end

    @turbo for k in 1 : s[3], j in s[2]-bpc+1 : s[2] , i in 1 : s[1]
        cur_stencil_2 = stencil_2[j - s[2] + bpc]
        cur_coeff_2   = coeff_2[j]
        x_temp1 = zero(T)
        x_temp3 = zero(T)
        for idx in 1:bstl
            x_temp1 += cur_coeff_2*cur_stencil_2[idx]*u[i,s[2]-bstl+idx+2,k,3]
            x_temp3 += -cur_coeff_2*cur_stencil_2[idx]*u[i,s[2]-bstl+idx+2,k,1]
        end
        x_temp[i,j,k,1] += x_temp1
        x_temp[i,j,k,3] += x_temp3
    end

    @turbo for k in s[3]-bpc+1 : s[3], j in 1 : s[2] , i in 1 : s[1]
        cur_stencil_3 = stencil_3[k - s[3] + bpc]
        cur_coeff_3   = coeff_3[k]
        x_temp1 = zero(T)
        x_temp2 = zero(T)
        for idx in 1:bstl
            x_temp1 += -cur_coeff_3*cur_stencil_3[idx]*u[i,j,s[3]-bstl+idx+2,2]
            x_temp2 += cur_coeff_3*cur_stencil_3[idx]*u[i,j,s[3]-bstl+idx+2,1]
        end
        x_temp[i,j,k,1] += x_temp1
        x_temp[i,j,k,2] += x_temp2
    end
end