# mul! implementation for the case when output array contains vector elements 

function LinearAlgebra.mul!(x_temp::AbstractArray{Array{T,1}, N2}, A::DerivativeOperator{T,N}, M::AbstractArray{T,N2}; overwrite = true) where {T,N,N2}

    # Check that x_temp has valid dimensions, allowing unnecessary padding in M
    v = zeros(ndims(x_temp))
    v .= 2
    @assert all(([size(x_temp)...] .== [size(M)...])
        .| (([size(x_temp)...] .+ v) .== [size(M)...])
        )

    # Check that axis of differentiation is in the dimensions of M and x_temp
    ndims_M = ndims(M)
    @assert N <= ndims_M
    @assert size(x_temp, N) + 2 == size(M, N) # differentiated dimension must be padded

    alldims = [1:ndims(M);]
    otherdims = setdiff(alldims, N)

    idx = Any[first(ind) for ind in axes(M)]
    nidx = length(otherdims)

    dims_M = [axes(M)...]
    dims_x_temp = [axes(x_temp)...]
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
        mul!(view(x_temp, idx...), A, view(minimally_padded_M, idx...), overwrite = true)
    end
end