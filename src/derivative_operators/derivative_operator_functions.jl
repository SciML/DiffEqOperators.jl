function LinearAlgebra.mul!(x_temp::AbstractArray{T}, A::DerivativeOperator{T,N}, M::AbstractArray{T}) where {T<:Real,N}

    # Check that x_temp has correct dimensions
    v = zeros(ndims(x_temp))
    v[N] = 2
    @assert [size(x_temp)...]+v == [size(M)...]

    # Check that axis of differentiation is in the dimensions of M and x_temp
    ndimsM = ndims(M)
    @assert N <= ndimsM

    dimsM = [axes(M)...]
    alldims = [1:ndims(M);]
    otherdims = setdiff(alldims, N)

    idx = Any[first(ind) for ind in axes(M)]
    itershape = tuple(dimsM[otherdims]...)
    nidx = length(otherdims)
    indices = Iterators.drop(CartesianIndices(itershape), 0)

    setindex!(idx, :, N)
    for I in indices
        Base.replace_tuples!(nidx, idx, idx, otherdims, I)
        mul!(view(x_temp, idx...), A, view(M, idx...))
    end
end

for MT in [2,3]
    @eval begin
        function LinearAlgebra.mul!(x_temp::AbstractArray{T,$MT}, A::DerivativeOperator{T,N,Wind,T2,S1}, M::AbstractArray{T,$MT}) where {T<:Real,N,Wind,T2,SL,S1<:SArray{Tuple{SL},T,1,SL}}

            # Check that x_temp has correct dimensions
            v = zeros(ndims(x_temp))
            v[N] = 2
            @assert [size(x_temp)...]+v == [size(M)...]

            # Check that axis of differentiation is in the dimensions of M and x_temp
            ndimsM = ndims(M)
            @assert N <= ndimsM

            # Respahe x_temp for NNlib.conv!
            new_size = Any[size(x_temp)...]
            bpc = A.boundary_point_count
            setindex!(new_size, new_size[N]- 2*bpc, N)
            new_shape = []
            for i in 1:ndimsM
                if i != N
                    push!(new_shape,:)
                else
                    push!(new_shape,bpc+1:new_size[N]+bpc)
                end
             end
             _x_temp = reshape(view(x_temp, new_shape...), (new_size...,1,1))

            # Reshape M for NNlib.conv!
            _M = reshape(M, (size(M)...,1,1))
            s = A.stencil_coefs
            sl = A.stencil_length

            # Setup W, the kernel for NNlib.conv!
            Wdims = ones(Int64, ndims(_x_temp))
            Wdims[N] = sl
            W = zeros(Wdims...)
            Widx = Any[Wdims...]
            setindex!(Widx,:,N)
            W[Widx...] = s ./ A.dx^A.derivative_order # this will change later 
            cv = DenseConvDims(_M, W)

            conv!(_x_temp, _M, W, cv)

            # Now deal with boundaries
            dimsM = [axes(M)...]
            alldims = [1:ndims(M);]
            otherdims = setdiff(alldims, N)

            idx = Any[first(ind) for ind in axes(M)]
            itershape = tuple(dimsM[otherdims]...)
            nidx = length(otherdims)
            indices = Iterators.drop(CartesianIndices(itershape), 0)

            setindex!(idx, :, N)
            for I in indices
                Base.replace_tuples!(nidx, idx, idx, otherdims, I)
                convolve_BC_left!(view(x_temp, idx...), view(M, idx...), A)
                convolve_BC_right!(view(x_temp, idx...), view(M, idx...), A)
            end
        end
    end
end

function *(A::DerivativeOperator{T,N},M::AbstractArray{T}) where {T<:Real,N}
    size_x_temp = [size(M)...]
    size_x_temp[N] -= 2
    x_temp = zeros(promote_type(eltype(A),eltype(M)), size_x_temp...)
    LinearAlgebra.mul!(x_temp, A, M)
    return x_temp
end
