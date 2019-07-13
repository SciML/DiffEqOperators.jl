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
        function LinearAlgebra.mul!(x_temp::AbstractArray{T,$MT}, A::DerivativeOperator{T,N,false,T2,S1}, M::AbstractArray{T,$MT}) where {T<:Real,N,T2,SL,S1<:SArray{Tuple{SL},T,1,SL}}

            # Check that x_temp has correct dimensions
            v = zeros(ndims(x_temp))
            v[N] = 2
            @assert [size(x_temp)...]+v == [size(M)...]

            # Check that axis of differentiation is in the dimensions of M and x_temp
            ndimsM = ndims(M)
            @assert N <= ndimsM

            # Determine padding for NNlib.conv!
            bpc = A.boundary_point_count
            pad = zeros(Int64,ndimsM)
            pad[N] = bpc

            # Reshape x_temp for NNlib.conv!
            _x_temp = reshape(x_temp, (size(x_temp)...,1,1))

            # Reshape M for NNlib.conv!
            _M = reshape(M, (size(M)...,1,1))

            # Setup W, the kernel for NNlib.conv!
            s = A.stencil_coefs
            sl = A.stencil_length
            Wdims = ones(Int64, ndims(_x_temp))
            Wdims[N] = sl
            W = zeros(Wdims...)
            Widx = Any[Wdims...]
            setindex!(Widx,:,N)
            W[Widx...] = s

            cv = DenseConvDims(_M, W, padding=pad,flipkernel=true)
            conv!(_x_temp, _M, W, cv)

            # Now deal with boundaries
            if bpc > 0
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
end

function *(A::DerivativeOperator{T,N},M::AbstractArray{T}) where {T<:Real,N}
    size_x_temp = [size(M)...]
    size_x_temp[N] -= 2
    x_temp = zeros(promote_type(eltype(A),eltype(M)), size_x_temp...)
    LinearAlgebra.mul!(x_temp, A, M)
    return x_temp
end

function LinearAlgebra.ldiv!(M_temp::AbstractArray{T,MT}, A::DerivativeOperator{T,N}, M::AbstractArray{T,MT}) where {T<:Real, N, MT}

    # The case where M is a vector or matrix and A is differentiating along the first dimension
    if N == 1 && MT <= 2
        ldiv!(M_temp, factorize(Array(A)), M)

    # The case where M is differentiating along an arbitrary dimension
    else
        Mshape = size(M)
        if N != 1

            # Compute the high dimensional concretization B of A

            B = Matrix(I, Mshape[1],Mshape[1])
            for i in length(Mshape)-1:-1:1
                if N != length(Mshape) - i + 1
                    B = Kron(Matrix(I,Mshape[i],Mshape[i]),B)
                else
                    B = Kron(Array(A),B)
                end
            end
        else
            B = Array(A)
            for i in len(Mshape)-1:1
                B = Kron(Matrix(I,Mshape[i],Mshape[i]),B)
            end
        end

        # compute ldiv!
        ldiv!(vec(M_temp), factorize(Array(B)), vec(M))
    end
end

function \(A::DerivativeOperator{T,N}, M::AbstractArray{T,MT}) where {T<:Real, N,MT}
    M_temp_shape = [size(M)...]
    M_temp_shape[N] += 2
    M_temp = zeros(M_temp_shape...)
    ldiv!(M_temp, A, M)
    return M_temp
end
