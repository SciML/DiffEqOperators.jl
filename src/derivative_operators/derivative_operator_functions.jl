for MT in [AbstractArray, AbstractMatrix]
    @eval begin
        function LinearAlgebra.mul!(x_temp::$MT{T}, A::DerivativeOperator{T,N}, M::$MT{T}) where {T<:Real,N}

            v = zeros(ndims(x_temp))
            v[N] = 2
            @assert [size(x_temp)...]+v == [size(M)...]

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
    end
end

for MT in [AbstractVector, AbstractMatrix, AbstractArray]
    @eval begin
        function wacky(A::DerivativeOperator{T,N},M::$MT{T}) where {T<:Real,N}
            size_x_temp = [size(M)...]
            size_x_temp[N] -= 2
            x_temp = zeros(promote_type(eltype(A),eltype(M)), size_x_temp...)
            LinearAlgebra.mul!(x_temp, A, M)
            return x_temp
        end
    end
end
