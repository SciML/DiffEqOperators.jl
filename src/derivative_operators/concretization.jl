function Base.convert(::Type{Array}, A::AbstractDerivativeOperator{T}, N::Int=A.dimension) where T
    @assert N >= A.stencil_length # stencil must be able to fit in the matrix
    mat = zeros(T, (N, N+2))
    v = zeros(T, N+2)
    for i=1:N+2
        v[i] = one(T)
        #=
            calculating the effect on a unit vector to get the matrix of transformation
            to get the vector in the new vector space.
        =#
        mul!(view(mat,:,i), A, v)
        v[i] = zero(T)
    end
    return mat
end

function SparseArrays.sparse(A::AbstractDerivativeOperator{T}) where T
    N = A.dimension
    mat = spzeros(T, N, N)
    v = zeros(T, N)
    row = zeros(T, N)
    for i=1:N
        v[i] = one(T)
        #=
            calculating the effect on a unit vector to get the matrix of transformation
            to get the vector in the new vector space.
        =#
        mul!(row, A, v)
        copyto!(view(mat,:,i), row)
        @. row = 0 * row;
        v[i] = zero(T)
    end
    return mat
end
