function LinearAlgebra.mul!(x_temp::AbstractArray{T}, A::DerivativeOperator{T,N,Wind,T2,S1,S2,T3,F}, M::AbstractArray{T,MN}) where {T<:Real,N,Wind,T2,S1,S2<:SVector,T3,F,MN<:Int}

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

function *(A::DerivativeOperator{T,N,Wind,T2,S1,S2,T3,F},M::AbstractArray{T,MN}) where {T<:Real,N,Wind,T2,S1,S2<:SVector,T3,F,MN<:Int}
    size_x_temp = [size(M)...]
    size_x_temp[N] -= 2
    x_temp = zeros(promote_type(eltype(A),eltype(M)), size_x_temp...)
    LinearAlgebra.mul!(x_temp, A, M)
    return x_temp
end

function *(A::DerivativeOperator{T,N,Wind,T2,S1,S2,T3,F},x::BoundaryPaddedVector) where {T<:Real,N,Wind,T2,S1,S2<:SVector,T3,F}
    y = zeros(promote_type(eltype(A),eltype(x)), length(x)-2)
    LinearAlgebra.mul!(y, A::AbstractDerivativeOperator, x::BoundaryPaddedVector)
    return y
end
