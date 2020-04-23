#
# Boundary Padded Arrays
#
# These are returned by boundary condition operators.
#

abstract type AbstractBoundaryPaddedArray{T, N} <: AbstractArray{T, N} end
abstract type AbstractDirectionalBoundaryPaddedArray{T, N, D} <: AbstractBoundaryPaddedArray{T, N} end
abstract type AbstractComposedBoundaryPaddedArray{T, N} <: AbstractBoundaryPaddedArray{T,N} end
"""
A vector type that extends a vector u with one ghost point at each end.
"""
struct BoundaryPaddedVector{T,T2 <: AbstractVector{T}} <: AbstractBoundaryPaddedArray{T, 1}
    l::T
    r::T
    u::T2
end

Base.length(Q::BoundaryPaddedVector) = length(Q.u) + 2
Base.size(Q::BoundaryPaddedVector) = (length(Q.u) + 2,)
Base.lastindex(Q::BoundaryPaddedVector) = Base.length(Q)

function Base.getindex(Q::BoundaryPaddedVector,i)
    if i == 1
        return Q.l
    elseif i == length(Q)
        return Q.r
    else
        return Q.u[i-1]
    end
end

"""
Higher dimensional generalization of BoundaryPaddedVector, pads an array of dimension N along the dimension D with 2 Arrays of dimension N-1, stored in lower and upper

"""
struct BoundaryPaddedArray{T, D, N, M, V<:AbstractArray{T, N}, B<: AbstractArray{T, M}} <: AbstractDirectionalBoundaryPaddedArray{T,N, D}
    lower::B #an array of dimension M = N-1, used to extend the lower index boundary
    upper::B #Ditto for the upper index boundary
    u::V
end

getaxis(Q::BoundaryPaddedArray{T,D,N,M,V,B}) where {T,D,N,M,V,B} = D

function Base.size(Q::BoundaryPaddedArray)
    S = [size(Q.u)...]
    S[getaxis(Q)] += 2
    return Tuple(S)
end

"""
A = compose(padded_arrays::BoundaryPaddedArray...)

-------------------------------------------------------------------------------------

Example:
A = compose(Ax, Ay, Az) # 3D domain
A = compose(Ax, Ay) # 2D Domain

Composes BoundaryPaddedArrays that extend the same u for each different dimension that u has in to a ComposedBoundaryPaddedArray

Ax Ay and Az can be passed in any order, as long as there is exactly one BoundaryPaddedArray that extends each dimension.
"""
function compose(padded_arrays::BoundaryPaddedArray...)
    N = ndims(padded_arrays[1])
    Ds = getaxis.(padded_arrays)
    (length(padded_arrays) == N) || throw(ArgumentError("The padded_arrays must cover every dimension - make sure that the number of padded_arrays is equal to ndims(u)."))
    for D in Ds
        length(setdiff(Ds, D)) < N || throw(ArgumentError("There are multiple Arrays that extend along dimension $D - make sure every dimension has a unique extension"))
    end
    any(fill(padded_arrays[1].u, (length(padded_arrays),)) .== getfield.(padded_arrays, :u)) || throw(ArgumentError("The padded_arrays do not all extend the same u!"))
    padded_arrays = padded_arrays[sortperm([Ds...])]
    lower = [padded_array.lower for padded_array in padded_arrays]
    upper = [padded_array.upper for padded_array in padded_arrays]

    ComposedBoundaryPaddedArray{gettype(padded_arrays[1]),N,N-1,typeof(padded_arrays[1].u),typeof(lower[1])}(lower, upper, padded_arrays[1].u)
end

# Composed BoundaryPaddedArray

struct ComposedBoundaryPaddedArray{T, N, M, V<:AbstractArray{T, N}, B<: AbstractArray{T, M}} <: AbstractComposedBoundaryPaddedArray{T, N}
    lower::Vector{B}
    upper::Vector{B}
    u::V
end

# Aliases
AbstractBoundaryPaddedMatrix{T} = AbstractBoundaryPaddedArray{T,2}
AbstractBoundaryPadded3Tensor{T} = AbstractBoundaryPaddedArray{T,3}

BoundaryPaddedMatrix{T, D, V, B} = BoundaryPaddedArray{T, D, 2, 1, V, B}
BoundaryPadded3Tensor{T, D, V, B} = BoundaryPaddedArray{T, D, 3, 2, V, B}

ComposedBoundaryPaddedMatrix{T,V,B} = ComposedBoundaryPaddedArray{T,2,1,V,B}
ComposedBoundaryPadded3Tensor{T,V,B} = ComposedBoundaryPaddedArray{T,3,2,V,B}

Base.size(Q::ComposedBoundaryPaddedArray) = size(Q.u).+2

"""
Ax, Ay,... = decompose(A::ComposedBoundaryPaddedArray)

-------------------------------------------------------------------------------------

Decomposes a ComposedBoundaryPaddedArray in to components that extend along each dimension individually
"""
decompose(A::ComposedBoundaryPaddedArray) = Tuple([BoundaryPaddedArray{gettype(A), i, ndims(A), ndims(A)-1, typeof(lower[1])}(A.lower[i], A.upper[i], A.u) for i in 1:ndims(A)])

Base.length(Q::AbstractBoundaryPaddedArray) = reduce((*), size(Q))
Base.firstindex(Q::AbstractBoundaryPaddedArray, d::Int) = 1
Base.lastindex(Q::AbstractBoundaryPaddedArray) = length(Q)
Base.lastindex(Q::AbstractBoundaryPaddedArray, d::Int) = size(Q)[d]
gettype(Q::AbstractBoundaryPaddedArray{T,N}) where {T,N} = T
Base.ndims(Q::AbstractBoundaryPaddedArray{T,N}) where {T,N} = N

function Base.getindex(Q::BoundaryPaddedArray{T,D,N,M,V,B}, _inds::Vararg{Int,N}) where {T,D,N,M,V,B} #supports range and colon indexing!
    inds = [_inds...]
    S = size(Q)
    dim = D
    otherinds = inds[setdiff(1:N, dim)]
    @assert length(S) == N
    if inds[dim] == 1
        return Q.lower[otherinds...]
    elseif inds[dim] == S[dim]
        return Q.upper[otherinds...]
    elseif typeof(inds[dim]) <: Integer
        inds[dim] = inds[dim] - 1
        return Q.u[inds...]
    elseif typeof(inds[dim]) == Colon
        if mapreduce(x -> typeof(x) != Colon, (|), otherinds)
            return vcat(Q.lower[otherinds...],  Q.u[inds...], Q.upper[otherinds...])
        else
            throw("A colon on the extended dim is as yet incompatible with additional colons")
        end
    elseif typeof(inds[dim]) <: AbstractArray
        throw("Range indexing not yet supported!")
    end
end

function Base.getindex(Q::ComposedBoundaryPaddedArray{T, N, M, V, B} , inds::Vararg{Int, N}) where {T, N, M, V, B} #as yet no support for range indexing or colon indexing
    S = size(Q)
    @assert reduce((&), inds .<= S)
    for (dim, index) in enumerate(inds)
        if index == 1
            _inds = inds[setdiff(1:N, dim)]
            if (1 ∈ _inds) | any(S[setdiff(1:N, dim)] .== _inds)
                return zero(T)
            else
                return Q.lower[dim][(_inds.-1)...]
            end
        elseif index == S[dim]
            _inds = inds[setdiff(1:N, dim)]
            if (1 ∈ _inds) | any(S[setdiff(1:N, dim)] .== _inds)
                return zero(T)
            else
                return Q.upper[dim][(_inds.-1)...]
            end
        end
     end
    return Q.u[(inds.-1)...]
end
