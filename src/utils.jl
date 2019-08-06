"""
Efficiently reduce a BitArray with or (|)
"""
function or(B::BitArray)
    for b in B
        if b
            return true
        end
    end
    return false
end

function or(B::NTuple{N,Bool}) where N
    for b in B
        if b
            return true
        end
    end
    return false
end

"""
Efficiently reduce a BitArray with and (&)
"""
function and(B::BitArray)
    for b in B
        if !b
            return false
        end
    end
    return true
end

function and(B::NTuple{N, Bool}) where N
    for b in B
        if !b
            return false
        end
    end
    return true
end

"""
A function to generate the correct permutation to flip an array of dimension `N` to be orthogonal to `dim`
"""
function orth_perm(N::Int, dim::Int)
    if dim == N
        return Vector(1:N)
    elseif dim < N
        P = experms(N, dim+1)
        P[dim], P[dim+1] = P[dim+1], P[dim]
        return P
    else
        throw("Dim is greater than N!")
    end
end

"""
A function that creates a tuple of CartesianIndices of unit length and `N` dimensions, one pointing along each dimension
"""
function unit_indices(N::Int) #create unit CartesianIndex for each dimension
    out = Vector{CartesianIndex{N}}(undef, N)
    null = zeros(Int64, N)
    for i in 1:N
        unit_i = copy(null)
        unit_i[i] = 1
        out[i] = CartesianIndex(Tuple(unit_i))
    end
    Tuple(out)
end

add_dims(A::AbstractArray, n::Int) = cat(ndims(a) + n, a)

""
perpindex(A, dim::Integer) = A[setdiff(1:length(A), dim)]

"""
the size of A perpendicular to dim
"""
perpsize(A::AbstractArray{T,N}, dim::Integer) where {T,N} = size(A)[setdiff(1:N, dim)]
