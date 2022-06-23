"""
A function that creates a tuple of CartesianIndices of unit length and `N` dimensions, one pointing along each dimension.
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

function cartesian_to_linear(I::CartesianIndex, s)  #Not sure if there is a builtin that does this - convert cartesian index to linear index of an array of size s
    out = I[1]
    for i in 1:(length(s) - 1)
        out += (I[i + 1] - 1) * prod(s[1:i])
    end
    return out
end

add_dims(A::AbstractArray, n::Int; dims::Int = 1) = cat(ndims(A) + n, A, dims = dims)

""
perpindex(A, dim::Integer) = A[setdiff(1:length(A), dim)]

"""
the size of A perpendicular to dim
"""
perpsize(A::AbstractArray{T, N}, dim::Integer) where {T, N} = size(A)[setdiff(1:N, dim)]
