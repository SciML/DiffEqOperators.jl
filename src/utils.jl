
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
