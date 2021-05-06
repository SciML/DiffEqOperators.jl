using SymbolicUtils

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
    for i in 1:length(s)-1
        out += (I[i+1]-1)*prod(s[1:i])
    end
    return out
end

# Counts the Differential operators for given variable x. This is used to determine
# the order of a PDE.
function count_differentials(term, x::Symbolics.Symbolic)
    S = Symbolics
    SU = SymbolicUtils
    if !S.istree(term)
        return 0
    else
        op = SU.operation(term)
        count_children = sum(map(arg -> count_differentials(arg, x), SU.arguments(term)))
        if op isa Differential && op.x === x
            return 1 + count_children
        end
        return count_children
    end
end

# return list of differential orders in the equation
function differential_order(eq, x::Symbolics.Symbolic)
    S = Symbolics
    SU = SymbolicUtils
    orders = Set{Int}()
    if S.istree(eq)
        op = SU.operation(eq)
        if op isa Differential
            push!(orders, count_differentials(eq, x))
        else
            for o in map(ch -> differential_order(ch, x), SU.arguments(eq))
                union!(orders, o)
            end
        end
    end
    return filter(!iszero, orders)
end

add_dims(A::AbstractArray, n::Int; dims::Int = 1) = cat(ndims(A) + n, A, dims = dims)

""
perpindex(A, dim::Integer) = A[setdiff(1:length(A), dim)]

"""
the size of A perpendicular to dim
"""
perpsize(A::AbstractArray{T,N}, dim::Integer) where {T,N} = size(A)[setdiff(1:N, dim)]
