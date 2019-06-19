struct GhostDerivativeOperator{T<:Real, E<:AbstractDerivativeOperator{T}, F<:AbstractBC{T}}
    L :: E
    Q :: F
end

function *(L::AbstractDerivativeOperator{T}, Q::AbstractBC{T}) where{T}
    return GhostDerivativeOperator{T, typeof(L), typeof(Q)}(L,Q)
end

function *(A::GhostDerivativeOperator{T,E,F}, u::AbstractVector{T}) where {T,E,F}
    @assert length(u) == A.L.len
    return A.L*(A.Q*u)
end

Base.ndims(A::GhostDerivativeOperator) = 2
Base.size(A::GhostDerivativeOperator) = (A.L.len, A.L.len)
Base.size(A::GhostDerivativeOperator,i::Integer) = size(A)[i]
Base.length(A::GhostDerivativeOperator) = reduce(*, size(A))

# Concretizations, will be moved to concretizations.jl later
function LinearAlgebra.Array(A::GhostDerivativeOperator{T, E, F},N::Int=A.L.len) where {T,E,F}
    return (Array(A.L,N)*Array(A.Q,A.L.len)[1], Array(A.L,N)*Array(A.Q,A.L.len)[2])
end

function BandedMatrices.BandedMatrix(A::GhostDerivativeOperator{T, E, F},N::Int=A.L.len) where {T,E,F}
    return (BandedMatrix(A.L,N)*BandedMatrix(A.Q,A.L.len)[1], BandedMatrix(A.L,N)*BandedMatrix(A.Q,A.L.len)[2])
end

function SparseArrays.SparseMatrixCSC(A::GhostDerivativeOperator{T, E, F},N::Int=A.L.len) where {T,E,F}
    return (SparseMatrixCSC(A.L,N)*SparseMatrixCSC(A.Q,A.L.len)[1], SparseMatrixCSC(A.L,N)*SparseMatrixCSC(A.Q,A.L.len)[2])
end

function SparseArrays.sparse(A::GhostDerivativeOperator{T, E, F},N::Int=A.L.len) where {T,E,F}
    return SparseMatrixCSC(A,N)
end
