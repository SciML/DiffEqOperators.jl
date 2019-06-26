struct GhostDerivativeOperator{T<:Real, E<:AbstractDerivativeOperator{T}, F<:AbstractBC{T}}
    L :: E
    Q :: F
end

function *(L::AbstractDerivativeOperator{T}, Q::AbstractBC{T}) where{T}
    return GhostDerivativeOperator{T, typeof(L), typeof(Q)}(L,Q)
end

function LinearAlgebra.mul!(x::AbstractVector{T}, A::GhostDerivativeOperator{T,E,F}, u::AbstractVector{T}) where {T,E,F}
    @assert length(u) == A.L.len == length(x)
    LinearAlgebra.mul!(x, A.L, A.Q*u)
end

function *(A::GhostDerivativeOperator{T,E,F}, u::AbstractVector{T}) where {T,E,F}
    @assert length(u) == A.L.len
    x = zeros(T, A.L.len)
    LinearAlgebra.mul!(x, A, u)
    return x
end

function LinearAlgebra.mul!(M_temp::AbstractMatrix{T}, A::GhostDerivativeOperator{T,E,F}, M::AbstractMatrix{T}) where {T,E,F}
    @assert size(M,1) == size(M_temp,1) == A.L.len
    for i in 1:size(M,2)
        mul!(view(M_temp,:,i), A, view(M,:,i))
    end
end

function *(A::GhostDerivativeOperator{T,E,F}, M::AbstractMatrix{T}) where {T,E,F}
    @assert size(M,1) == A.L.len
    M_temp = zeros(T, A.L.len, size(M,2))
    LinearAlgebra.mul!(M_temp, A, M)
    return M_temp
end


function LinearAlgebra.ldiv!(x::AbstractVector{T}, A::GhostDerivativeOperator{T,E,F}, u::AbstractVector{T}) where {T,E,F}
    @assert length(x) == size(A,2)
    (AL,Ab) = Array(A)
    LinearAlgebra.ldiv!(x, lu!(AL), u-Ab)
end


function \(A::GhostDerivativeOperator{T,E,F}, u::AbstractVector{T}) where {T,E,F}
    x = zeros(T,size(A,2))
    LinearAlgebra.ldiv!(x, A, u)
    return x
end

function LinearAlgebra.ldiv!(M_temp::AbstractMatrix{T}, A::GhostDerivativeOperator{T,E,F}, M::AbstractMatrix{T}) where {T,E,F}
    (AL,Ab) = Array(A)
    LinearAlgebra.ldiv!(M_temp, lu!(AL), M .- Ab)
end

function \(A::GhostDerivativeOperator{T,E,F}, M::AbstractMatrix{T}) where {T,E,F}
    M_temp = zeros(T, A.L.len, size(M,2))
    LinearAlgebra.ldiv!(M_temp, A, M)
    return M_temp
end


# length and sizes
Base.ndims(A::GhostDerivativeOperator) = 2
Base.size(A::GhostDerivativeOperator) = (A.L.len, A.L.len)
Base.size(A::GhostDerivativeOperator,i::Integer) = size(A)[i]
Base.length(A::GhostDerivativeOperator) = reduce(*, size(A))


# Concretizations, will be moved to concretizations.jl later
function LinearAlgebra.Array(A::GhostDerivativeOperator{T, E, F},N::Int=A.L.len) where {T,E,F}
    return (Array(A.L,N)*Array(A.Q,A.L.len)[1], Array(A.L,N)*Array(A.Q,A.L.len)[2])
end

function BandedMatrices.BandedMatrix(A::GhostDerivativeOperator{T, E, F},N::Int=A.L.len) where {T,E,F}
    return (BandedMatrix(A.L,N)*Array(A.Q,A.L.len)[1], BandedMatrix(A.L,N)*Array(A.Q,A.L.len)[2])
end

function SparseArrays.SparseMatrixCSC(A::GhostDerivativeOperator{T, E, F},N::Int=A.L.len) where {T,E,F}
    return (SparseMatrixCSC(A.L,N)*SparseMatrixCSC(A.Q,A.L.len)[1], SparseMatrixCSC(A.L,N)*SparseMatrixCSC(A.Q,A.L.len)[2])
end

function SparseArrays.sparse(A::GhostDerivativeOperator{T, E, F},N::Int=A.L.len) where {T,E,F}
    return SparseMatrixCSC(A,N)
end
