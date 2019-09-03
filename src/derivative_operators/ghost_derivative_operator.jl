struct GhostDerivativeOperator{T, E<:AbstractDiffEqLinearOperator{T}, F<:AbstractBC{T}}
    L :: E
    Q :: F
end

function Base.:*(L::AbstractDiffEqLinearOperator{T}, Q::AbstractBC{T}) where{T}
    return GhostDerivativeOperator{T, typeof(L), typeof(Q)}(L,Q)
end

function LinearAlgebra.mul!(x::AbstractArray{T,N}, A::GhostDerivativeOperator{T,E,F}, u::AbstractArray{T,N}) where {T,E,F,N}
    @assert size(u) == size(x)
    LinearAlgebra.mul!(x, A.L, u)
end

function *(A::GhostDerivativeOperator{T,E,F}, u::AbstractArray{T}) where {T,E,F}
    @assert length(u) == A.L.len
    x = similar(u)
    LinearAlgebra.mul!(x, A, A.Q*u)
    return x
end

function LinearAlgebra.ldiv!(x::AbstractVector{T}, A::GhostDerivativeOperator{T,E,F}, u::AbstractVector{T}) where {T,E,F}
    @assert length(x) == A.L.len
    (AL,Ab) = Array(A)
    LinearAlgebra.ldiv!(x, lu!(AL), u-Ab)
end


function \(A::GhostDerivativeOperator{T,E,F}, u::AbstractVector{T}) where {T,E,F}
    @assert length(u) == A.L.len
    x = zeros(T,size(A,2))
    LinearAlgebra.ldiv!(x, A, u)
    return x
end

function LinearAlgebra.ldiv!(M_temp::AbstractMatrix{T}, A::GhostDerivativeOperator{T,E,F}, M::AbstractMatrix{T}) where {T,E,F}
    @assert size(M_temp) == size(M)
    @assert A.L.len == size(M,1)
    (AL,Ab) = Array(A)
    LinearAlgebra.ldiv!(M_temp, lu!(AL), M .- Ab)
end

function \(A::GhostDerivativeOperator{T,E,F}, M::AbstractMatrix{T}) where {T,E,F}
    @assert A.L.len == size(M,1)
    M_temp = zeros(T, A.L.len, size(M,2))
    LinearAlgebra.ldiv!(M_temp, A, M)
    return M_temp
end

# update coefficients
function DiffEqBase.update_coefficients!(A::GhostDerivativeOperator{T,E,F},u,p,t) where {T,E,F}
    DiffEqBase.update_coefficients!(A.L,u,p,t)
end

function *(coeff_func::Function, A::GhostDerivativeOperator{T,N,Wind}) where {T,N,Wind}
    (coeff_func*A.L)*A.Q
end

# length and sizes
Base.ndims(A::GhostDerivativeOperator) = 2
Base.size(A::GhostDerivativeOperator) = (A.L.len, A.L.len)
Base.size(A::GhostDerivativeOperator,i::Integer) = size(A)[i]
Base.length(A::GhostDerivativeOperator) = reduce(*, size(A))
