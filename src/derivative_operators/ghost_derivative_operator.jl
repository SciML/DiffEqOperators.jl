struct GhostDerivativeOperator{T, E<:AbstractDiffEqLinearOperator{T}, F<:AbstractBC{T}} <: AbstractDiffEqLinearOperator{T}
    L :: E
    Q :: F
end

function *(L::AbstractDiffEqLinearOperator{T}, Q::AbstractBC{T}) where{T}
    return GhostDerivativeOperator{T, typeof(L), typeof(Q)}(L,Q)
end

function *(L::AbstractDiffEqCompositeOperator{T}, Q::AbstractBC{T}) where{T}
    return sum(map(op -> op * Q, L.ops))
end

function LinearAlgebra.mul!(x::AbstractArray{T,N}, A::GhostDerivativeOperator{T,E,F}, u::AbstractArray{T,N}) where {T,E,F,N}
    @assert size(u) == size(x)
    LinearAlgebra.mul!(x, A.L, u)
end

function *(A::GhostDerivativeOperator{T1}, u::AbstractVector{T2}) where {T1,T2}
    @assert length(u) == A.L.len
    x = similar(u, promote_type(T1,T2))
    LinearAlgebra.mul!(x, A, A.Q*u)
    return x
end

<<<<<<< HEAD
function LinearAlgebra.mul!(M_temp::AbstractMatrix, A::GhostDerivativeOperator, M::AbstractMatrix)
    @assert size(M,1) == size(M_temp,1) == A.L.len
    for i in 1:size(M,2)
        mul!(view(M_temp,:,i), A, view(M,:,i))
    end
end

function *(A::GhostDerivativeOperator{T1}, M::AbstractMatrix{T2}) where {T1,T2}
    @assert size(M,1) == A.L.len
    M_temp = zeros(promote_type(T1,T2), A.L.len, size(M,2))
    LinearAlgebra.mul!(M_temp, A, M)
    return M_temp
end


function LinearAlgebra.ldiv!(x::AbstractVector, A::GhostDerivativeOperator, u::AbstractVector)
=======
function LinearAlgebra.ldiv!(x::AbstractVector{T}, A::GhostDerivativeOperator{T,E,F}, u::AbstractVector{T}) where {T,E,F}
>>>>>>> f00c2c5... Allowed AbstractArrays instead of AbstractVecOrMat, allowed AbstractArrays for GhostDerivativeOperator
    @assert length(x) == A.L.len
    (AL,Ab) = Array(A)
    LinearAlgebra.ldiv!(x, lu!(AL), u-Ab)
end


function \(A::GhostDerivativeOperator{T1}, u::AbstractVector{T2}) where {T1,T2}
    @assert length(u) == A.L.len
    x = zeros(promote_type(T1,T2),size(A,2))
    LinearAlgebra.ldiv!(x, A, u)
    return x
end

function LinearAlgebra.ldiv!(M_temp::AbstractMatrix, A::GhostDerivativeOperator, M::AbstractMatrix)
    @assert size(M_temp) == size(M)
    @assert A.L.len == size(M,1)
    (AL,Ab) = Array(A)
    LinearAlgebra.ldiv!(M_temp, lu!(AL), M .- Ab)
end

function \(A::GhostDerivativeOperator{T1}, M::AbstractMatrix{T2}) where {T1,T2}
    @assert A.L.len == size(M,1)
    M_temp = zeros(promote_type(T1,T2), A.L.len, size(M,2))
    LinearAlgebra.ldiv!(M_temp, A, M)
    return M_temp
end

# # Interface with other GhostDerivativeOperator and with
# # AbstractDiffEqCompositeOperator

# update coefficients
function DiffEqBase.update_coefficients!(A::GhostDerivativeOperator,u,p,t)
    DiffEqBase.update_coefficients!(A.L,u,p,t)
end

# Implement multiplication for coefficients
function *(c::Number, A::GhostDerivativeOperator)
    (c * A.L) * A.Q
end

function *(c::Vector{<:Number}, A::GhostDerivativeOperator)
    (c * A.L) * A.Q
end

function *(coeff_func::Function, A::GhostDerivativeOperator)
    (coeff_func*A.L)*A.Q
end

=======
>>>>>>> f00c2c5... Allowed AbstractArrays instead of AbstractVecOrMat, allowed AbstractArrays for GhostDerivativeOperator
# length and sizes
Base.ndims(A::GhostDerivativeOperator) = 2
Base.size(A::GhostDerivativeOperator) = (A.L.len, A.L.len)
Base.size(A::GhostDerivativeOperator,i::Integer) = size(A)[i]
Base.length(A::GhostDerivativeOperator) = reduce(*, size(A))
