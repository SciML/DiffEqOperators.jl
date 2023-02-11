struct GhostDerivativeOperator{T, E <: AbstractDiffEqLinearOperator{T}, F <: AbstractBC{T}
                               } <: AbstractDiffEqLinearOperator{T}
    L::E
    Q::F
end

function *(L::AbstractDiffEqLinearOperator{T}, Q::AbstractBC{T}) where {T}
    return GhostDerivativeOperator{T, typeof(L), typeof(Q)}(L, Q)
end

function *(L::AbstractDiffEqCompositeOperator{T}, Q::AbstractBC{T}) where {T}
    return sum(map(op -> op * Q, L.ops))
end

function LinearAlgebra.mul!(out::AbstractVecOrMat, A::GhostDerivativeOperator,
                            u::AbstractVecOrMat)
    padded = A.Q * u  # assume: creates boundary padded array w/o realloc
    LinearAlgebra.mul!(out, A.L, padded)
end
function LinearAlgebra.mul!(out::AbstractArray, A::GhostDerivativeOperator,
                            u::AbstractArray)
    padded = A.Q * u  # assume: creates boundary padded array w/o realloc
    LinearAlgebra.mul!(out, A.L, padded)
end

function *(A::GhostDerivativeOperator{T1}, u::AbstractVecOrMat{T2}) where {T1, T2}
    #TODO Implement a function domaincheck(L::AbstractDiffEqLinearOperator, u) to see if components of L along each dimension match the size of u
    x = zeros(promote_type(T1, T2), unpadded_size(u))
    LinearAlgebra.mul!(x, A, u)
    return x
end
function *(A::GhostDerivativeOperator{T1}, u::AbstractArray{T2}) where {T1, T2}
    #TODO Implement a function domaincheck(L::AbstractDiffEqLinearOperator, u) to see if components of L along each dimension match the size of u
    x = zeros(promote_type(T1, T2), unpadded_size(u))
    LinearAlgebra.mul!(x, A, u)
    return x
end

function \(A::GhostDerivativeOperator, u::AbstractVector) # FIXME as above, should promote_type(T1,T2)
    # TODO: is this specialization to u::AbstractVector really any faster?
    @assert length(u) == size(A.L, 1)
    (A_l, A_b) = sparse(A, length(u))
    A_l \ Vector(u .- A_b)
end
function \(A::GhostDerivativeOperator, u::AbstractMatrix) # FIXME should have T1,T2 and promote result
    #TODO implement check that A has compatible size with u
    s = size(u)
    (A_l, A_b) = sparse(A, s)
    x = A_l \ Vector(reshape(u, length(u)) .- A_b) #Has to be converted to vector to work, A_b being sparse was causing a conversion to sparse.
    return reshape(x, s)
end
function \(A::GhostDerivativeOperator, u::AbstractArray) # FIXME should have T1,T2 and promote result
    #TODO implement check that A has compatible size with u
    s = size(u)
    (A_l, A_b) = sparse(A, s)
    x = A_l \ Vector(reshape(u, length(u)) .- A_b) #Has to be converted to vector to work, A_b being sparse was causing a conversion to sparse.
    return reshape(x, s)
end

# update coefficients
function DiffEqBase.update_coefficients!(A::GhostDerivativeOperator, u, p, t)
    DiffEqBase.update_coefficients!(A.L, u, p, t)
end

# Implement multiplication for coefficients
function *(c::Number, A::GhostDerivativeOperator)
    (c * A.L) * A.Q
end

function *(c::Vector{<:Number}, A::GhostDerivativeOperator)
    (c * A.L) * A.Q
end

function *(coeff_func::Function, A::GhostDerivativeOperator)
    (coeff_func * A.L) * A.Q
end

# length and sizes
Base.ndims(A::GhostDerivativeOperator) = 2
Base.size(A::GhostDerivativeOperator) = (A.L.len, A.L.len)
Base.size(A::GhostDerivativeOperator, i::Integer) = size(A)[i]
Base.length(A::GhostDerivativeOperator) = reduce(*, size(A))

@inline function ==(A1::GhostDerivativeOperator, A2::GhostDerivativeOperator)
    A1.L == A2.L && A1.Q == A2.Q
end
