#=
    Most of the functionality for linear combination of operators is already
    covered in LinearMaps.jl.
=#

(L::LinearCombination)(u,p,t) = L*u
(L::LinearCombination)(du,u,p,t) = mul!(du,L,u)

#=
    The fallback implementation in LinearMaps.jl effectively computes A*eye(N),
    which is very inefficient.

    Instead, build up the full matrix for each operator iteratively.
=#
# TODO: Type dispatch for this is incorrect at the moment
# function Base.full(A::LinearCombination{T,Tuple{Vararg{O}},Ts}) where {T,O<:Union{DiffEqBase.AbstractDiffEqLinearOperator,IdentityMap},Ts}
#     out = zeros(T,size(A))
#     for i = 1:length(A.maps)
#         c = A.coeffs[i]
#         op = A.maps[i]
#         if isa(op, IdentityMap)
#             @. out += c * eye(size(A,1))
#         else
#             @. out += c * full(op)
#         end
#     end
#     return out
# end

#=
    Fallback methods that use the full representation
=#
Base.exp(A::LinearCombination) = exp(full(A))
Base.:\(A::AbstractVecOrMat, B::LinearCombination) = A \ full(B)
Base.:\(A::LinearCombination, B::AbstractVecOrMat) = full(A) \ B
Base.:/(A::AbstractVecOrMat, B::LinearCombination) = A / full(B)
Base.:/(A::LinearCombination, B::AbstractVecOrMat) = full(A) / B

LinearAlgebra.opnorm(A::IdentityMap{T}, p::Real=2) where T = real(one(T))
LinearAlgebra.opnorm(A::LinearCombination, p::Real=2) = opnorm(full(A), p)
#=
    The opnorm of A+B is difficult to calculate, but in many applications we only
    need an estimate of the opnorm (e.g. for error analysis) so it makes sense to
    compute the upper bound given by the triangle inequality

        |A + B| <= |A| + |B|

    For derivative operators A and B, their Inf opnorm can be calculated easily
    and thus so is the Inf opnorm bound of A + B.
=#
opnormbound(a::Number, p::Real=2) = abs(a)
opnormbound(A::AbstractArray, p::Real=2) = opnorm(A, p)
opnormbound(A::Union{DiffEqBase.AbstractDiffEqLinearOperator,IdentityMap}, p::Real=2) = opnorm(A, p)
opnormbound(A::LinearCombination, p::Real=2) = sum(abs.(A.coeffs) .* opnormbound.(A.maps, p))
