#=
    Most of the functionality for linear combination of operators is already
    covered in LinearMaps.jl.
=#

#=
    The fallback implementation in LinearMaps.jl effectively computes A*eye(N), 
    which is very inefficient.

    Instead, build up the full matrix for each operator iteratively.
=#
# TODO: Type dispatch for this is incorrect at the moment
# function Base.full(A::LinearCombination{T,Tuple{Vararg{O}},Ts}) where {T,O<:Union{AbstractDiffEqLinearOperator,IdentityMap},Ts}
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

Base.expm(A::LinearCombination) = expm(full(A))
