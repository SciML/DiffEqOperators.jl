# The composite operators are built using basic operators (scalar, array and
# derivative) using arithmetic or other operator compositions. The composite
# operator types are lazy and maintains the structure used to build them.

# Common defaults
## Recursive routines that use `getops`
function update_coefficients!(L::AbstractDiffEqCompositeOperator,u,p,t)
  for op in getops(L)
    update_coefficients!(op,u,p,t)
  end
  L
end
is_constant(L::AbstractDiffEqCompositeOperator) = all(is_constant, getops(L))
## Routines that use the AbstractMatrix representation
size(L::AbstractDiffEqCompositeOperator, args...) = size(convert(AbstractMatrix,L), args...)
opnorm(L::AbstractDiffEqCompositeOperator, p::Real=2) = opnorm(convert(AbstractMatrix,L), p)
getindex(L::AbstractDiffEqCompositeOperator, i::Int) = convert(AbstractMatrix,L)[i]
getindex(L::AbstractDiffEqCompositeOperator, I::Vararg{Int, N}) where {N} = 
  convert(AbstractMatrix,L)[I...]
for op in (:*, :/, :\)
  @eval $op(L::AbstractDiffEqCompositeOperator, x) = $op(convert(AbstractMatrix,L), x)
  @eval $op(x, L::AbstractDiffEqCompositeOperator) = $op(x, convert(AbstractMatrix,L))
end
mul!(Y, L::AbstractDiffEqCompositeOperator, B) = mul!(Y, convert(AbstractMatrix,L), B)
ldiv!(Y, L::AbstractDiffEqCompositeOperator, B) = ldiv!(Y, convert(AbstractMatrix,L), B)
for pred in (:isreal, :issymmetric, :ishermitian, :isposdef)
  @eval LinearAlgebra.$pred(L::AbstractDiffEqCompositeOperator) = $pred(convert(AbstractArray, L))
end
factorize(L::AbstractDiffEqCompositeOperator) = 
  FactorizedDiffEqArrayOperator(factorize(convert(AbstractArray, L)))
for fact in (:lu, :lu!, :qr, :qr!, :chol, :chol!, :ldlt, :ldlt!, 
  :bkfact, :bkfact!, :lq, :lq!, :svd, :svd!)
  @eval LinearAlgebra.$fact(L::AbstractDiffEqCompositeOperator, args...) = 
    FactorizedDiffEqArrayOperator($fact(convert(AbstractArray, L), args...))
end
## Routines that use the full matrix representation
LinearAlgebra.exp(L::AbstractDiffEqCompositeOperator) = exp(Matrix(L))

# The (u,p,t) and (du,u,p,t) interface
for T in subtypes(AbstractDiffEqCompositeOperator)
  (L::T)(u,p,t) = (update_coefficients!(L,u,p,t); L * u)
  (L::T)(du,u,p,t) = (update_coefficients!(L,u,p,t); mul!(du,L,u))
end
