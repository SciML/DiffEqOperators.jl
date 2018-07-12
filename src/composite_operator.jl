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
  @eval $op(L::AbstractDiffEqCompositeOperator{T}, x::AbstractVecOrMat{T}) where {T} =
    $op(convert(AbstractMatrix,L), x)
  @eval $op(x::AbstractVecOrMat{T}, L::AbstractDiffEqCompositeOperator{T}) where {T} =
    $op(x, convert(AbstractMatrix,L))
end
mul!(Y::AbstractVecOrMat{T}, L::AbstractDiffEqCompositeOperator{T},
  B::AbstractVecOrMat{T}) where {T} = mul!(Y, convert(AbstractMatrix,L), B)
ldiv!(Y::AbstractVecOrMat{T}, L::AbstractDiffEqCompositeOperator{T},
  B::AbstractVecOrMat{T}) where {T} = ldiv!(Y, convert(AbstractMatrix,L), B)
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

# Scaled operator (α * A)
struct DiffEqScaledOperator{T,F,OpType<:AbstractDiffEqLinearOperator{T}} <: AbstractDiffEqCompositeOperator{T}
  coeff::DiffEqScalar{T,F}
  op::OpType
end
*(α::DiffEqScalar{T,F}, L::AbstractDiffEqLinearOperator{T}) where {T,F} = DiffEqScaledOperator(α, L)
-(L::AbstractDiffEqLinearOperator{T}) where {T} = DiffEqScalar(-one(T)) * L
getops(L::DiffEqScaledOperator) = (L.coeff, L.op)
Matrix(L::DiffEqScaledOperator) = L.coeff * Matrix(L.op)
convert(::Type{AbstractMatrix}, L::DiffEqScaledOperator) = L.coeff * convert(AbstractMatrix, L.op)

size(L::DiffEqScaledOperator, args...) = size(L.op, args...)
opnorm(L::DiffEqScaledOperator, p::Real=2) = abs(L.coeff) * opnorm(L.op, p)
getindex(L::DiffEqScaledOperator, i::Int) = L.coeff * L.op[i]
getindex(L::DiffEqScaledOperator, I::Vararg{Int, N}) where {N} = 
  L.coeff * L.op[I...]
*(L::DiffEqScaledOperator{T,F}, x::AbstractVecOrMat{T}) where {T,F} = L.coeff * (L.op * x)
*(x::AbstractVecOrMat{T}, L::DiffEqScaledOperator{T,F}) where {T,F} = (L.op * x) * L.coeff
/(L::DiffEqScaledOperator{T,F}, x::AbstractVecOrMat{T}) where {T,F} = L.coeff * (L.op / x)
/(x::AbstractVecOrMat{T}, L::DiffEqScaledOperator{T,F}) where {T,F} = 1/L.coeff * (x / L.op)
\(L::DiffEqScaledOperator{T,F}, x::AbstractVecOrMat{T}) where {T,F} = 1/L.coeff * (L.op \ x)
\(x::AbstractVecOrMat{T}, L::DiffEqScaledOperator{T,F}) where {T,F} = L.coeff * (x \ L)
mul!(Y::AbstractVecOrMat{T}, L::DiffEqScaledOperator{T,F},
  B::AbstractVecOrMat{T}) where {T,F} = lmul!(L.coeff, mul!(Y, L.op, B))
ldiv!(Y::AbstractVecOrMat{T}, L::DiffEqScaledOperator{T,F},
  B::AbstractVecOrMat{T}) where {T,F} = lmul!(1/L.coeff, ldiv!(Y, L.op, B))
factorize(L::DiffEqScaledOperator) = L.coeff * factorize(L.op)
for fact in (:lu, :lu!, :qr, :qr!, :chol, :chol!, :ldlt, :ldlt!, 
  :bkfact, :bkfact!, :lq, :lq!, :svd, :svd!)
  @eval LinearAlgebra.$fact(L::DiffEqScaledOperator, args...) = 
    L.coeff * fact(L.op, args...)
end

# Linear Combination
struct DiffEqOperatorCombination{T,O<:Tuple{Vararg{AbstractDiffEqLinearOperator{T}}},
  C<:AbstractVector{T}} <: AbstractDiffEqCompositeOperator{T}
  ops::O
  cache::C
  function DiffEqOperatorCombination(ops; cache=nothing)
    T = eltype(ops[1])
    if cache == nothing
      cache = Vector{T}(undef, size(ops[1], 1))
    end
    # TODO: safecheck dimensions
    new{T,typeof(ops),typeof(cache)}(ops, cache)
  end
end
+(ops::AbstractDiffEqLinearOperator...) = DiffEqOperatorCombination(ops)
+(L1::DiffEqOperatorCombination, L2::AbstractDiffEqLinearOperator) = DiffEqOperatorCombination((L1.ops..., L2))
+(L1::AbstractDiffEqLinearOperator, L2::DiffEqOperatorCombination) = DiffEqOperatorCombination((L1, L2.ops...))
+(L1::DiffEqOperatorCombination, L2::DiffEqOperatorCombination) = DiffEqOperatorCombination((L1.ops..., L2.ops...))
-(L1::AbstractDiffEqLinearOperator, L2::AbstractDiffEqLinearOperator) = L1 + (-L2)
getops(L::DiffEqOperatorCombination) = L.ops
Matrix(L::DiffEqOperatorCombination) = sum(Matrix, L.ops)
convert(::Type{AbstractMatrix}, L::DiffEqOperatorCombination) =
  sum(op -> convert(AbstractMatrix, op), L.ops)

size(L::DiffEqOperatorCombination, args...) = size(L.ops[1], args...)
getindex(L::DiffEqOperatorCombination, i::Int) = sum(op -> op[i], L.ops)
getindex(L::DiffEqOperatorCombination, I::Vararg{Int, N}) where {N} = sum(op -> op[I...], L.ops)
*(L::DiffEqOperatorCombination, x::AbstractVecOrMat) = sum(op -> op * x, L.ops)
*(x::AbstractVecOrMat, L::DiffEqOperatorCombination) = sum(op -> x * op, L.ops)
/(L::DiffEqOperatorCombination, x::AbstractVecOrMat) = sum(op -> op / x, L.ops)
\(x::AbstractVecOrMat, L::DiffEqOperatorCombination) = sum(op -> x \ op, L.ops)
function mul!(y::AbstractVector, L::DiffEqOperatorCombination, b::AbstractVector)
  mul!(y, L.ops[1], b)
  for op in L.ops[2:end]
    mul!(L.cache, op, b)
    y .+= L.cache
  end
  return y
end

# Composition (A ∘ B)
struct DiffEqOperatorComposition{T,O<:Tuple{Vararg{AbstractDiffEqLinearOperator{T}}},
  C<:Tuple{Vararg{AbstractVector{T}}}} <: AbstractDiffEqCompositeOperator{T}
  ops::O # stored in the order of application
  caches::C
  function DiffEqOperatorComposition(ops; caches=nothing)
    T = eltype(ops[1])
    # TODO: safecheck dimensions
    if caches == nothing
      # Construct a list of caches to be used by mul! and ldiv!
      caches = []
      for op in ops[1:end-1]
        push!(caches, Vector{T}(undef, size(op, 1)))
      end
      caches = tuple(caches...)
    end
    new{T,typeof(ops),typeof(caches)}(ops, caches)
  end
end
*(ops::AbstractDiffEqLinearOperator...) = DiffEqOperatorComposition(reverse(ops))
∘(L1::AbstractDiffEqLinearOperator, L2::AbstractDiffEqLinearOperator) = DiffEqOperatorComposition((L2, L1))
*(L1::DiffEqOperatorComposition, L2::AbstractDiffEqLinearOperator) = DiffEqOperatorComposition((L2, L1.ops...))
∘(L1::DiffEqOperatorComposition, L2::AbstractDiffEqLinearOperator) = DiffEqOperatorComposition((L2, L1.ops...))
*(L1::AbstractDiffEqLinearOperator, L2::DiffEqOperatorComposition) = DiffEqOperatorComposition((L2.ops..., L1))
∘(L1::AbstractDiffEqLinearOperator, L2::DiffEqOperatorComposition) = DiffEqOperatorComposition((L2.ops..., L1))
*(L1::DiffEqOperatorComposition, L2::DiffEqOperatorComposition) = DiffEqOperatorComposition((L2.ops..., L1.ops...))
∘(L1::DiffEqOperatorComposition, L2::DiffEqOperatorComposition) = DiffEqOperatorComposition((L2.ops..., L1.ops...))
getops(L::DiffEqOperatorComposition) = L.ops
Matrix(L::DiffEqOperatorComposition) = prod(Matrix, reverse(L.ops))
convert(::Type{AbstractMatrix}, L::DiffEqOperatorComposition) =
  prod(op -> convert(AbstractMatrix, op), reverse(L.ops))

size(L::DiffEqOperatorComposition) = (size(L.ops[end], 1), size(L.ops[1], 2))
size(L::DiffEqOperatorComposition, m::Integer) = size(L)[m]
opnorm(L::DiffEqOperatorComposition) = prod(opnorm, L.ops)
*(L::DiffEqOperatorComposition, x::AbstractVecOrMat) = foldl((acc, op) -> op*acc, L.ops; init=x)
*(x::AbstractVecOrMat, L::DiffEqOperatorComposition) = foldl((acc, op) -> acc*op, reverse(L.ops); init=x)
/(L::DiffEqOperatorComposition, x::AbstractVecOrMat) = foldl((acc, op) -> op/acc, L.ops; init=x)
/(x::AbstractVecOrMat, L::DiffEqOperatorComposition) = foldl((acc, op) -> acc/op, L.ops; init=x)
\(L::DiffEqOperatorComposition, x::AbstractVecOrMat) = foldl((acc, op) -> op\acc, reverse(L.ops); init=x)
\(x::AbstractVecOrMat, L::DiffEqOperatorComposition) = foldl((acc, op) -> acc\op, reverse(L.ops); init=x)
function mul!(y::AbstractVector, L::DiffEqOperatorComposition, b::AbstractVector)
  mul!(L.caches[1], L.ops[1], b)
  for i in 2:length(L.ops) - 1
    mul!(L.caches[i], L.ops[i], L.caches[i-1])
  end
  mul!(y, L.ops[end], L.caches[end])
end
function ldiv!(y::AbstractVector, L::DiffEqOperatorComposition, b::AbstractVector)
  ldiv!(L.caches[end], L.ops[end], b)
  for i in length(L.ops) - 1:-1:2
    ldiv!(L.caches[i-1], L.ops[i], L.caches[i])
  end
  ldiv!(y, L.ops[1], L.caches[1])
end
factorize(L::DiffEqOperatorComposition) = prod(factorize, reverse(L.ops))
for fact in (:lu, :lu!, :qr, :qr!, :chol, :chol!, :ldlt, :ldlt!, 
  :bkfact, :bkfact!, :lq, :lq!, :svd, :svd!)
  @eval LinearAlgebra.$fact(L::DiffEqOperatorComposition, args...) = 
    prod(op -> $fact(op, args...), reverse(L.ops))
end

# The (u,p,t) and (du,u,p,t) interface
for T in [DiffEqScaledOperator, DiffEqOperatorCombination, DiffEqOperatorComposition]
  (L::T)(u,p,t) = (update_coefficients!(L,u,p,t); L * u)
  (L::T)(du,u,p,t) = (update_coefficients!(L,u,p,t); mul!(du,L,u))
end
