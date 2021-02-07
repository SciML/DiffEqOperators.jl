# The composite operators are built using basic operators (scalar, array, and
# derivative) using arithmetic or other operator compositions. The composite
# operator types are lazy and maintain the structure used to build them.

# Recursive routines that use `getops`
function update_coefficients!(L::AbstractDiffEqCompositeOperator,u,p,t)
  for op in getops(L)
    update_coefficients!(op,u,p,t)
  end
  L
end
isconstant(L::AbstractDiffEqCompositeOperator) = all(isconstant, getops(L))

# Scaled operator (α * A)
struct DiffEqScaledOperator{T,F,OpType<:AbstractDiffEqLinearOperator{T}} <: AbstractDiffEqCompositeOperator{T}
  coeff::DiffEqScalar{T,F}
  op::OpType
end
*(α::DiffEqScalar{T,F}, L::AbstractDiffEqCompositeOperator{T}) where {T,F} = DiffEqScaledOperator(α, L)
*(α::Number, L::AbstractDiffEqCompositeOperator{T}) where T = DiffEqScaledOperator(DiffEqScalar(convert(T,α)), L)
-(L::AbstractDiffEqCompositeOperator{T}) where {T} = DiffEqScalar(-one(T)) * L
getops(L::DiffEqScaledOperator) = (L.coeff, L.op)
Matrix(L::DiffEqScaledOperator) = L.coeff * Matrix(L.op)
convert(::Type{AbstractMatrix}, L::DiffEqScaledOperator) = L.coeff * convert(AbstractMatrix, L.op)

size(L::DiffEqScaledOperator, args...) = size(L.op, args...)
opnorm(L::DiffEqScaledOperator, p::Real=2) = abs(L.coeff) * opnorm(L.op, p)
getindex(L::DiffEqScaledOperator, i::Int) = L.coeff * L.op[i]
getindex(L::DiffEqScaledOperator, I::Vararg{Int, N}) where {N} =
  L.coeff * L.op[I...]
*(L::DiffEqScaledOperator, x::AbstractArray) = L.coeff * (L.op * x)
*(x::AbstractArray, L::DiffEqScaledOperator) = (L.op * x) * L.coeff
/(L::DiffEqScaledOperator, x::AbstractArray) = L.coeff * (L.op / x)
/(x::AbstractArray, L::DiffEqScaledOperator) = 1/L.coeff * (x / L.op)
\(L::DiffEqScaledOperator, x::AbstractArray) = 1/L.coeff * (L.op \ x)
\(x::AbstractArray, L::DiffEqScaledOperator) = L.coeff * (x \ L)
for N in (2,3)
  @eval begin
    mul!(Y::AbstractArray{T,$N}, L::DiffEqScaledOperator{T}, B::AbstractArray{T,$N}) where {T} =
        lmul!(Y, L.coeff, mul!(Y, L.op, B))
  end
end
ldiv!(Y::AbstractArray, L::DiffEqScaledOperator, B::AbstractArray) =
  lmul!(1/L.coeff, ldiv!(Y, L.op, B))
factorize(L::DiffEqScaledOperator) = L.coeff * factorize(L.op)
for fact in (:lu, :lu!, :qr, :qr!, :cholesky, :cholesky!, :ldlt, :ldlt!,
  :bunchkaufman, :bunchkaufman!, :lq, :lq!, :svd, :svd!)
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
        for i in 2:length(ops)
            @assert size(ops[i]) == size(ops[1]) "Operators must be of the same size to be combined! Mismatch between $(ops[i]) and $(ops[i-1]), which are operators $i and $(i-1) respectively"
        end
        if cache == nothing
            cache = zeros(T, size(ops[1], 1))
        end
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
*(L::DiffEqOperatorCombination, x::AbstractArray) = sum(op -> op * x, L.ops)
*(x::AbstractArray, L::DiffEqOperatorCombination) = sum(op -> x * op, L.ops)
/(L::DiffEqOperatorCombination, x::AbstractArray) = sum(op -> op / x, L.ops)
\(x::AbstractArray, L::DiffEqOperatorCombination) = sum(op -> x \ op, L.ops)
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
    for i in 2:length(ops)
      @assert size(ops[i-1], 1) == size(ops[i], 2) "Operations do not have compatible sizes! Mismatch between $(ops[i]) and $(ops[i-1]), which are operators $i and $(i-1) respectively."
    end

    if caches == nothing
      # Construct a list of caches to be used by mul! and ldiv!
      caches = []
      for op in ops[1:end-1]
        tmp = Vector{T}(undef, size(op, 1))
        fill!(tmp,0)
        push!(caches,tmp)
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
*(L::DiffEqOperatorComposition, x::AbstractArray) = foldl((acc, op) -> op*acc, L.ops; init=x)
*(x::AbstractArray, L::DiffEqOperatorComposition) = foldl((acc, op) -> acc*op, reverse(L.ops); init=x)
/(L::DiffEqOperatorComposition, x::AbstractArray) = foldl((acc, op) -> op/acc, L.ops; init=x)
/(x::AbstractArray, L::DiffEqOperatorComposition) = foldl((acc, op) -> acc/op, L.ops; init=x)
\(L::DiffEqOperatorComposition, x::AbstractArray) = foldl((acc, op) -> op\acc, reverse(L.ops); init=x)
\(x::AbstractArray, L::DiffEqOperatorComposition) = foldl((acc, op) -> acc\op, reverse(L.ops); init=x)
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
for fact in (:lu, :lu!, :qr, :qr!, :cholesky, :cholesky!, :ldlt, :ldlt!,
  :bunchkaufman, :bunchkaufman!, :lq, :lq!, :svd, :svd!)
  @eval LinearAlgebra.$fact(L::DiffEqOperatorComposition, args...) =
    prod(op -> $fact(op, args...), reverse(L.ops))
end
