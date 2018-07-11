"""
    DiffEqArrayOperator(A[; update_func])

Represents a time-dependent linear operator given by an AbstractMatrix. The
update function is called by `update_coefficients!` and is assumed to have
the following signature:

    update_func(A::AbstractMatrix,u,p,t) -> [modifies A]

You can also use `setval!(α,A)` to bypass the `update_coefficients!` interface
and directly mutate the array's value.
"""
mutable struct DiffEqArrayOperator{T,AType<:AbstractMatrix{T},F} <: AbstractDiffEqLinearOperator{T}
  A::AType
  update_func::F
  DiffEqArrayOperator(A::AType; update_func=DEFAULT_UPDATE_FUNC()) where {AType} = 
    new{eltype(A),AType,typeof(update_func)}(A, update_func)
end

update_coefficients!(L::DiffEqArrayOperator,u,p,t) = (L.update_func(L.A,u,p,t); L)
setval!(L::DiffEqArrayOperator, A) = (L.A = A; L)
is_constant(L::DiffEqArrayOperator) = L.update_func == DEFAULT_UPDATE_FUNC()
(L::DiffEqArrayOperator)(u,p,t) = (update_coefficients!(L,u,p,t); L.A * u)
(L::DiffEqArrayOperator)(du,u,p,t) = (update_coefficients!(L,u,p,t); mul!(du, L.A, u))

# Forward operations that use the underlying array
for pred in (:isreal, :issymmetric, :ishermitian, :isposdef)
  @eval LinearAlgebra.$pred(L::DiffEqArrayOperator) = $pred(L.A)
end
size(L::DiffEqArrayOperator) = size(L.A)
size(L::DiffEqArrayOperator, m) = size(L.A, m)
opnorm(L::DiffEqArrayOperator, p::Real=2) = opnorm(L.A, p)
getindex(L::DiffEqArrayOperator, i::Int) = L.A[i]
getindex(L::DiffEqArrayOperator, I::Vararg{Int, N}) where {N} = L.A[I...]
setindex!(L::DiffEqArrayOperator, v, i::Int) = (L.A[i] = v)
setindex!(L::DiffEqArrayOperator, v, I::Vararg{Int, N}) where {N} = (L.A[I...] = v)
*(L::DiffEqArrayOperator, x) = L.A * x
*(x, L::DiffEqArrayOperator) = x * L.A
/(L::DiffEqArrayOperator, x) = L.A / x
/(x, L::DiffEqArrayOperator) = x / L.A
mul!(Y, L::DiffEqArrayOperator, B) = mul!(Y, L.A, B)
ldiv!(Y, L::DiffEqArrayOperator, B) = ldiv!(Y, L.A, B)

# Forward operations that use the full matrix
Matrix(L::DiffEqArrayOperator) = Matrix(L.A)
Base.exp(L::DiffEqArrayOperator) = exp(Matrix(L))

# Factorization
struct FactorizedDiffEqArrayOperator{T<:Number,FType<:Factorization{T}} <: AbstractDiffEqLinearOperator{T}
  F::FType
end

factorize(L::DiffEqArrayOperator) = FactorizedDiffEqArrayOperator(factorize(L.A))
for fact in (:lu, :lu!, :qr, :qr!, :chol, :chol!, :ldlt, :ldlt!,
  :bkfact, :bkfact!, :lq, :lq!, :svd, :svd!)
  @eval LinearAlgebra.$fact(L::DiffEqArrayOperator, args...) = FactorizedDiffEqArrayOperator($fact(L.A, args...))
end

ldiv!(Y, L::FactorizedDiffEqArrayOperator, B) = ldiv!(Y, L.F, B)
\(L::FactorizedDiffEqArrayOperator, x) = L.F \ x
