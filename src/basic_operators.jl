"""
    DiffEqScalar(val[; update_func])

Represents a time-dependent scalar/scaling operator. The update function
is called by `update_coefficients!` and is assumed to have the following
signature:

    update_func(oldval,u,p,t) -> newval

You can also use `setval!(α,val)` to bypass the `update_coefficients!`
interface and directly mutate the scalar's value.
"""
mutable struct DiffEqScalar{T<:Number,F} <: AbstractDiffEqLinearOperator{T}
  val::T
  update_func::F
  DiffEqScalar(val::T; update_func=DEFAULT_UPDATE_FUNC) where {T} =
    new{T,typeof(update_func)}(val, update_func)
end

size(::DiffEqScalar) = ()
size(::DiffEqScalar, ::Integer) = 1
update_coefficients!(α::DiffEqScalar,u,p,t) = (α.val = α.update_func(α.val,u,p,t); α)
setval!(α::DiffEqScalar, val) = (α.val = val; α)
is_constant(α::DiffEqScalar) = α.update_func == DEFAULT_UPDATE_FUNC

for op in (:*, :/, :\)
  @eval $op(α::DiffEqScalar, x::AbstractVecOrMat) = $op(α.val, x)
  @eval $op(x::AbstractVecOrMat, α::DiffEqScalar) = $op(x, α.val)
end
lmul!(α::DiffEqScalar, B::AbstractVecOrMat) = lmul!(α.val, B)
rmul!(B::AbstractVecOrMat, α::DiffEqScalar) = rmul!(B, α.val)
mul!(Y::AbstractVecOrMat, α::DiffEqScalar, B::AbstractVecOrMat) = mul!(Y, α.val, B)
axpy!(α::DiffEqScalar, X::AbstractVecOrMat, Y::AbstractVecOrMat) = axpy!(α.val, X, Y)
Base.abs(α::DiffEqScalar) = abs(α.val)

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
  DiffEqArrayOperator(A::AType; update_func=DEFAULT_UPDATE_FUNC) where {AType} = 
    new{eltype(A),AType,typeof(update_func)}(A, update_func)
end

update_coefficients!(L::DiffEqArrayOperator,u,p,t) = (L.update_func(L.A,u,p,t); L)
setval!(L::DiffEqArrayOperator, A) = (L.A = A; L)
is_constant(L::DiffEqArrayOperator) = L.update_func == DEFAULT_UPDATE_FUNC

convert(::Type{AbstractMatrix}, L::DiffEqArrayOperator) = L.A
setindex!(L::DiffEqArrayOperator, v, i::Int) = (L.A[i] = v)
setindex!(L::DiffEqArrayOperator, v, I::Vararg{Int, N}) where {N} = (L.A[I...] = v)

"""
    FactorizedDiffEqArrayOperator(F)

Like DiffEqArrayOperator, but stores a Factorization instead.

Supports left division and `ldiv!` when applied to an array.
"""
struct FactorizedDiffEqArrayOperator{T<:Number,FType<:Factorization{T}} <: AbstractDiffEqLinearOperator{T}
  F::FType
end

Matrix(L::FactorizedDiffEqArrayOperator) = Matrix(L.F)
convert(::Type{AbstractMatrix}, L::FactorizedDiffEqArrayOperator) = convert(AbstractMatrix, L.F)
size(L::FactorizedDiffEqArrayOperator, args...) = size(L.F, args...)
ldiv!(Y::AbstractVecOrMat, L::FactorizedDiffEqArrayOperator, B::AbstractVecOrMat) = ldiv!(Y, L.F, B)
\(L::FactorizedDiffEqArrayOperator, x::AbstractVecOrMat) = L.F \ x
