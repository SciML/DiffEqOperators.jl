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

update_coefficients!(α::DiffEqScalar,u,p,t) = (α.val = α.update_func(α.val,u,p,t); α)
setval!(α::DiffEqScalar, val) = (α.val = val; α)
is_constant(α::DiffEqScalar) = α.update_func == DEFAULT_UPDATE_FUNC

for op in (:*, :/, :\)
  @eval $op(α::DiffEqScalar{T,F}, x::AbstractVecOrMat{T}) where {T,F} = $op(α.val, x)
  @eval $op(x::AbstractVecOrMat{T}, α::DiffEqScalar{T,F}) where {T,F} = $op(x, α.val)
end
lmul!(α::DiffEqScalar{T,F}, B::AbstractVecOrMat{T}) where {T,F} = lmul!(α.val, B)
rmul!(B::AbstractVecOrMat{T}, α::DiffEqScalar{T,F}) where {T,F} = rmul!(B, α.val)
mul!(Y::AbstractVecOrMat{T}, α::DiffEqScalar{T,F},
  B::AbstractVecOrMat{T}) where {T,F} = mul!(Y, α.val, B)
axpy!(α::DiffEqScalar{T,F}, X::AbstractVecOrMat{T},
  Y::AbstractVecOrMat{T}) where {T,F} = axpy!(α.val, X, Y)
Base.abs(α::DiffEqScalar) = abs(α.val)

(α::DiffEqScalar)(u,p,t) = (update_coefficients!(α,u,p,t); α.val * u)
(α::DiffEqScalar)(du,u,p,t) = (update_coefficients!(α,u,p,t); @. du = α.val * u)
