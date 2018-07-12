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

*(α::DiffEqScalar, x) = α.val * x
*(x, α::DiffEqScalar) = x * α.val
lmul!(α::DiffEqScalar, B) = lmul!(α.val, B)
rmul!(B, α::DiffEqScalar) = rmul!(B, α.val)
mul!(Y, α::DiffEqScalar, B) = mul!(Y, α.val, B)
axpy!(α::DiffEqScalar, X, Y) = axpy!(α.val, X, Y)

(α::DiffEqScalar)(u,p,t) = (update_coefficients!(α,u,p,t); α.val * u)
(α::DiffEqScalar)(du,u,p,t) = (update_coefficients!(α,u,p,t); @. du = α.val * u)
