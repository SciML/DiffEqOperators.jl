### AbstractDiffEqLinearOperator defined by an array and update functions
mutable struct DiffEqArrayOperator{T,Arr<:Union{T,AbstractMatrix{T}},Sca,F} <: AbstractDiffEqLinearOperator{T}
    A::Arr
    α::Sca
    _isreal::Bool
    _issymmetric::Bool
    _ishermitian::Bool
    _isposdef::Bool
    update_func::F
end

DEFAULT_UPDATE_FUNC = (L,u,p,t)->nothing

function DiffEqArrayOperator(A::Number,α=1.0,
                             update_func = DEFAULT_UPDATE_FUNC)
    if (typeof(α) <: Number)
        _α = DiffEqScalar(nothing,α)
    elseif (typeof(α) <: DiffEqScalar) # Must be a DiffEqScalar already
        _α = α
    else # Assume it's some kind of function
        # Wrapping the function call in one() should solve any cases
        # where the function is not well-behaved at 0.0, as long as
        # the return type is correct.
        _α = DiffEqScalar(α,one(α(0.0)))
    end
    DiffEqArrayOperator{typeof(A),typeof(A),typeof(_α),
    typeof(update_func)}(
    A,_α,isreal(A),issymmetric(A),ishermitian(A),
    isposdef(A),update_func)
end

function DiffEqArrayOperator(A::AbstractMatrix{T},α=1.0,
                             update_func = DEFAULT_UPDATE_FUNC) where T
    if (typeof(α) <: Number)
        _α = DiffEqScalar(nothing,α)
    elseif (typeof(α) <: DiffEqScalar) # Must be a DiffEqScalar already
        _α = α
    else # Assume it's some kind of function
        # Wrapping the function call in one() should solve any cases
        # where the function is not well-behaved at 0.0, as long as
        # the return type is correct.
        _α = DiffEqScalar(α,one(α(0.0)))
    end
    DiffEqArrayOperator{T,typeof(A),typeof(_α),
    typeof(update_func)}(
    A,_α,isreal(A),issymmetric(A),ishermitian(A),
    isposdef(A),update_func)
end

Base.isreal(L::DiffEqArrayOperator) = L._isreal
Base.issymmetric(L::DiffEqArrayOperator) = L._issymmetric
Base.ishermitian(L::DiffEqArrayOperator) = L._ishermitian
Base.isposdef(L::DiffEqArrayOperator) = L._isposdef
DiffEqBase.is_constant(L::DiffEqArrayOperator) = L.update_func == DEFAULT_UPDATE_FUNC
Base.full(L::DiffEqArrayOperator) = full(L.A) .* L.α.coeff
Base.expm(L::DiffEqArrayOperator) = expm(full(L))
DiffEqBase.has_expm(L::DiffEqArrayOperator) = true
Base.size(L::DiffEqArrayOperator) = size(L.A)
Base.norm(L::DiffEqArrayOperator, p::Real=2) = norm(L.A, p) * abs(L.α.coeff)
DiffEqBase.update_coefficients!(L::DiffEqArrayOperator,u,p,t) = (L.update_func(L.A,u,p,t); L.α = L.α(t); nothing)
DiffEqBase.update_coefficients(L::DiffEqArrayOperator,u,p,t)  = (L.update_func(L.A,u,p,t); L.α = L.α(t); L)

function (L::DiffEqArrayOperator)(u,p,t)
  update_coefficients!(L,u,p,t)
  L*u
end

function (L::DiffEqArrayOperator)(du,u,p,t)
  update_coefficients!(L,u,p,t)
  A_mul_B!(du,L,u)
end

### Forward some extra operations
function Base.:*(α::Number,L::DiffEqArrayOperator)
    DiffEqArrayOperator(L.A,DiffEqScalar(L.α.func,L.α.coeff*α),L.update_func)
end

function Base.:*(α::Number,L::DiffEqArrayOperator{T,Arr,Sca,F}) where {T,Arr<:Number,Sca,F}
    L.α.coeff*α*L.A
end

Base.:*(L::DiffEqArrayOperator,α::Number) = α*L
Base.:*(L::DiffEqArrayOperator,b::AbstractVector) = L.α.coeff*L.A*b
Base.:*(L::DiffEqArrayOperator,b::AbstractArray) = L.α.coeff*L.A*b

function Base.A_mul_B!(v::AbstractVector,L::DiffEqArrayOperator,b::AbstractVector)
    A_mul_B!(v,L.A,b)
    scale!(v,L.α.coeff)
end

function Base.A_mul_B!(v::AbstractArray,L::DiffEqArrayOperator,b::AbstractArray)
    A_mul_B!(v,L.A,b)
    scale!(v,L.α.coeff)
end

function Base.A_ldiv_B!(x,L::DiffEqArrayOperator, b::AbstractArray)
    A_ldiv_B!(x,L.A,b)
    scale!(x,inv(L.α.coeff))
end

function Base.:/(x,L::DiffEqArrayOperator)
    x/(L.α.coeff*L.A)
end

function Base.:/(L::DiffEqArrayOperator,x)
    L.α.coeff*L.A/x
end

"""
FactorizedDiffEqArrayOperator{T,I}

A helper function for holding factorized version of the DiffEqArrayOperator
"""
struct FactorizedDiffEqArrayOperator{T,I}
    A::T
    inv_coeff::I
end

Base.factorize(L::DiffEqArrayOperator)         = FactorizedDiffEqArrayOperator(factorize(L.A),inv(L.α.coeff))
Base.lufact(L::DiffEqArrayOperator,args...)    = FactorizedDiffEqArrayOperator(lufact(L.A,args...),inv(L.α.coeff))
Base.lufact!(L::DiffEqArrayOperator,args...)   = FactorizedDiffEqArrayOperator(lufact!(L.A,args...),inv(L.α.coeff))
Base.qrfact(L::DiffEqArrayOperator,args...)    = FactorizedDiffEqArrayOperator(qrfact(L.A,args...),inv(L.α.coeff))
Base.qrfact!(L::DiffEqArrayOperator,args...)   = FactorizedDiffEqArrayOperator(qrfact!(L.A,args...),inv(L.α.coeff))
Base.cholfact(L::DiffEqArrayOperator,args...)  = FactorizedDiffEqArrayOperator(cholfact(L.A,args...),inv(L.α.coeff))
Base.cholfact!(L::DiffEqArrayOperator,args...) = FactorizedDiffEqArrayOperator(cholfact!(L.A,args...),inv(L.α.coeff))
Base.ldltfact(L::DiffEqArrayOperator,args...)  = FactorizedDiffEqArrayOperator(ldltfact(L.A,args...),inv(L.α.coeff))
Base.ldltfact!(L::DiffEqArrayOperator,args...) = FactorizedDiffEqArrayOperator(ldltfact!(L.A,args...),inv(L.α.coeff))
Base.bkfact(L::DiffEqArrayOperator,args...)    = FactorizedDiffEqArrayOperator(bkfact(L.A,args...),inv(L.α.coeff))
Base.bkfact!(L::DiffEqArrayOperator,args...)   = FactorizedDiffEqArrayOperator(bkfact!(L.A,args...),inv(L.α.coeff))
Base.lqfact(L::DiffEqArrayOperator,args...)    = FactorizedDiffEqArrayOperator(lqfact(L.A,args...),inv(L.α.coeff))
Base.lqfact!(L::DiffEqArrayOperator,args...)   = FactorizedDiffEqArrayOperator(lqfact!(L.A,args...),inv(L.α.coeff))
Base.svdfact(L::DiffEqArrayOperator,args...)   = FactorizedDiffEqArrayOperator(svdfact(L.A,args...),inv(L.α.coeff))
Base.svdfact!(L::DiffEqArrayOperator,args...)  = FactorizedDiffEqArrayOperator(svdfact!(L.A,args...),inv(L.α.coeff))

function Base.A_ldiv_B!(x,L::FactorizedDiffEqArrayOperator, b::AbstractArray)
    A_ldiv_B!(x,L.A,b)
    scale!(x,inv(L.inv_coeff))
end

function Base.:\(L::FactorizedDiffEqArrayOperator, b::AbstractArray)
    (L.A \ b) * L.inv_coeff
end

@inline Base.getindex(L::DiffEqArrayOperator,i::Int) = L.A[i]
@inline Base.getindex(L::DiffEqArrayOperator,I::Vararg{Int, N}) where {N} = L.A[I...]
@inline Base.setindex!(L::DiffEqArrayOperator, v, i::Int) = (L.A[i]=v)
@inline Base.setindex!(L::DiffEqArrayOperator, v, I::Vararg{Int, N}) where {N} = (L.A[I...]=v)
