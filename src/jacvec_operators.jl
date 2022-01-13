### Operator Implementation
mutable struct JacVecOperator{T,F,T1,T2,uType,P,tType,O} <:
               DiffEqBase.AbstractDiffEqLinearOperator{T}
    f::F
    cache1::T1
    cache2::T2
    u::uType
    p::P
    t::tType
    autodiff::Bool
    ishermitian::Bool
    opnorm::O

    function JacVecOperator{T}(
        f,
        p = nothing,
        t::Union{Nothing,Number} = nothing;
        autodiff = true,
        ishermitian = false,
        opnorm = true,
    ) where {T}
        p === nothing ? P = Any : P = typeof(p)
        t === nothing ? tType = Any : tType = typeof(t)
        new{T,typeof(f),Nothing,Nothing,Any,P,tType,typeof(opnorm)}(
            f,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
            autodiff,
            ishermitian,
        )
    end
    function JacVecOperator{T}(
        f,
        u::AbstractArray,
        p = nothing,
        t::Union{Nothing,Number} = nothing;
        autodiff = true,
        ishermitian = false,
        opnorm = true,
    ) where {T}
        if autodiff
            cache1 = ForwardDiff.Dual{typeof(ForwardDiff.Tag(SparseDiffTools.DeivVecTag(),eltype(x))),eltype(x),1}.(u, u)
            cache2 = ForwardDiff.Dual{typeof(ForwardDiff.Tag(SparseDiffTools.DeivVecTag(),eltype(x))),eltype(x),1}.(u, u)
        else
            cache1 = similar(u)
            cache2 = similar(u)
        end
        p === nothing ? P = Any : P = typeof(p)
        t === nothing ? tType = Any : tType = typeof(t)
        new{
            T,
            typeof(f),
            typeof(cache1),
            typeof(cache2),
            typeof(u),
            P,
            tType,
            typeof(opnorm),
        }(
            f,
            cache1,
            cache2,
            u,
            p,
            t,
            autodiff,
            ishermitian,
            opnorm,
        )
    end
    function JacVecOperator(f, u, args...; kwargs...)
        JacVecOperator{eltype(u)}(f, u, args...; kwargs...)
    end
end

LinearAlgebra.opnorm(L::JacVecOperator, p::Real = 2) = L.opnorm
LinearAlgebra.ishermitian(L::JacVecOperator) = L.ishermitian

Base.size(L::JacVecOperator) = (length(L.cache1), length(L.cache1))
Base.size(L::JacVecOperator, i::Int) = length(L.cache1)
function update_coefficients!(L::JacVecOperator, u, p, t)
    L.u = u
    L.p = p
    L.t = t
    !L.autodiff && L.cache1 !== nothing && L.f(L.cache1, L.u, L.p, L.t)
end

# Interpret the call as df/du * u

function (L::JacVecOperator)(u, p, t::Number)
    update_coefficients!(L, u, p, t)
    L * u
end

function (L::JacVecOperator)(du, u, p, t::Number)
    update_coefficients!(L, u, p, t)
    mul!(du, L, u)
end


function Base.:*(L::JacVecOperator, x::AbstractVector)
    if hasmethod(L.f, typeof.((L.u, L.p, L.t)))
        return vec(
            L.autodiff ? auto_jacvec(_u -> L.f(_u, L.p, L.t), L.u, x) :
            num_jacvec(_u -> L.f(_u, L.p, L.t), L.u, x),
        )
    end
    return mul!(similar(vec(L.u)), L, x)
end


function LinearAlgebra.mul!(
    du::AbstractVector,
    L::JacVecOperator,
    x::AbstractVector,
)
    du = reshape(du, size(L.u))
    let p = L.p, t = L.t
        if L.cache1 === nothing
            if L.autodiff
                auto_jacvec!(du, (_du, _u) -> L.f(_du, _u, p, t), L.u, x)
            else
                num_jacvec!(
                    du,
                    (_du, _u) -> L.f(_du, _u, p, t),
                    L.u,
                    x;
                    compute_f0 = true,
                )
            end
        else
            if L.autodiff
                auto_jacvec!(
                    du,
                    (_du, _u) -> L.f(_du, _u, p, t),
                    L.u,
                    x,
                    L.cache1,
                    L.cache2,
                )
            else
                num_jacvec!(
                    du,
                    (_du, _u) -> L.f(_du, _u, p, t),
                    L.u,
                    x,
                    L.cache1,
                    L.cache2;
                    compute_f0 = true,
                )
            end
        end
    end
    return vec(du)
end

function Base.resize!(J::JacVecOperator, i)
    resize!(J.cache1, i)
    resize!(J.cache2, i)
end

### AnalyticalOperator Implementation

mutable struct AnalyticalJacVecOperator{T,F,uType,P,tType,O} <:
               DiffEqBase.AbstractDiffEqLinearOperator{T}
    f::F
    u::uType
    p::P
    t::tType
    ishermitian::Bool
    opnorm::O
    function AnalyticalJacVecOperator{T}(
        f,
        u = nothing,
        p = nothing,
        t::Union{Nothing,Number} = nothing;
        ishermitian = false,
        opnorm = true,
    ) where {T}
        u === nothing ? uType = Any : uType = typeof(u)
        p === nothing ? P = Any : P = typeof(p)
        t === nothing ? tType = Any : tType = typeof(t)
        new{T,typeof(f),uType,P,tType,typeof(opnorm)}(
            f,
            u,
            p,
            t,
            ishermitian,
            opnorm,
        )
    end
    function AnalyticalJacVecOperator(f, u, args...; kwargs...)
        AnalyticalJacVecOperator{eltype(u)}(f, u, args...; kwargs...)
    end
end

LinearAlgebra.opnorm(L::AnalyticalJacVecOperator, p::Real = 2) = L.opnorm
LinearAlgebra.ishermitian(L::AnalyticalJacVecOperator) = L.ishermitian

Base.size(L::AnalyticalJacVecOperator) = (length(L.u), length(L.u))
Base.size(L::AnalyticalJacVecOperator, i::Int) = length(L.u)
function update_coefficients!(L::AnalyticalJacVecOperator, u, p, t)
    L.u = u
    L.p = p
    L.t = t
    return L
end

# Interpret the call as df/du * u

function (L::AnalyticalJacVecOperator)(u, p, t::Number)
    update_coefficients!(L, u, p, t)
    L * u
end

function (L::AnalyticalJacVecOperator)(du, u, p, t::Number)
    update_coefficients!(L, u, p, t)
    mul!(du, L, u)
end

Base.:*(L::AnalyticalJacVecOperator, x::AbstractVector) = L.f(x, L.u, L.p, L.t)

function LinearAlgebra.mul!(
    du::AbstractVector,
    L::AnalyticalJacVecOperator,
    x::AbstractVector,
)
    L.f(du, x, L.u, L.p, L.t)
end
