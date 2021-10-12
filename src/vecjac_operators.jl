function auto_vecjac!(du, f, x, v, cache1 = nothing, cache2 = nothing)
    DiffEqBase.numargs(f) != 1 && error("For inplace function use autodiff = false")
    du .= auto_vecjac(f, x, v)
end

function auto_vecjac(f, x, v)
    vv, back = Zygote.pullback(f, x)
    return vec(back(reshape(v, size(vv)))[1])
end

function num_vecjac!(
    du,
    f,
    x,
    v,
    cache1 = similar(v),
    cache2 = similar(v);
    compute_f0 = true,
)
    compute_f0 && (f(cache1, x))
    T = eltype(x)
    # Should it be min? max? mean?
    ϵ = sqrt(eps(real(T))) * max(one(real(T)), abs(norm(x)))
    vv = reshape(v, size(x))
    for i in 1:length(x)
        x[i] += ϵ
        f(cache2, x)
        x[i] -= ϵ
        du[i] = (((cache2 .- cache1) ./ ϵ)' * vv)[1]
    end
    return du
end

function num_vecjac(f, x, v, f0 = nothing)
    vv = reshape(v, axes(x))
    f0 === nothing ? _f0 = f(x) : _f0 = f0
    T = eltype(x)
    # Should it be min? max? mean?
    ϵ = sqrt(eps(real(T))) * max(one(real(T)), abs(norm(x)))
    du = similar(x)
    for i in 1:length(x)
        x[i] += ϵ
        f0 = f(x)
        x[i] -= ϵ
        du[i] = (((f0 .- _f0) ./ ϵ)' * vv)[1]
    end
    return du
end

mutable struct VecJacOperator{T,F,T1,T2,uType,P,tType,O} <:
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

    function VecJacOperator{T}(
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
    function VecJacOperator{T}(
        f,
        u::AbstractArray,
        p = nothing,
        t::Union{Nothing,Number} = nothing;
        autodiff = true,
        ishermitian = false,
        opnorm = true,
    ) where {T}
        cache1 = similar(u)
        cache2 = similar(u)
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
    function VecJacOperator(f, u, args...; kwargs...)
        VecJacOperator{eltype(u)}(f, u, args...; kwargs...)
    end
end

LinearAlgebra.opnorm(L::VecJacOperator, p::Real = 2) = L.opnorm
LinearAlgebra.ishermitian(L::VecJacOperator) = L.ishermitian

Base.size(L::VecJacOperator) = (length(L.cache1), length(L.cache1))
Base.size(L::VecJacOperator, i::Int) = length(L.cache1)
function update_coefficients!(L::VecJacOperator, u, p, t)
    L.u = u
    L.p = p
    L.t = t
    !L.autodiff && L.cache1 !== nothing && L.f(L.cache1, L.u, L.p, L.t)
end

# Interpret the call as df/du' * u
function (L::VecJacOperator)(u, p, t::Number)
    update_coefficients!(L, u, p, t)
    L * u
end

function (L::VecJacOperator)(du, u, p, t::Number)
    update_coefficients!(L, u, p, t)
    mul!(du, L, u)
end


function Base.:*(L::VecJacOperator,x::AbstractVector)
    if DiffEqBase.numargs(L.f) == 3
        return L.autodiff ? auto_vecjac(_u->L.f(_u,L.p,L.t),L.u,x) : num_vecjac(_u->L.f(_u,L.p,L.t),L.u,x)
    end
    return mul!(similar(vec(L.u)), L, x)
end

function LinearAlgebra.mul!(
    du::AbstractVector,
    L::VecJacOperator,
    x::AbstractVector,
)
    let p = L.p, t = L.t
        if L.cache1 === nothing
            if L.autodiff
                if DiffEqBase.numargs(L.f) == 4
                    auto_vecjac!(du, (_du, _u) -> L.f(_du, _u, p, t), L.u, x)
                else
                    auto_vecjac!(du, _u -> L.f(_u, p, t), L.u, x)
                end
            else
                num_vecjac!(
                    du,
                    (_du, _u) -> L.f(_du, _u, p, t),
                    L.u,
                    x;
                    compute_f0 = false,
                )
            end
        else
            if L.autodiff
                if DiffEqBase.numargs(L.f) == 4
                    auto_vecjac!(du, (_du, _u) -> L.f(_du, _u, p, t), L.u, x, L.cache1, L.cache2)
                else
                    auto_vecjac!(du, _u -> L.f(_u, p, t), L.u, x, L.cache1, L.cache2)
                end
            else
                num_vecjac!(
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
end

function Base.resize!(J::VecJacOperator, i)
    resize!(J.cache1, i)
    resize!(J.cache2, i)
end
