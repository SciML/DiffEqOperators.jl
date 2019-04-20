### jacvec functions
### should be upstreamed to ForwardDiff

struct JacVecTag end

# J(f(x))*v
function jacvec!(du, f, x, v,
                 cache1 = ForwardDiff.Dual{JacVecTag}.(x, v),
                 cache2 = ForwardDiff.Dual{JacVecTag}.(x, v))
    cache1 .= ForwardDiff.Dual{JacVecTag}.(x, v)
    f(cache2,cache1)
    du .= ForwardDiff.partials.(cache2, 1)
end
function jacvec(f, x, v)
    ForwardDiff.partials.(f(ForwardDiff.Dual{JacVecTag}.(x, v)), 1)
end

#=
### vecjec
### Not useful right now

"""
    vecjac(f, x, v) -> u
``v'J(f(x))``
"""
function vecjac(f, x, v)
    tp = ReverseDiff.InstructionTape()
    tx = ReverseDiff.track(x, tp)
    ty = f(tx)
    ReverseDiff.increment_deriv!(ty, v)
    ReverseDiff.reverse_pass!(tp)
    return ReverseDiff.deriv(tx)'
end
=#

### Operator Implementation

mutable struct JacVecOperator{T,F,T1,T2,uType,P,tType} <: DiffEqBase.AbstractDiffEqLinearOperator{T}
    f::F
    cache1::T1
    cache2::T2
    u::uType
    p::P
    t::tType

    function JacVecOperator{T}(f,p=nothing,t::Union{Nothing,Number}=nothing) where T
        p===nothing ? P = Any : P = typeof(p)
        t===nothing ? tType = Any : tType = typeof(t)
        new{T,typeof(f),Nothing,Nothing,Any,P,tType}(f,nothing,nothing,nothing,nothing,nothing)
    end
    function JacVecOperator{T}(f,u::AbstractArray,p=nothing,t::Union{Nothing,Number}=nothing) where T
        cache1 = ForwardDiff.Dual{JacVecTag}.(u, u)
        cache2 = ForwardDiff.Dual{JacVecTag}.(u, u)
        p===nothing ? P = Any : P = typeof(p)
        t===nothing ? tType = Any : tType = typeof(t)
        new{T,typeof(f),typeof(cache1),typeof(cache2),typeof(u),P,tType}(f,cache1,cache2,u,p,t)
    end
    function JacVecOperator(f,u,args...)
        JacVecOperator{eltype(u)}(f,u,args...)
    end

end

Base.size(L::JacVecOperator) = (length(L.cache1),length(L.cache1))
Base.size(L::JacVecOperator,i::Int) = length(L.cache1)
function update_coefficients!(L::JacVecOperator,u,p,t)
    L.u = u
    L.p = p
    L.t = t
end

# Interpret the call as df/du * u

function (L::JacVecOperator)(u,p,t::Number)
    update_coefficients!(L,u,p,t)
    L*u
end

function (L::JacVecOperator)(du,u,p,t::Number)
    update_coefficients!(L,u,p,t)
    mul!(du,L,u)
end

Base.:*(L::JacVecOperator,x::AbstractVector) = jacvec(_u->L.f(_u,L.p,L.t),L.u,x)

function LinearAlgebra.mul!(du::AbstractVector,L::JacVecOperator,x::AbstractVector)
    let p=L.p,t=L.t
        if L.cache1 === nothing
            jacvec!(du,(_du,_u)->L.f(_du,_u,p,t),L.u,x)
        else
            jacvec!(du,(_du,_u)->L.f(_du,_u,p,t),L.u,x,L.cache1,L.cache2)
        end
    end
end
