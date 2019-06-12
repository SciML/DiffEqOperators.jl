# For a boundary condition, the variables correspond to al*u(0) + bl*u'(0) = cl
struct RobinBC{T}
    al::T
    bl::T
    cl::T
    dx_l::T # should grid size be Real or more general?
    ar::T
    br::T
    cr::T
    dx_r::T

    function RobinBC(al::T, bl::T, cl::T, dx_l::T, ar::T, br::T, cr::T, dx_r::T) where T
        return new{T}(al, bl, cl, dx_l, ar, br, cr, dx_r)
    end
end

struct RobinBCExtended{T,T2<:AbstractVector{T}} <: AbstractVector{T}
    l::T
    r::T
    u::T2

    function RobinBCExtended(u::T2, al::T, bl::T, cl::T,
                                    dx_l::T, ar::T, br::T, cr::T, dx_r::T) where
                                    {T,T2<:AbstractVector{T}}
        l = (cl - bl*u[1]/dx_l)*(1/(al-bl/dx_l))
        r = (cr + br*u[end]/dx_r)*(1/(ar+br/dx_r))
        return new{T,T2}(l, r, u)

    end
end


Base.:*(Q::RobinBC,u) = RobinBCExtended(u, Q.al, Q.bl, Q.cl, Q.dx_l, Q.ar, Q.br, Q.cr, Q.dx_r)
Base.length(Q::RobinBCExtended) = length(Q.u) + 2
Base.size(Q::RobinBCExtended) = (length(Q.u)+2,)
Base.lastindex(Q::RobinBCExtended) = Base.length(Q)

function Base.getindex(Q::RobinBCExtended,i)
    if i == 1
        return Q.l
    elseif i == length(Q)
        return Q.r
    else
        return Q.u[i-1]
    end
end

function LinearAlgebra.Array(Q::RobinBC, N::Int)
    Q_L = [(-Q.bl/Q.dx_l)/(Q.al-Q.bl/Q.dx_l) transpose(zeros(N-1)); Diagonal(ones(N)); transpose(zeros(N-1)) (Q.br/Q.dx_r)/(Q.ar+Q.br/Q.dx_r)]
    Q_b = [Q.cl/(Q.al-Q.bl/Q.dx_l); zeros(N); Q.cr/(Q.ar+Q.br/Q.dx_r)]
    return (Q_L, Q_b)
end

function LinearAlgebra.Array(Q::RobinBCExtended)
    return [Q.l; Q.u; Q.r]
end
