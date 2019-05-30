struct DirichletBC{T<:Real} <: AbstractDerivativeOperator{T}
    l::T
    r::T
end

struct DirichletBCExtended{T<:Real,S<:SVector} <: AbstractDerivativeOperator{T}
    derivative_order    :: Int
    approximation_order :: Int
    dx                  :: T
    dimension           :: Int
    stencil_length      :: Int
    ghost_point_count   :: Int
    stencil_coefs       :: S
    u                   :: AbstractVector{T}
    lbc                 :: T
    rbc                 :: T
    v                   :: Ref{AbstractVector{T}}

    function DirichletBCExtended{T,S}(u::AbstractVector{T}, lbc::T, rbc::T, derivative_order::Int,
                                    approximation_order::Int, dx::T) where
                                    {T<:Real,S<:SVector}
        u                    = u
        lbc                  = lbc
        rbc                  = rbc
        dimension            = length(u)
        dx                   = dx
        stencil_length       = derivative_order + approximation_order - 1 + (derivative_order+approximation_order)%2
        ghost_point_count    = div(stencil_length,2)
        grid_step            = one(T)
        stencil_coefs        = convert(SVector{stencil_length, T}, calculate_weights(derivative_order, zero(T),
                               grid_step .* collect(-div(stencil_length,2) : 1 : div(stencil_length,2))))
        v                    = Vector{T}
        # zeros(T, dimension + ghost_point_count)

        out = new(derivative_order, approximation_order, dx, dimension, stencil_length, ghost_point_count, stencil_coefs, u, lbc, rbc)
        convolve_interior!(v, u, out)
        return out
    end
    DirichletBCExtended{T}(u::AbstractVector{T},lbc::T,rbc::T,dorder::Int,aorder::Int,dx::T) where {T<:Real} =
        DirichletBCExtended{T, SVector{dorder+aorder-1+(dorder+aorder)%2,T}}(u, lbc, rbc, dorder, aorder, dx)
end


Base.:*(Q::DirichletBC,u) = DirichletBCExtended(u, Q.l, Q.r, 2, 2, 1.0)
Base.length(Q::DirichletBCExtended) = length(Q.u) + 2
Base.lastindex(Q::DirichletBCExtended) = Base.length(Q)

function Base.getindex(Q::DirichletBCExtended,i)
    if i == 1
        return Q.lbc
    elseif i == length(Q)
        return Q.rbc
    else
        return Q.u[i-1]
    end
end


#################################################################################################


(L::DirichletBCExtended)(u,p,t) = L*u
(L::DirichletBCExtended)(du,u,p,t) = mul!(du,L,u)
get_type(::DirichletBCExtended{A,B}) where {A,B} = A

#=
    The Inf opnorm can be calculated easily using the stencil coeffiicents, while other opnorms
    default to compute from the full matrix form.
=#
function LinearAlgebra.opnorm(A::DirichletBCExtended{T,S}, p::Real=2) where {T,S}
    if p == Inf
        sum(abs.(A.stencil_coefs)) / A.dx^A.derivative_order
    else
        opnorm(convert(Array,A), p)
    end
end
