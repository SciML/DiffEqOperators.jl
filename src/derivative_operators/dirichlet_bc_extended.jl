make_array(T, l) = zeros(T, l)

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
    stencil_coefs       :: S
    boundary_length     :: Int
    boundary_point_count:: Tuple{Int,Int}
    low_boundary_coefs  :: Ref{Vector{Vector{T}}}
    high_boundary_coefs :: Ref{Vector{Vector{T}}}
    t                   :: Ref{Int}

    function DirichletBCExtended{T,S}(derivative_order::Int,
                                            approximation_order::Int, dx::T,
                                            dimension::Int) where
                                            {T<:Real,S<:SVector}
        dimension            = dimension
        dx                   = dx
        stencil_length       = derivative_order + approximation_order - 1 + (derivative_order+approximation_order)%2
        bl                   = derivative_order + approximation_order
        boundary_length      = bl
        bpc                  = div(stencil_length,2)
        grid_step            = one(T)
        low_boundary_coefs   = Vector{T}[]
        high_boundary_coefs  = Vector{T}[]
        stencil_coefs        = convert(SVector{stencil_length, T}, calculate_weights(derivative_order,                                zero(T), grid_step .* collect(-div(stencil_length,2) : 1 : div(stencil_length,2))))
        boundary_point_count = (bpc, bpc)

        left_BC!(Val{:LO},low_boundary_coefs,stencil_length,derivative_order, grid_step,boundary_length)
        right_BC!(Val{:LO},high_boundary_coefs,stencil_length,derivative_order, grid_step,boundary_length)

        t = 0

        new(derivative_order, approximation_order, dx, dimension, stencil_length,
            stencil_coefs,
            boundary_length,
            boundary_point_count,
            low_boundary_coefs,
            high_boundary_coefs,
            t
            )
    end
    DirichletBCExtended{T}(dorder::Int,aorder::Int,dx::T,dim::Int) where {T<:Real} = DirichletBCExtended{T, SVector{dorder+aorder-1+(dorder+aorder)%2,T}}(dorder, aorder, dx, dim)
end

#################################################################################################

function left_BC!(::Type{Val{:LO}},low_boundary_coefs,stencil_length,derivative_order,
                       grid_step::T,boundary_length) where T
    # Fixes the problem of excessive boundary points
    boundary_point_count = div(stencil_length,2)
    mid                  = div(stencil_length,2)

    for i in 1 : boundary_point_count
        # One-sided stencils require more points for same approximation order
        # TODO: I don't know if this is the correct stencil length for i > 1?
        push!(low_boundary_coefs, calculate_weights(derivative_order, (i-1)*grid_step, collect(zero(T) : grid_step : (boundary_length-1)*grid_step)))
    end
end


function right_BC!(::Type{Val{:LO}},high_boundary_coefs,stencil_length,derivative_order,
                        grid_step::T,boundary_length) where T
    boundary_point_count = div(stencil_length,2)
    high_temp            = zeros(T,boundary_length)
    flag                 = derivative_order*boundary_point_count%2
    aorder               = boundary_length - 1

    for i in 1 : boundary_point_count
        # One-sided stencils require more points for same approximation order
        push!(high_boundary_coefs, calculate_weights(derivative_order, -(i-1)*grid_step, reverse(collect(zero(T) : -grid_step : -(boundary_length-1)*grid_step))))
    end
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
