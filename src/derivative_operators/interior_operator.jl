struct InteriorOperator{T<:Real,S<:SVector} <: AbstractDerivativeOperator{T}
    derivative_order    :: Int
    approximation_order :: Int
    dx                  :: T
    dimension           :: Int
    stencil_length      :: Int
    ghost_point_count   :: Int
    stencil_coefs       :: S

    function InteriorOperator{T,S}(derivative_order::Int,
                                    approximation_order::Int, dx::T,
                                    dimension::Int) where
                                    {T<:Real,S<:SVector}
        dimension            = dimension
        dx                   = dx
        stencil_length       = derivative_order + approximation_order - 1 + (derivative_order+approximation_order)%2
        ghost_point_count    = div(stencil_length,2)
        grid_step            = one(T)
        stencil_coefs        = convert(SVector{stencil_length, T}, calculate_weights(derivative_order, zero(T),
                               grid_step .* collect(-div(stencil_length,2) : 1 : div(stencil_length,2))))


        new(derivative_order, approximation_order, dx, dimension, stencil_length, ghost_point_count, stencil_coefs)
    end
    InteriorOperator{T}(dorder::Int,aorder::Int,dx::T,dim::Int) where {T<:Real} =
        InteriorOperator{T, SVector{dorder+aorder-1+(dorder+aorder)%2,T}}(dorder, aorder, dx, dim)
end

#################################################################################################


(L::InteriorOperator)(u,p,t) = L*u
(L::InteriorOperator)(du,u,p,t) = mul!(du,L,u)
get_type(::InteriorOperator{A,B}) where {A,B} = A

#=
    The Inf opnorm can be calculated easily using the stencil coeffiicents, while other opnorms
    default to compute from the full matrix form.
=#
function LinearAlgebra.opnorm(A::InteriorOperator{T,S}, p::Real=2) where {T,S}
    if p == Inf
        sum(abs.(A.stencil_coefs)) / A.dx^A.derivative_order
    else
        opnorm(convert(Array,A), p)
    end
end
