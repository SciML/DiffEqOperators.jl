struct DerivativeOperator{T<:Real,S1<:SVector,S2<:SVector} <: AbstractDerivativeOperator{T}
    derivative_order    :: Int
    approximation_order :: Int
    dx                  :: T
    dimension           :: Int
    stencil_length      :: Int
    stencil_coefs       :: S1
    boundary_length     :: Int
    low_boundary_coefs  :: Vector{S2}
    high_boundary_coefs :: Vector{S2}

    function DerivativeOperator{T,S1,S2}(derivative_order::Int,
                                     approximation_order::Int, dx::T,
                                     dimension::Int) where
                                     {T<:Real,S1<:SVector,S2<:SVector}
        dimension            = dimension
        dx                   = dx
        stencil_length       = derivative_order + approximation_order - 1 + (derivative_order+approximation_order)%2
        boundary_length      = derivative_order + approximation_order
        dummy_x              = -div(stencil_length,2) : div(stencil_length,2)
        boundary_x           = -boundary_length+1:0

        # Because it's a N x (N+2) operator, the last stencil on the sides are the [b,0,x,x,x,x] stencils, not the [0,x,x,x,x,x] stencils, since we're never solving for the derivative at the boundary point.
        deriv_spots          = (-div(stencil_length,2)+1) : -1
        boundary_deriv_spots = boundary_x[1:div(stencil_length,2)-1]

        stencil_coefs        = convert(SVector{stencil_length, T}, calculate_weights(derivative_order, zero(T), dummy_x))
        low_boundary_coefs   = [convert(SVector{boundary_length, T}, calculate_weights(derivative_order, oneunit(T)*x0, boundary_x)) for x0 in boundary_deriv_spots]
        high_boundary_coefs  = reverse!(copy(low_boundary_coefs))

        new(derivative_order, approximation_order, dx, dimension, stencil_length,
            stencil_coefs,
            boundary_length,
            low_boundary_coefs,
            high_boundary_coefs
            )
    end
    DerivativeOperator{T}(dorder::Int,aorder::Int,dx::T,dim::Int) where {T<:Real} =
        DerivativeOperator{T, SVector{dorder+aorder-1+(dorder+aorder)%2,T}, SVector{dorder+aorder,T}}(dorder, aorder, dx, dim)
end

#=
    This function is used to update the boundary conditions especially if they evolve with
    time.
=#
function DiffEqBase.update_coefficients!(A::DerivativeOperator{T,S}) where {T<:Real,S<:SVector}
    nothing
end

#################################################################################################

(L::DerivativeOperator)(u,p,t) = L*u
(L::DerivativeOperator)(du,u,p,t) = mul!(du,L,u)

#=
    The Inf opnorm can be calculated easily using the stencil coeffiicents, while other opnorms
    default to compute from the full matrix form.
=#
function LinearAlgebra.opnorm(A::DerivativeOperator{T,S}, p::Real=2) where {T,S}
    if p == Inf && LBC in [:Dirichlet0, :Neumann0, :periodic] && RBC in [:Dirichlet0, :Neumann0, :periodic]
        sum(abs.(A.stencil_coefs)) / A.dx^A.derivative_order
    else
        opnorm(convert(Array,A), p)
    end
end
