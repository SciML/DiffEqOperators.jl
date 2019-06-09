struct DerivativeOperator{T<:Real,S<:SVector} <: AbstractDerivativeOperator{T}
    derivative_order    :: Int
    approximation_order :: Int
    dx                  :: T
    dimension           :: Int
    stencil_length      :: Int
    stencil_coefs       :: S
    boundary_length     :: Int
    low_boundary_coefs  :: Vector{S}
    high_boundary_coefs :: Vector{S}

    function DerivativeOperator{T,S}(derivative_order::Int,
                                     approximation_order::Int, dx::T,
                                     dimension::Int) where
                                     {T<:Real,S<:SVector}
        dimension            = dimension
        dx                   = dx
        stencil_length       = derivative_order + approximation_order - 1 + (derivative_order+approximation_order)%2
        dummy_x              = -div(stencil_length,2) : div(stencil_length,2)
        deriv_spots          = -div(stencil_length,2) : -1
        boundary_length      = length(deriv_spots)

        stencil_coefs        = convert(SVector{stencil_length, T}, calculate_weights(derivative_order, zero(T), dummy_x))
        low_boundary_coefs   = [convert(SVector{stencil_length, T}, calculate_weights(derivative_order, oneunit(T)*x0, dummy_x)) for x0 in deriv_spots]
        high_boundary_coefs  = reverse!(copy(low_boundary_coefs))

        new(derivative_order, approximation_order, dx, dimension, stencil_length,
            stencil_coefs,
            boundary_length,
            low_boundary_coefs,
            high_boundary_coefs
            )
    end
    DerivativeOperator{T}(dorder::Int,aorder::Int,dx::T,dim::Int) where {T<:Real} =
        DerivativeOperator{T, SVector{dorder+aorder-1+(dorder+aorder)%2,T}}(dorder, aorder, dx, dim)
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
