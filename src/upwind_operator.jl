immutable UpwindOperator{T<:Real,S<:SVector,LBC,RBC} <: AbstractLinearOperator{T}
    derivative_order    :: Int
    approximation_order :: Int
    dx                  :: T
    dimension           :: Int
    stencil_length      :: Int
    stencil_coefs       :: S
    boundary_point_count:: Int
    boundary_length     :: Int
    low_boundary_coefs  :: Vector{Vector{T}}
    high_boundary_coefs :: Vector{Vector{T}}
    boundary_fn         :: Tuple{Tuple{T,T,T},Tuple{T,T,T}}

    Base.@pure function UpwindOperator{T,S,LBC,RBC}(derivative_order::Int, approximation_order::Int, dx::T,
                                            dimension::Int, bndry_fn) where {T<:Real,S<:SVector,LBC,RBC}
        dimension            = dimension
        dx                   = dx
        stencil_length       = derivative_order + approximation_order - 1 + (derivative_order+approximation_order)%2
        boundary_length      = derivative_order + approximation_order
        boundary_point_count = stencil_length - div(stencil_length,2) + 1
        grid_step            = one(T)
        low_boundary_coefs   = Vector{T}[]
        high_boundary_coefs  = Vector{T}[]
        stencil_coefs        = convert(SVector{stencil_length, T}, calculate_weights(derivative_order, zero(T),
                               grid_step .* collect(-div(stencil_length,2) : 1 : div(stencil_length,2))))

        left_bndry = initialize_left_boundary!(low_boundary_coefs,stencil_coefs,bndry_fn,derivative_order,grid_step,boundary_length,dx,LBC)
        right_bndry = initialize_right_boundary!(high_boundary_coefs,stencil_coefs,bndry_fn,derivative_order,grid_step,boundary_length,dx,RBC)
        boundary_fn = (left_bndry, right_bndry)

        new(derivative_order, approximation_order, dx, dimension, stencil_length,
            stencil_coefs,
            boundary_point_count,
            boundary_length,
            low_boundary_coefs,
            high_boundary_coefs,
            boundary_fn
            )
    end
    (::Type{UpwindOperator{T}}){T<:Real}(dorder::Int,aorder::Int,dx::T,dim::Int,LBC::Symbol,RBC::Symbol;bndry_fn=(zero(T),zero(T),zero(T))) =
        UpwindOperator{T, SVector{dorder+aorder-1+(dorder+aorder)%2,T}, LBC, RBC}(dorder, aorder, dx, dim, bndry_fn)
end
