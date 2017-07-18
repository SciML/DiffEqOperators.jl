immutable UpwindOperator{T<:Real,S<:SVector,LBC,RBC} <: AbstractLinearOperator{T}
    derivative_order    :: Int
    approximation_order :: Int
    dx                  :: T
    dimension           :: Int
    directions          :: BitArray{1}
    stencil_length      :: Int
    up_stencil_coefs    :: S
    down_stencil_coefs  :: S
    boundary_point_count:: Int
    boundary_length     :: Int
    low_boundary_coefs  :: Ref{Vector{Vector{T}}}
    high_boundary_coefs :: Ref{Vector{Vector{T}}}
    boundary_condition  :: Ref{Tuple{Tuple{T,T,Any},Tuple{T,T,Any}}}

    Base.@pure function UpwindOperator{T,S,LBC,RBC}(derivative_order::Int, approximation_order::Int, dx::T,
                                            dimension::Int, directions::BitArray{1}, bndry_fn) where {T<:Real,S<:SVector,LBC,RBC}
        dimension            = dimension
        dx                   = dx
        directions           = directions
        stencil_length       = derivative_order + approximation_order
        boundary_length      = derivative_order + approximation_order
        boundary_point_count = stencil_length - div(stencil_length,2) + 1
        grid_step            = one(T)
        low_boundary_coefs   = Vector{T}[]
        high_boundary_coefs  = Vector{T}[]

        up_stencil_coefs        = convert(SVector{stencil_length, T}, calculate_weights(derivative_order, zero(T),
                                          grid_step .* collect(zero(T) : grid_step : stencil_length-1)))

        down_stencil_coefs      = reverse(up_stencil_coefs)
        derivative_order%2 == 1 ? negate!(down_stencil_coefs) : nothing

        left_bndry = initialize_left_boundary!(low_boundary_coefs,up_stencil_coefs,bndry_fn,derivative_order,grid_step,boundary_length,dx,LBC)
        right_bndry = initialize_right_boundary!(high_boundary_coefs,down_stencil_coefs,bndry_fn,derivative_order,grid_step,boundary_length,dx,RBC)
        boundary_fn = (left_bndry, right_bndry)

        new(derivative_order, approximation_order, dx, dimension, directions,
            stencil_length,
            up_stencil_coefs,
            down_stencil_coefs,
            boundary_point_count,
            boundary_length,
            low_boundary_coefs,
            high_boundary_coefs,
            boundary_fn
            )
    end
    (::Type{UpwindOperator{T}}){T<:Real}(dorder::Int,aorder::Int,dx::T,dim::Int,direction::BitArray{1},LBC::Symbol,RBC::Symbol;bndry_fn=(zero(T),zero(T),zero(T))) =
        UpwindOperator{T, SVector{dorder+aorder,T}, LBC, RBC}(dorder, aorder, dx, dim, direction, bndry_fn)
end


(L::UpwindOperator)(t,u) = L*u
(L::UpwindOperator)(t,u,du) = A_mul_B!(du,L,u)
