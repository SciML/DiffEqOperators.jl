index(i::Int, N::Int) = i + div(N, 2) + 1

struct DerivativeOperator{T<:Real,N,Wind,T2,S1,S2<:SVector,T3,F} <: AbstractDerivativeOperator{T}
    derivative_order        :: Int
    approximation_order     :: Int
    dx                      :: T2
    len                     :: Int
    stencil_length          :: Int
    stencil_coefs           :: S1
    boundary_stencil_length :: Int
    boundary_point_count    :: Int
    low_boundary_coefs      :: S2
    high_boundary_coefs     :: S2
    coefficients            :: T3
    coeff_func              :: F
end

struct CenteredDifference{N} end

function CenteredDifference{N}(derivative_order::Int,
                            approximation_order::Int, dx::T,
                            len::Int, coeff_func=nothing) where {T<:Real,N}

    stencil_length          = derivative_order + approximation_order - 1 + (derivative_order+approximation_order)%2
    boundary_stencil_length = derivative_order + approximation_order
    dummy_x                 = -div(stencil_length,2) : div(stencil_length,2)
    boundary_x              = -boundary_stencil_length+1:0
    boundary_point_count    = div(stencil_length,2) - 1 # -1 due to the ghost point
    # Because it's a N x (N+2) operator, the last stencil on the sides are the [b,0,x,x,x,x] stencils, not the [0,x,x,x,x,x] stencils, since we're never solving for the derivative at the boundary point.
    deriv_spots             = (-div(stencil_length,2)+1) : -1
    boundary_deriv_spots    = boundary_x[2:div(stencil_length,2)]

    stencil_coefs           = convert(SVector{stencil_length, T}, calculate_weights(derivative_order, zero(T), dummy_x))
    _low_boundary_coefs     = SVector{boundary_stencil_length, T}[convert(SVector{boundary_stencil_length, T}, calculate_weights(derivative_order, oneunit(T)*x0, boundary_x)) for x0 in boundary_deriv_spots]
    low_boundary_coefs      = convert(SVector{boundary_point_count},_low_boundary_coefs)
    high_boundary_coefs     = convert(SVector{boundary_point_count},reverse(SVector{boundary_stencil_length, T}[reverse(low_boundary_coefs[i]) for i in 1:boundary_point_count]))

    coefficients            = coeff_func isa Nothing ? nothing : Vector{T}(undef,len)
    DerivativeOperator{T,N,false,T,typeof(stencil_coefs),
        typeof(low_boundary_coefs),typeof(coefficients),
        typeof(coeff_func)}(
        derivative_order, approximation_order, dx, len, stencil_length,
        stencil_coefs,
        boundary_stencil_length,
        boundary_point_count,
        low_boundary_coefs,
        high_boundary_coefs,coefficients,coeff_func
        )
end

function CenteredDifference{N}(derivative_order::Int,
                            approximation_order::Int, dx::AbstractVector{T},
                            len::Int, coeff_func=nothing) where {T<:Real,N}

    stencil_length          = derivative_order + approximation_order - 1 + (derivative_order+approximation_order)%2
    boundary_stencil_length = derivative_order + approximation_order
    stencil_x               = zeros(T, stencil_length)
    boundary_point_count    = div(stencil_length,2) - 1 # -1 due to the ghost point

    interior_x              = boundary_point_count+2:N+1-boundary_point_count
    dummy_x                 = -div(stencil_length,2) : div(stencil_length,2)-1
    boundary_x              = -boundary_stencil_length+1:0

    # Because it's a N x (N+2) operator, the last stencil on the sides are the [b,0,x,x,x,x] stencils, not the [0,x,x,x,x,x] stencils, since we're never solving for the derivative at the boundary point.
    deriv_spots             = (-div(stencil_length,2)+1) : -1
    boundary_deriv_spots    = boundary_x[2:div(stencil_length,2)]

    function generate_coordinates(i, stencil_x, dummy_x, dx)
        j = 1
        N = length(stencil_x)
        stencil_x .= stencil_x.*zero(T)
        for idx in 1:div(N,2)
            shifted_idx1 = index(idx, N)
            shifted_idx2 = index(-idx, N)
            @show shifted_idx1, shifted_idx2, i+idx, i-idx
            stencil_x[shifted_idx1] = stencil_x[shifted_idx1-1] + dx[i+idx-1]
            stencil_x[shifted_idx2] = stencil_x[shifted_idx2+1] - dx[i-idx]
        end
        return stencil_x
    end

    stencil_coefs           = [convert(SVector{stencil_length, T}, calculate_weights(derivative_order, zero(T), generate_coordinates(i, stencil_x, dummy_x, dx))) for i in interior_x]

    _low_boundary_coefs     = SVector{boundary_stencil_length, T}[convert(SVector{boundary_stencil_length, T}, calculate_weights(derivative_order, oneunit(T)*x0, boundary_x)) for x0 in boundary_deriv_spots]
    low_boundary_coefs      = convert(SVector{boundary_point_count},_low_boundary_coefs)
    high_boundary_coefs     = convert(SVector{boundary_point_count},reverse(SVector{boundary_stencil_length, T}[reverse(low_boundary_coefs[i]) for i in 1:boundary_point_count]))

    coefficients            = coeff_func isa Nothing ? nothing : Vector{T}(undef,len)

    DerivativeOperator{T,N,false,typeof(dx),typeof(stencil_coefs),
        typeof(low_boundary_coefs),typeof(coefficients),
        typeof(coeff_func)}(
        derivative_order, approximation_order, dx,
        len, stencil_length,
        stencil_coefs,
        boundary_stencil_length,
        boundary_point_count,
        low_boundary_coefs,
        high_boundary_coefs,coefficients,coeff_func,false
        )
end

struct UpwindDifference{N} end

function UpwindDifference{N}(derivative_order::Int,
                          approximation_order::Int, dx::T,
                          len::Int, coeff_func=nothing) where {T<:Real,N}

    stencil_length          = derivative_order + approximation_order - 1 + (derivative_order+approximation_order)%2
    boundary_stencil_length = derivative_order + approximation_order
    dummy_x                 = -div(stencil_length,2) : div(stencil_length,2)
    boundary_x              = -boundary_stencil_length+1:0
    boundary_point_count    = div(stencil_length,2) - 1 # -1 due to the ghost point
    # Because it's a N x (N+2) operator, the last stencil on the sides are the [b,0,x,x,x,x] stencils, not the [0,x,x,x,x,x] stencils, since we're never solving for the derivative at the boundary point.
    deriv_spots             = (-div(stencil_length,2)+1) : -1
    boundary_deriv_spots    = boundary_x[2:div(stencil_length,2)]

    stencil_coefs           = convert(SVector{stencil_length, T}, calculate_weights(derivative_order, zero(T), dummy_x))
    _low_boundary_coefs     = SVector{boundary_stencil_length, T}[convert(SVector{boundary_stencil_length, T}, calculate_weights(derivative_order, oneunit(T)*x0, boundary_x)) for x0 in boundary_deriv_spots]
    low_boundary_coefs      = convert(SVector{boundary_point_count},_low_boundary_coefs)
    high_boundary_coefs     = convert(SVector{boundary_point_count},reverse(SVector{boundary_stencil_length, T}[reverse(low_boundary_coefs[i]) for i in 1:boundary_point_count]))

    coefficients            = Vector{T}(undef,len)

    DerivativeOperator{T,N,true,T,typeof(stencil_coefs),
        typeof(low_boundary_coefs),Vector{T},
        typeof(coeff_func)}(
        derivative_order, approximation_order, dx, len, stencil_length,
        stencil_coefs,
        boundary_stencil_length,
        boundary_point_count,
        low_boundary_coefs,
        high_boundary_coefs,coefficients,coeff_func
        )
end

function UpwindDifference{N}(derivative_order::Int,
                          approximation_order::Int, dx::AbstractVector{T},
                          len::Int, coeff_func=nothing) where {T<:Real,N}

    stencil_length          = derivative_order + approximation_order - 1 + (derivative_order+approximation_order)%2
    boundary_stencil_length = derivative_order + approximation_order
    dummy_x                 = -div(stencil_length,2) : div(stencil_length,2)
    boundary_x              = -boundary_stencil_length+1:0
    boundary_point_count    = div(stencil_length,2) - 1 # -1 due to the ghost point
    # Because it's a N x (N+2) operator, the last stencil on the sides are the [b,0,x,x,x,x] stencils, not the [0,x,x,x,x,x] stencils, since we're never solving for the derivative at the boundary point.
    deriv_spots             = (-div(stencil_length,2)+1) : -1
    boundary_deriv_spots    = boundary_x[2:div(stencil_length,2)]

    stencil_coefs           = convert(SVector{stencil_length, T}, calculate_weights(derivative_order, zero(T), dummy_x))
    _low_boundary_coefs     = SVector{boundary_stencil_length, T}[convert(SVector{boundary_stencil_length, T}, calculate_weights(derivative_order, oneunit(T)*x0, boundary_x)) for x0 in boundary_deriv_spots]
    low_boundary_coefs      = convert(SVector{boundary_point_count},_low_boundary_coefs)
    high_boundary_coefs     = convert(SVector{boundary_point_count},reverse(SVector{boundary_stencil_length, T}[reverse(low_boundary_coefs[i]) for i in 1:boundary_point_count]))

    coefficients            = Vector{T}(undef,len)

    DerivativeOperator{T,N,true,typeof(dx),typeof(stencil_coefs),
        typeof(low_boundary_coefs),Vector{T},
        typeof(coeff_func)}(
        derivative_order, approximation_order, dx, len, stencil_length,
        stencil_coefs,
        boundary_stencil_length,
        boundary_point_count,
        low_boundary_coefs,
        high_boundary_coefs,coefficients,coeff_func
        )
end

CenteredDifference(args...) = CenteredDifference{1}(args...)
UpwindDifference(args...) = UpwindDifference{1}(args...)
use_winding(A::DerivativeOperator{T,N,Wind}) where {T,N,Wind} = Wind
