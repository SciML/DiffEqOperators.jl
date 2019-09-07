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
    @assert approximation_order>1 "approximation_order must be greater than 1."
    stencil_length          = derivative_order + approximation_order - 1 + (derivative_order+approximation_order)%2
    boundary_stencil_length = derivative_order + approximation_order
    dummy_x                 = -div(stencil_length,2) : div(stencil_length,2)
    left_boundary_x         = 0:(boundary_stencil_length-1)
    right_boundary_x        = reverse(-boundary_stencil_length+1:0)

    boundary_point_count    = div(stencil_length,2) - 1 # -1 due to the ghost point
    # Because it's a N x (N+2) operator, the last stencil on the sides are the [b,0,x,x,x,x] stencils, not the [0,x,x,x,x,x] stencils, since we're never solving for the derivative at the boundary point.
    deriv_spots             = (-div(stencil_length,2)+1) : -1
    L_boundary_deriv_spots  = left_boundary_x[2:div(stencil_length,2)]
    R_boundary_deriv_spots  = right_boundary_x[2:div(stencil_length,2)]

    stencil_coefs           = convert(SVector{stencil_length, T}, (1/dx^derivative_order) * calculate_weights(derivative_order, zero(T), dummy_x))
    _low_boundary_coefs     = SVector{boundary_stencil_length, T}[convert(SVector{boundary_stencil_length, T}, (1/dx^derivative_order) * calculate_weights(derivative_order, oneunit(T)*x0, left_boundary_x)) for x0 in L_boundary_deriv_spots]
    low_boundary_coefs      = convert(SVector{boundary_point_count},_low_boundary_coefs)

    # _high_boundary_coefs    = SVector{boundary_stencil_length, T}[convert(SVector{boundary_stencil_length, T}, (1/dx^derivative_order) * calculate_weights(derivative_order, oneunit(T)*x0, reverse(right_boundary_x))) for x0 in R_boundary_deriv_spots]
    high_boundary_coefs      = convert(SVector{boundary_point_count},reverse(map(reverse, _low_boundary_coefs)))

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

    @assert approximation_order>1 "approximation_order must be greater than 1."
    @assert len + 1 == length(dx) "Please provide grid distance values for ghost points also"

    stencil_length          = derivative_order + approximation_order - 1 + (derivative_order+approximation_order)%2
    boundary_stencil_length = derivative_order + approximation_order
    boundary_point_count    = div(stencil_length,2) - 1 # -1 due to the ghost point

    total_dx_indices        = boundary_point_count+2:len-boundary_point_count
    dummy_x                 = -div(stencil_length,2) : div(stencil_length,2)
    left_boundary_x         = 0:(boundary_stencil_length-1)
    right_boundary_x        = reverse(-boundary_stencil_length+1:0)

    left_dx_indices         = 1:2*boundary_point_count
    right_dx_indices        = N-(2*boundary_point_count):N+1
    interior_dx_indices     = 2*boundary_point_count + 1 : N-(2*boundary_point_count) - 1

    # Because it's a N x (N+2) operator, the last stencil on the sides are the [b,0,x,x,x,x] stencils, not the [0,x,x,x,x,x] stencils, since we're never solving for the derivative at the boundary point.
    interior_stencil_x      = zeros(T, stencil_length)
    boundary_stencil_x      = zeros(T, boundary_stencil_length)

    deriv_spots             = (-div(stencil_length,2)+1) : -1
    L_boundary_deriv_spots  = left_boundary_x[2:div(stencil_length,2)]
    R_boundary_deriv_spots  = reverse(L_boundary_deriv_spots)

    function generate_interior_coordinates(i, stencil_x, dx)
        len = length(stencil_x)
        stencil_x .= stencil_x.*zero(T)
        for idx in 1:div(len,2)
            shifted_idx1 = index(idx, len)
            shifted_idx2 = index(-idx, len)
            stencil_x[shifted_idx1] = stencil_x[shifted_idx1-1] + dx[i+idx-1]
            stencil_x[shifted_idx2] = stencil_x[shifted_idx2+1] - dx[i-idx]
        end
        return stencil_x
    end

    function generate_left_boundary_coordinates(i, stencil_x, dx)
        len = length(stencil_x)
        _dx = dx[1:len-1]
        stencil_x[i] = zero(typeof(stencil_x[1]))
        for j in i-1:-1:1
            stencil_x[j] = stencil_x[j+1] - _dx[j]
        end
        for j in i+1:len
            stencil_x[j] = stencil_x[j-1] + _dx[j-1]
        end
        return stencil_x
    end

    function generate_right_boundary_coordinates(i, stencil_x, dx)
        len = length(stencil_x)
        len_dx = length(dx)
        _dx = dx[N-len+1:len_dx]
        i = len_dx-i+len
        @show i
        stencil_x[i] = zero(typeof(stencil_x[1]))
        for j in i-1:-1:1
            stencil_x[j] = stencil_x[j+1] - _dx[j]
        end
        for j in i+1:len
            stencil_x[j] = stencil_x[j-1] + _dx[j-1]
        end
        @show stencil_x
        return stencil_x
    end

    stencil_coefs           = convert(SVector{length(interior_dx_indices)}, [convert(SVector{stencil_length, T}, calculate_weights(derivative_order, zero(T), generate_interior_coordinates(i, interior_stencil_x, dx))) for i in interior_dx_indices])

    _low_boundary_coefs     = SVector{boundary_stencil_length, T}[convert(SVector{boundary_stencil_length, T}, calculate_weights(derivative_order, zero(T), generate_left_boundary_coordinates(i, boundary_stencil_x, dx))) for i in L_boundary_deriv_spots]

    low_boundary_coefs      = convert(SVector{boundary_point_count},_low_boundary_coefs)

    _high_boundary_coefs     = SVector{boundary_stencil_length, T}[convert(SVector{boundary_stencil_length, T}, calculate_weights(derivative_order, zero(T), generate_right_boundary_coordinates(i, boundary_stencil_x, dx))) for i in R_boundary_deriv_spots]

    boundary_x              = -boundary_stencil_length+1:0

    # Because it's a N x (N+2) operator, the last stencil on the sides are the [b,0,x,x,x,x] stencils, not the [0,x,x,x,x,x] stencils, since we're never solving for the derivative at the boundary point.
    deriv_spots             = (-div(stencil_length,2)+1) : -1
    boundary_deriv_spots    = boundary_x[2:div(stencil_length,2)]
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
        high_boundary_coefs,coefficients,coeff_func
        )
end

struct UpwindDifference{N} end

function UpwindDifference{N}(derivative_order::Int,
                          approximation_order::Int, dx::T,
                          len::Int, coeff_func=nothing) where {T<:Real,N}
    stencil_length          = derivative_order + approximation_order
    boundary_stencil_length = derivative_order + approximation_order
    # Fornberg generates ghost order approach incompatible stencils for even approximation orders
    if boundary_stencil_length%2 == 0
        boundary_stencil_length += 1
    end

    dummy_x                 = -1.0 : stencil_length - 2.0
    boundary_x              = -boundary_stencil_length+1:0
    boundary_point_count    = boundary_stencil_length - 1 # -1 due to the ghost point
    # Because it's a N x (N+2) operator, the last stencil on the sides are the [b,0,x,x,x,x] stencils, not the [0,x,x,x,x,x] stencils, since we're never solving for the derivative at the boundary point.
    boundary_deriv_spots    = boundary_x[2:stencil_length]
    stencil_pivot           = (stencil_length+1)%2 - 1.0
    stencil_coefs           = convert(SVector{stencil_length, T}, (1/dx^derivative_order) * calculate_weights(derivative_order, stencil_pivot, dummy_x))

    left_boundary_x         = 0:(boundary_stencil_length-1)
    right_boundary_x        = reverse(-(boundary_stencil_length-1):0)

    # Because it's a N x (N+2) operator, the last stencil on the sides are the [b,0,x,x,x,x] stencils, not the [0,x,x,x,x,x] stencils, since we're never solving for the derivative at the boundary point.
    L_boundary_deriv_spots  = left_boundary_x[2:boundary_stencil_length]
    R_boundary_deriv_spots  = reverse(right_boundary_x[2:boundary_stencil_length])

    _low_boundary_coefs     = SVector{boundary_stencil_length, T}[convert(SVector{boundary_stencil_length, T}, (1/dx^derivative_order) * calculate_weights(derivative_order, oneunit(T)*x0, left_boundary_x)) for x0 in L_boundary_deriv_spots]
    low_boundary_coefs      = convert(SVector{boundary_point_count},_low_boundary_coefs)

    # _high_boundary_coefs    = SVector{boundary_stencil_length, T}[convert(SVector{boundary_stencil_length, T}, (1/dx^derivative_order) * calculate_weights(derivative_order, oneunit(T)*x0, reverse(right_boundary_x))) for x0 in R_boundary_deriv_spots]
    # high_boundary_coefs      = convert(SVector{boundary_point_count},_high_boundary_coefs)
    high_boundary_coefs      = convert(SVector{boundary_point_count},reverse(map(reverse, _low_boundary_coefs)))

    coefficients            = Vector{T}(undef,len)
    for i in 1:len
        coefficients[i] = coeff_func(i)
    end

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
diff_axis(A::DerivativeOperator{T,N}) where {T,N} = N
