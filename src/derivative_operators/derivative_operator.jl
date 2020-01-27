# 
# The types and constructors for derivative operators.
# 
# The type of the operator objects is DerivativeOperator.  This is a
# dumb data structure.  It is initialized by the constructors for
# CenteredDifference and UpwindDifference, which call the calculate_weights
# routine in fornberg.jl to generate the stencils.  The derivatives
# are computed by the mul! methods defined in
# derivative_operator_functions.jl.
# 

index(i::Int, N::Int) = i + div(N, 2) + 1

struct DerivativeOperator{T<:Real,N,Wind,T2,S1,S2<:SArray,T3,F} <: AbstractDerivativeOperator{T}
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
    deriv_spots             = (-div(stencil_length,2)+1) : -1  # unused
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

    stencil_length          = derivative_order + approximation_order - 1 + (derivative_order+approximation_order)%2
    boundary_stencil_length = derivative_order + approximation_order
    stencil_x               = zeros(T, stencil_length)
    boundary_point_count    = div(stencil_length,2) - 1# -1 due to the ghost point

    interior_x              = boundary_point_count+2:len+1-boundary_point_count
    dummy_x                 = -div(stencil_length,2) : div(stencil_length,2)-1
    boundary_x              = -boundary_stencil_length+1:0

    # Because it's a N x (N+2) operator, the last stencil on the sides are the [b,0,x,x,x,x] stencils, not the [0,x,x,x,x,x] stencils, since we're never solving for the derivative at the boundary point.
    deriv_spots             = (-div(stencil_length,2)+1) : -1
    boundary_deriv_spots    = boundary_x[2:div(stencil_length,2)]

    function generate_coordinates(i, stencil_x, dummy_x, dx)
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

    stencil_coefs           = convert(SVector{length(interior_x)}, [convert(SVector{stencil_length, T}, calculate_weights(derivative_order, zero(T), generate_coordinates(i, stencil_x, dummy_x, dx))) for i in interior_x])
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
        high_boundary_coefs,coefficients,coeff_func
        )
end

struct UpwindDifference{N} end

"""
```
UpwindDifference{N}(derivative_order, approximation_order, dx, len, coeff_func = nothing)
```
constructs a DerivativeOperator that automatically implements upwinding.

### Inputs
* `dx::T` or `dx::Vector{T}`: grid spacing
* `coeff_func`: function mapping index in the grid to coefficient at that grid location

### Examples
julia> drift = [1., 1., -1.]
julia> L1 = UpwindDifference(1, 1, 1., 3, i -> drift[i])
julia> L2 = UpwindDifference(1, 1, 1., 3, i -> 1.)
julia> Q = Neumann0BC(1, 1.)
julia> Array(L1 * Q)[1]
3×3 Array{Float64,2}:
 -1.0   1.0   0.0
  0.0  -1.0   1.0
  0.0   1.0  -1.0
julia> Array(L2 * Q)[1]
3×3 Array{Float64,2}:
 -1.0   1.0  0.0
  0.0  -1.0  1.0
  0.0   0.0  0.0

"""
function UpwindDifference{N}(derivative_order::Int,
                             approximation_order::Int, dx::T,
                             len::Int, coeff_func=nothing) where {T<:Real,N}

    stencil_length          = derivative_order + approximation_order
    boundary_stencil_length = derivative_order + approximation_order
    boundary_point_count    = boundary_stencil_length - 2

    # TODO: Clean up the implementation here so that it is more readable and easier to extend in the future
    dummy_x = 0.0 : stencil_length - 1.0
    stencil_coefs = convert(SVector{stencil_length, T}, (1/dx^derivative_order) * calculate_weights(derivative_order, 0.0, dummy_x))

    low_boundary_x         = 0.0:(boundary_stencil_length-1)
    L_boundary_deriv_spots = 1.0:boundary_stencil_length - 2.0
    _low_boundary_coefs     = SVector{boundary_stencil_length, T}[convert(SVector{boundary_stencil_length, T}, (1/dx^derivative_order) * calculate_weights(derivative_order, oneunit(T)*x0, low_boundary_x)) for x0 in L_boundary_deriv_spots]
    low_boundary_coefs      = convert(SVector{boundary_point_count},_low_boundary_coefs)

    high_boundary_x         = 0.0:-1.0:-(boundary_stencil_length-1.0)
    R_boundary_deriv_spots = -1.0:-1.0:-(boundary_stencil_length-2.0)
    _high_boundary_coefs     = SVector{boundary_stencil_length, T}[convert(SVector{boundary_stencil_length, T}, ((-1/dx)^derivative_order) * calculate_weights(derivative_order, oneunit(T)*x0, high_boundary_x)) for x0 in R_boundary_deriv_spots]
    high_boundary_coefs = convert(SVector{boundary_point_count},_high_boundary_coefs)

    coefficients = Vector{T}(undef,len)
    if coeff_func != nothing
        compute_coeffs!(coeff_func, coefficients)
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

# TODO implement the non-uniform grid
function UpwindDifference{N}(derivative_order::Int,
                          approximation_order::Int, dx::AbstractVector{T},
                          len::Int, coeff_func=nothing) where {T<:Real,N}

    stencil_length          = derivative_order + approximation_order
    boundary_stencil_length = derivative_order + approximation_order
    boundary_point_count    = boundary_stencil_length - 2

    # Compute Stencils
    # Compute grid from dx
    x = [0.0, cumsum(dx)...]

    # compute low_boundary_coefs: low_boundary_coefs[upwind = 1 downwind = 2, index of point]
    _upwind_coefs = SMatrix{1,boundary_point_count}([convert(SVector{boundary_stencil_length, T}, calculate_weights(derivative_order, x[i+1], x[i+1:i+boundary_stencil_length])) for i in 1:boundary_point_count])
    _downwind_coefs = SMatrix{1,boundary_point_count}([convert(SVector{boundary_stencil_length,T}, calculate_weights(derivative_order, x[i+1], x[1:boundary_stencil_length])) for i in 1:boundary_point_count])
    low_boundary_coefs = [_upwind_coefs ; _downwind_coefs]

    # compute stencil_coefs: low_boundary_coefs[upwind = 1 downwind = 2, index of point]
    _upwind_coefs = SMatrix{1,len - 2*boundary_point_count}([convert(SVector{stencil_length, T}, calculate_weights(derivative_order, x[i+1], x[i+1:i+stencil_length])) for i in boundary_point_count+1:len-boundary_point_count])
    _downwind_coefs = SMatrix{1,len - 2*boundary_point_count}([convert(SVector{stencil_length,T}, calculate_weights(derivative_order, x[i+1], x[i-stencil_length+2:i+1])) for i in boundary_point_count+1:len-boundary_point_count])
    stencil_coefs = [_upwind_coefs ; _downwind_coefs]

    # compute high_boundary_coefs: low_boundary_coefs[upwind = 1 downwind = 2, index of point]
    _upwind_coefs = SMatrix{1,boundary_point_count}([convert(SVector{boundary_stencil_length, T}, calculate_weights(derivative_order, x[i+1], x[len-boundary_stencil_length+3:len+2])) for i in len-boundary_point_count+1:len])
    _downwind_coefs = SMatrix{1,boundary_point_count}([convert(SVector{boundary_stencil_length,T}, calculate_weights(derivative_order, x[i+1], x[i-stencil_length+2:i+1])) for i in len-boundary_point_count+1:len])
    high_boundary_coefs = [_upwind_coefs ; _downwind_coefs]

    # Compute coefficients
    coefficients = Vector{T}(undef,len)
    if coeff_func != nothing
        compute_coeffs!(coeff_func, coefficients)
    end

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
