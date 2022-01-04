#
# The types and constructors for derivative operators.
#
# The type of the operator objects is DerivativeOperator. This is a
# dumb data structure. It is initialized by the constructors for
# CenteredDifference and UpwindDifference, which call the calculate_weights
# routine in fornberg.jl to generate the stencils. The derivatives
# are computed by the mul! methods defined in
# derivative_operator_functions.jl.
#

index(i::Int, N::Int) = i + div(N, 2) + 1

struct DerivativeOperator{T<:Real,N,Wind,T2,S1,S2,S3,T3,F} <: AbstractDerivativeOperator{T}
    derivative_order        :: Int
    approximation_order     :: Int
    dx                      :: T2
    len                     :: Int
    stencil_length          :: Int
    stencil_coefs           :: S1
    boundary_stencil_length :: Int
    boundary_point_count    :: Int
    low_boundary_coefs      :: S2
    high_boundary_coefs     :: S3
    offside                 :: Int
    coefficients            :: T3
    coeff_func              :: F
end

struct nonlinear_diffusion!{N} end
struct nonlinear_diffusion{N} end

function nonlinear_diffusion!{N}(du::AbstractVector{T}, second_differential_order::Int, first_differential_order::Int, approx_order::Int,
    p::AbstractVector{T}, q::AbstractVector{T}, dx::Union{T , AbstractVector{T} , Real},
    nknots::Int) where {T<:Real, N}
    #q is given by bc*u , u being the unknown function
    #p is given as some function of q , p being the diffusion coefficient

    @assert approx_order>1 "approximation_order must be greater than 1."
    if first_differential_order > 0 
    du .= (CenteredDifference{N}(first_differential_order,approx_order,dx,nknots)*q).*(CenteredDifference{N}(second_differential_order,approx_order,dx,nknots)*p)
    else 
    du .= q[2:(nknots + 1)].*(CenteredDifference{N}(second_differential_order,approx_order,dx,nknots)*p)
    end

    for l = 1:(second_differential_order - 1)
    du .= du .+ binomial(second_differential_order,l)*(CenteredDifference{N}(l + first_differential_order,approx_order,dx,nknots)*q).*(CenteredDifference{N}(second_differential_order - l,approx_order,dx,nknots)*p)
    end

    du .= du .+ (CenteredDifference{N}(first_differential_order + second_differential_order,approx_order,dx,nknots)*q).*p[2:(nknots + 1)]

end

# An out of place workaround for the mutating version
function nonlinear_diffusion{N}(second_differential_order::Int, first_differential_order::Int, approx_order::Int,
    p::AbstractVector{T}, q::AbstractVector{T}, dx::Union{T , AbstractVector{T} , Real},
    nknots::Int) where {T<:Real, N}

    du = similar(q,length(q) - 2)
    return nonlinear_diffusion!{N}(du,second_differential_order,first_differential_order,approx_order,p,q,dx,nknots)
end

struct CenteredDifference{N} end

function CenteredDifference{N}(derivative_order::Int,
                            approximation_order::Int, dx::T,
                            len::Int, coeff_func=1) where {T<:Real,N}
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
    high_boundary_coefs      = convert(SVector{boundary_point_count},reverse(map(reverse, _low_boundary_coefs*(-1)^derivative_order)))

    offside = 0

    coefficients            = fill!(Vector{T}(undef,len),0)
    
    compute_coeffs!(coeff_func, coefficients)
    
    

    DerivativeOperator{T,N,false,T,typeof(stencil_coefs),
        typeof(low_boundary_coefs),typeof(high_boundary_coefs),typeof(coefficients),
        typeof(coeff_func)}(
        derivative_order, approximation_order, dx, len, stencil_length,
        stencil_coefs,
        boundary_stencil_length,
        boundary_point_count,
        low_boundary_coefs,
        high_boundary_coefs,offside,coefficients,coeff_func
        )
end

function CompleteCenteredDifference{N}(derivative_order::Int,
    approximation_order::Int, dx::T,
    len::Int, coeff_func=1) where {T<:Real,N}
@assert approximation_order>1 "approximation_order must be greater than 1."
stencil_length          = derivative_order + approximation_order - 1 + (derivative_order+approximation_order)%2
boundary_stencil_length = derivative_order + approximation_order
dummy_x                 = -div(stencil_length,2) : div(stencil_length,2)
left_boundary_x         = -1:(boundary_stencil_length-1)
right_boundary_x        = reverse(-boundary_stencil_length+1:1)

boundary_point_count    = div(stencil_length,2) # -1 due to the ghost point
# Because it's a N x (N+2) operator, the last stencil on the sides are the [b,0,x,x,x,x] stencils, not the [0,x,x,x,x,x] stencils, since we're never solving for the derivative at the boundary point.
#deriv_spots             = (-div(stencil_length,2)+1) : -1  # unused
#L_boundary_deriv_spots  = left_boundary_x[2:div(stencil_length,2)]
#R_boundary_deriv_spots  = right_boundary_x[2:div(stencil_length,2)]

stencil_coefs           = convert(SVector{stencil_length, T}, (1/dx^derivative_order) * calculate_weights(derivative_order, zero(T), dummy_x))
_low_boundary_coefs     = SVector{boundary_stencil_length, T}[convert(SVector{boundary_stencil_length, T}, (1/dx^derivative_order) * calculate_weights(derivative_order, oneunit(T)*x0, left_boundary_x)) for x0 in L_boundary_deriv_spots]
low_boundary_coefs      = convert(SVector{boundary_point_count},_low_boundary_coefs)

# _high_boundary_coefs    = SVector{boundary_stencil_length, T}[convert(SVector{boundary_stencil_length, T}, (1/dx^derivative_order) * calculate_weights(derivative_order, oneunit(T)*x0, reverse(right_boundary_x))) for x0 in R_boundary_deriv_spots]
high_boundary_coefs      = convert(SVector{boundary_point_count},reverse(map(reverse, _low_boundary_coefs*(-1)^derivative_order)))

offside = 0

coefficients            = fill!(Vector{T}(undef,len),0)

compute_coeffs!(coeff_func, coefficients)



DerivativeOperator{T,N,false,T,typeof(stencil_coefs),
typeof(low_boundary_coefs),typeof(high_boundary_coefs),typeof(coefficients),
typeof(coeff_func)}(
derivative_order, approximation_order, dx, len, stencil_length,
stencil_coefs,
boundary_stencil_length,
boundary_point_count,
low_boundary_coefs,
high_boundary_coefs,offside,coefficients,coeff_func
)
end

function generate_coordinates(i::Int, stencil_x, dummy_x, dx::AbstractVector{T}) where T <: Real
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

function CenteredDifference{N}(derivative_order::Int,
                            approximation_order::Int, dx::AbstractVector{T},
                            len::Int, coeff_func=1) where {T<:Real,N}

    stencil_length          = derivative_order + approximation_order - 1 + (derivative_order+approximation_order)%2
    boundary_stencil_length = derivative_order + approximation_order
    stencil_x               = zeros(T, stencil_length)
    boundary_point_count    = div(stencil_length,2) - 1# -1 due to the ghost point

    interior_x              = boundary_point_count+2:len+1-boundary_point_count
    dummy_x                 = -div(stencil_length,2) : div(stencil_length,2)-1
    low_boundary_x          = [zero(T); cumsum(dx[1:boundary_stencil_length-1])]
    high_boundary_x         = cumsum(dx[end-boundary_stencil_length+1:end])
    # Because it's a N x (N+2) operator, the last stencil on the sides are the [b,0,x,x,x,x] stencils, not the [0,x,x,x,x,x] stencils, since we're never solving for the derivative at the boundary point.
    deriv_spots             = (-div(stencil_length,2)+1) : -1

    stencil_coefs           = [convert(SVector{stencil_length, T}, calculate_weights(derivative_order, zero(T), generate_coordinates(i, stencil_x, dummy_x, dx))) for i in interior_x]
    _low_boundary_coefs     = SVector{boundary_stencil_length, T}[convert(SVector{boundary_stencil_length, T},
                                                                  calculate_weights(derivative_order, low_boundary_x[i+1], low_boundary_x)) for i in 1:boundary_point_count]
    low_boundary_coefs      = convert(SVector{boundary_point_count},_low_boundary_coefs)
    _high_boundary_coefs     = SVector{boundary_stencil_length, T}[convert(SVector{boundary_stencil_length, T},
                                                                  calculate_weights(derivative_order, high_boundary_x[end-i], high_boundary_x)) for i in boundary_point_count:-1:1]
    high_boundary_coefs      = convert(SVector{boundary_point_count},_high_boundary_coefs)

    offside = 0

    coefficients            = zeros(T,len)

    compute_coeffs!(coeff_func, coefficients)
               
        
    DerivativeOperator{T,N,false,typeof(dx),typeof(stencil_coefs),
        typeof(low_boundary_coefs),typeof(high_boundary_coefs),typeof(coefficients),
        typeof(coeff_func)}(
        derivative_order, approximation_order, dx,
        len, stencil_length,
        stencil_coefs,
        boundary_stencil_length,
        boundary_point_count,
        low_boundary_coefs,
        high_boundary_coefs,offside,coefficients,coeff_func
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
                             len::Int, coeff_func=1; offside::Int=0) where {T<:Real,N}

    @assert offside > -1 "Number of offside points should be non-negative"
    @assert offside <= div(derivative_order + approximation_order - 1,2) "Number of offside points should not exceed the primary wind points"
    
    stencil_length          = derivative_order + approximation_order
    boundary_stencil_length = derivative_order + approximation_order
    boundary_point_count    = boundary_stencil_length - 2 - offside

    # TODO: Clean up the implementation here so that it is more readable and easier to extend in the future
    dummy_x = (0.0 - offside) : stencil_length - 1.0 - offside
    stencil_coefs = convert(SVector{stencil_length, T}, (1/dx^derivative_order) * calculate_weights(derivative_order, 0.0, dummy_x))

    low_boundary_x         = 0.0:(boundary_stencil_length-1)
    L_boundary_deriv_spots = 1.0:boundary_stencil_length - 2.0 - offside
    _low_boundary_coefs     = SVector{boundary_stencil_length, T}[convert(SVector{boundary_stencil_length, T}, (1/dx^derivative_order) * calculate_weights(derivative_order, oneunit(T)*x0, low_boundary_x)) for x0 in L_boundary_deriv_spots]
    low_boundary_coefs      = convert(SVector{boundary_point_count},_low_boundary_coefs)

    high_boundary_x         = 0.0:-1.0:-(boundary_stencil_length-1.0)
    R_boundary_deriv_spots = -1.0:-1.0:-(boundary_stencil_length-2.0)
    _high_boundary_coefs     = SVector{boundary_stencil_length, T}[convert(SVector{boundary_stencil_length, T}, ((-1/dx)^derivative_order) * calculate_weights(derivative_order, oneunit(T)*x0, high_boundary_x)) for x0 in R_boundary_deriv_spots]
    high_boundary_coefs = convert(SVector{boundary_point_count + offside},_high_boundary_coefs)

    coefficients = zeros(T,len)
    compute_coeffs!(coeff_func, coefficients)

    DerivativeOperator{T,N,true,T,typeof(stencil_coefs),
        typeof(low_boundary_coefs),typeof(high_boundary_coefs),Vector{T},
        typeof(coeff_func)}(
        derivative_order, approximation_order, dx, len, stencil_length,
        stencil_coefs,
        boundary_stencil_length,
        boundary_point_count,
        low_boundary_coefs,
        high_boundary_coefs,offside,coefficients,coeff_func
        )
end

# TODO implement the non-uniform grid
function UpwindDifference{N}(derivative_order::Int,
                          approximation_order::Int, dx::AbstractVector{T},
                          len::Int, coeff_func=1; offside::Int=0) where {T<:Real,N}

    @assert offside > -1 "Number of offside points should be non-negative"
    @assert offside <= div(derivative_order + approximation_order - 1,2) "Number of offside points should not exceed the primary wind points"

    stencil_length          = derivative_order + approximation_order
    boundary_stencil_length = derivative_order + approximation_order
    boundary_point_count    = boundary_stencil_length - 2 - offside

    # Compute Stencils
    # Compute grid from dx
    x = [0.0, cumsum(dx)...]

    # compute low_boundary_coefs: low_boundary_coefs[upwind = 1 downwind = 2, index of point]
    if offside == 0
        _upwind_coefs = SMatrix{1,boundary_point_count}([convert(SVector{boundary_stencil_length,T}, calculate_weights(derivative_order, x[i+1], x[i+1:i+boundary_stencil_length])) for i in 1:boundary_point_count])
    else
        _upwind_coefs = SMatrix{1,boundary_point_count}(append!([convert(SVector{boundary_stencil_length, T}, calculate_weights(derivative_order, x[i+1], x[1:boundary_stencil_length])) for i in 1:offside-1],[convert(SVector{boundary_stencil_length, T}, calculate_weights(derivative_order, x[i+1], x[i+1-offside:i+boundary_stencil_length-offside])) for i in offside:boundary_point_count]))
    end
    _downwind_coefs = SMatrix{1,boundary_point_count}([convert(SVector{boundary_stencil_length,T}, calculate_weights(derivative_order, x[i+1], x[1:boundary_stencil_length])) for i in 1:boundary_point_count])
    low_boundary_coefs = [_upwind_coefs ; _downwind_coefs]

    # compute stencil_coefs: low_boundary_coefs[upwind = 1 downwind = 2, index of point]
    _upwind_coefs = SMatrix{1,len - 2*boundary_point_count}([convert(SVector{stencil_length, T}, calculate_weights(derivative_order, x[i+1], x[i+1-offside:i+stencil_length-offside])) for i in boundary_point_count+1:len-boundary_point_count])
    _downwind_coefs = SMatrix{1,len - 2*boundary_point_count}([convert(SVector{stencil_length,T}, calculate_weights(derivative_order, x[i+1], x[i-stencil_length+2+offside:i+1+offside])) for i in boundary_point_count+1:len-boundary_point_count])
    stencil_coefs = [_upwind_coefs ; _downwind_coefs]

    # compute high_boundary_coefs: low_boundary_coefs[upwind = 1 downwind = 2, index of point]
    _upwind_coefs = SMatrix{1,boundary_point_count + offside}([convert(SVector{boundary_stencil_length, T}, calculate_weights(derivative_order, x[i+1], x[len-boundary_stencil_length+3:len+2])) for i in len-boundary_point_count+1-offside:len])
    if offside == 0 
        _downwind_coefs = SMatrix{1,boundary_point_count}([convert(SVector{boundary_stencil_length,T}, calculate_weights(derivative_order, x[i+1], x[i-stencil_length+2:i+1])) for i in len-boundary_point_count+1:len])
    elseif offside == 1
        _downwind_coefs = SMatrix{1,boundary_point_count + offside}([convert(SVector{boundary_stencil_length,T}, calculate_weights(derivative_order, x[i+1], x[i-stencil_length+2+offside:i+1+offside])) for i in len-boundary_point_count+1-offside:len-offside+1])
    else
        _downwind_coefs = SMatrix{1,boundary_point_count + offside}(append!([convert(SVector{boundary_stencil_length,T}, calculate_weights(derivative_order, x[i+1], x[i-stencil_length+2+offside:i+1+offside])) for i in len-boundary_point_count+1-offside:len-offside+1],[convert(SVector{boundary_stencil_length,T}, calculate_weights(derivative_order, x[i+1], x[len-stencil_length+3:len+2])) for i in len-offside+2:len]))
    end
    high_boundary_coefs = [_upwind_coefs ; _downwind_coefs]

    # Compute coefficients
    coefficients = zeros(T,len)
    compute_coeffs!(coeff_func, coefficients)

    DerivativeOperator{T,N,true,typeof(dx),typeof(stencil_coefs),
        typeof(low_boundary_coefs),typeof(high_boundary_coefs),Vector{T},
        typeof(coeff_func)}(
        derivative_order, approximation_order, dx, len, stencil_length,
        stencil_coefs,
        boundary_stencil_length,
        boundary_point_count,
        low_boundary_coefs,
        high_boundary_coefs,offside,coefficients,coeff_func
        )
end

CenteredDifference(args...) = CenteredDifference{1}(args...)
UpwindDifference(args...;kwargs...) = UpwindDifference{1}(args...;kwargs...)
nonlinear_diffusion(args...) = nonlinear_diffusion{1}(args...)
nonlinear_diffusion!(args...) = nonlinear_diffusion!{1}(args...)
use_winding(A::DerivativeOperator{T,N,Wind}) where {T,N,Wind} = Wind
diff_axis(A::DerivativeOperator{T,N}) where {T,N} = N
function ==(A1::DerivativeOperator, A2::DerivativeOperator)
    return all([eval(:($A1.$name == $A2.$name)) for name in fieldnames(DerivativeOperator)])
end
function Laplacian(aor::Int, dxyz::Union{NTuple{N, T}, NTuple{N,AbstractVector{T}}}, s::NTuple{N,I}, coeff_func=nothing) where {T,N,I<:Int}
    return sum(CenteredDifference{i}(2, aor, dxyz[i], s[i], coeff_func) for i in 1:N)
end
