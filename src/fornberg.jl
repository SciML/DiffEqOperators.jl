import LinearMaps: LinearMap, AbstractLinearMap
import BandedMatrices: BandedMatrix
import Base: *, getindex
export sparse_full, BandedMatrix
using StaticArrays

abstract AbstractLinearOperator{T} <: AbstractLinearMap{T}

function *(A::AbstractLinearOperator,x::AbstractVector)
    #=
        We will output a vector which is a supertype of the types of A and x
        to ensure numerical stability
    =#
    y = zeros(promote_type(eltype(A),eltype(x)), size(A,1))
    Base.A_mul_B!(y, A::AbstractLinearOperator, x::AbstractVector)
    return y
end


immutable LinearOperator{T<:Real,S<:SVector} <: AbstractLinearOperator{T}
    derivative_order    :: Int
    approximation_order :: Int
    dimension           :: Int
    stencil_length      :: Int
    stencil_coefs       :: S
    boundary_point_count:: Int
    boundary_length     :: Int
    # low_boundary_coefs  :: Vector{Vector{T}}
    # high_boundary_coefs :: Vector{Vector{T}}

    # function LinearOperator{T1<:Real}(dorder::Int, aorder::Int, dim::Int)
    #         slen = dorder + aorder - 1
    #         new{T1, SVector{slen,T1}}(dorder, aorder, dim)
    # end
    Base.@pure function LinearOperator(derivative_order::Int, approximation_order::Int, dimension::Int)
        dimension            = dimension
        stencil_length       = derivative_order + approximation_order - 1
        boundary_length      = derivative_order + approximation_order
        boundary_point_count = stencil_length - div(stencil_length,2) + 1
        grid_step            = one(T)
        #low_boundary_coefs   = Vector{T}[]
        #high_boundary_coefs  = Vector{T}[]
        stencil_coefs        = calculate_weights(derivative_order, zero(T),
                               grid_step .* collect(-div(stencil_length,2) : 1 : div(stencil_length,2)))

        # for i in 1 : boundary_point_count
        #     push!(low_boundary_coefs, calculate_weights(derivative_order, (i-1)*grid_step, collect(zero(T) : grid_step : (boundary_length-1) * grid_step)))
        #     push!(high_boundary_coefs, reverse(low_boundary_coefs[i]))
        #     isodd(derivative_order) ? high_boundary_coefs = -high_boundary_coefs : nothing
        # end
        new(derivative_order, approximation_order, dimension, stencil_length,
            stencil_coefs,
            boundary_point_count,
            boundary_length
            # low_boundary_coefs,
            # high_boundary_coefs
        )
    end
    (::Type{LinearOperator{T}}){T<:Real}(dorder::Int, aorder::Int, dim::Int) =
    LinearOperator{T, SVector{dorder+aorder-1,T}}(dorder, aorder, dim)
end



# ~ bound checking functions ~
checkbounds(A::AbstractLinearOperator, k::Integer, j::Integer) =
    (0 < k ≤ size(A, 1) && 0 < j ≤ size(A, 2) || throw(BoundsError(A, (k,j))))

checkbounds(A::AbstractLinearOperator, kr::Range, j::Integer) =
    (checkbounds(A, first(kr), j); checkbounds(A,  last(kr), j))

checkbounds(A::AbstractLinearOperator, k::Integer, jr::Range) =
    (checkbounds(A, k, first(jr)); checkbounds(A, k,  last(jr)))

checkbounds(A::AbstractLinearOperator, kr::Range, jr::Range) =
    (checkbounds(A, kr, first(jr)); checkbounds(A, kr,  last(jr)))

checkbounds(A::AbstractLinearOperator, k::Colon, j::Integer) =
    (0 < j ≤ size(A, 2) || throw(BoundsError(A, (size(A,1),j))))

checkbounds(A::AbstractLinearOperator, k::Integer, j::Colon) =
    (0 < k ≤ size(A, 1) || throw(BoundsError(A, (k,size(A,2)))))


BandedMatrix(A::LinearOperator) = BandedMatrix(full(A, A.stencil_length), A.stencil_length, div(A.stencil_length,2), div(A.stencil_length,2))

# ~~ getindex ~~
@inline function getindex(A::LinearOperator, i::Int, j::Int)
    @boundscheck checkbounds(A, i, j)
    mid = div(A.stencil_length, 2) + 1
    bpc = A.stencil_length - mid
    l = max(1, low(j, mid, bpc))
    h = min(A.stencil_length, high(j, mid, bpc, A.stencil_length, A.dimension))
    slen = h - l + 1
    if abs(i - j) > div(slen, 2)
        return 0
    else
        return A.stencil_coefs[mid + j - i]
    end
end

# scalar - colon - colon
@inline getindex(A::LinearOperator, kr::Colon, jr::Colon) = full(A)

@inline function getindex(A::LinearOperator, rc::Colon, j)
    T = eltype(A.stencil_coefs)
    v = zeros(T, A.dimension)
    v[j] = one(T)
    copy!(v, A*v)
    return v
end


# symmetric right now
@inline function getindex(A::LinearOperator, i, cc::Colon)
    T = eltype(A.stencil_coefs)
    v = zeros(T, A.dimension)
    v[i] = one(T)
    copy!(v, A*v)
    return v
end


function derivative{T<:Real}(y::Vector{T}, fd::LinearOperator{T})
    dy = zeros(T, length(y))
    derivative!(dy, y, fd)
    return dy
end


function derivative!{T<:Real}(dy::Vector{T}, y::Vector{T}, fd::LinearOperator{T})
    N = length(y)
    #=
        Derivative is calculated in 3 parts:-
            1. For the initial boundary points
            2. For the middle points
            3. For the terminating boundary points
    =#
    @inbounds for i in 1 : fd.boundary_point_count
        bc = fd.low_boundary_coefs[i]
        tmp = zero(T)
        for j in 1 : fd.boundary_length
            tmp += bc[j] * y[j]
        end
        dy[i] = tmp
    end

    d = div(fd.stencil_length, 2)

    @inbounds for i in fd.boundary_point_count+1 : N-fd.boundary_point_count
        c = fd.stencil_coefs
        tmp = zero(T)
        for j in 1 : fd.stencil_length
            tmp += c[j] * y[i-d+j-1]
        end
        dy[i] = tmp
    end

    @inbounds for i in 1 : fd.boundary_point_count
        bc = fd.high_boundary_coefs[i]
        tmp = zero(T)
        for j in 1 : fd.boundary_length
            tmp += bc[j] * y[N - fd.boundary_length + j]
        end
        dy[N - i + 1] = tmp
    end
    return dy
end


function construct_differentiation_matrix{T<:Real}(N::Int, fd::LinearOperator{T})
    #=
        This is for calculating the derivative in one go. But we are creating a function
        which can calculate the derivative by-passing the costly matrix multiplication.
    =#
    D = zeros(T, N, N)
    for i in 1 : fd.boundary_point_count
        D[i, 1 : fd.boundary_length] = fd.low_boundary_coefs[i]
    end
    d = div(fd.stencil_length, 2)
    for i in fd.boundary_point_count + 1 : N - fd.boundary_point_count
        D[i, i-d : i+d] = fd.stencil_coefs
    end
    for i in 1 : fd.boundary_point_count
        D[N - i + 1, N - fd.boundary_length + 1 : N] = fd.high_boundary_coefs[i]
    end
    return D
end


# immutable FiniteDifference <: AbstractLinearOperator
#     # TODO: the general case ie. with an uneven grid
# end


# This implements the Fornberg algorithm to obtain FD weights over arbitrary points to arbitrary order
function calculate_weights{T<:Real}(order::Int, x0::T, x::Vector{T})
    N = length(x)
    @assert order < N "Not enough points for the requested order."
    M = order
    c1 = one(T)
    c4 = x[1] - x0
    C = zeros(T, N, M+1)
    C[1,1] = 1
    @inbounds for i in 1 : N-1
        i1 = i + 1
        mn = min(i, M)
        c2 = one(T)
        c5 = c4
        c4 = x[i1] - x0
        for j in 0 : i-1
            j1 = j + 1
            c3 = x[i1] - x[j1]
            c2 *= c3
            if j == i-1
                for s in mn : -1 : 1
                    s1 = s + 1
                    C[i1,s1] = c1*(s*C[i,s] - c5*C[i,s1]) / c2
                end
                C[i1,1] = -c1*c5*C[i,1] / c2
           end
            for s in mn : -1 : 1
                s1 = s + 1
                C[j1,s1] = (c4*C[j1,s1] - s*C[j1,s]) / c3
            end
            C[j1,1] = c4 * C[j1,1] / c3
        end
        c1 = c2
    end
    #=
        This is to fix the problem of numerical instability which occurs when the sum of the stencil_coefficients is not
        exactly 0.
        https://scicomp.stackexchange.com/questions/11249/numerical-derivative-and-finite-difference-coefficients-any-update-of-the-fornb
        Stack Overflow answer on this issue.
        http://epubs.siam.org/doi/pdf/10.1137/S0036144596322507 - Modified Fornberg Algorithm
    =#
    _C = C[:,end]
    _C[div(N,2)+1] -= sum(_C)
    return convert(SVector{N, T}, _C)
    # return C
end


function convolve!{T<:Real}(x_temp::AbstractVector{T}, x::AbstractVector{T}, coeffs::SVector,
                   i::Int, mid::Int, wndw_low::Int, wndw_high::Int)
    #=
        Here we are taking the weighted sum of a window of the input vector to calculate the derivative
        at the middle point. This requires choosing the end points carefully which are being passed from above.
    =#
    xtempi = x_temp[i]
    @inbounds for idx in wndw_low:wndw_high
        xtempi += coeffs[idx] * x[i - (mid-idx)]
    end
    x_temp[i] = xtempi
end


low(i::Int, mid::Int, bpc::Int) = Int(mid + (i-1)*(1-mid)/bpc)
high(i::Int, mid::Int, bpc::Int, slen::Int, L::Int) = Int(slen - (slen-mid)*(i-L+bpc)/(bpc))


function Base.A_mul_B!{T<:Real}(x_temp::AbstractVector{T}, A::AbstractLinearOperator{T}, x::AbstractVector{T})
    coeffs = A.stencil_coefs
    stencil_length = A.stencil_length
    mid = div(stencil_length, 2) + 1
    bpc = stencil_length - mid
    L = length(x)

    #=
        The high and low functions determine the starting and ending indices of the weight vector.
        As we move along the input vector to calculate the derivative at the point, the weights which
        are to be considered to calculate the derivative are to be chosen carefully. eg. at the boundaries,
        only half of the stencil is going to be used to calculate the derivative at that point.
        So, observing that the left index grows as:-
                  i ^
                    |       mid = ceil(stencil_length/2)
               mid--|       bpc = boundary_point_count
                    |\
                    | \
               0  <_|__\________>
                    |  bpc      i

        And the right index grows as:-
                  i ^       mid = ceil(stencil_length/2)
                    |       bpc = boundary_point_count
               mid--|_______
                    |       \
                    |        \
               0  <_|_________\___>
                    |        bpc  i
        The high and low functions are basically equations of these graphs which are used to calculate
        the left and right index of the stencil as a function of the index i (where we need to find the derivative).
    =#

    Threads.@threads for i in 1 : length(x)
        wndw_low = max(1, low(i, mid, bpc))
        wndw_high = min(stencil_length, high(i, mid, bpc, stencil_length, L))
        convolve!(x_temp, x, coeffs, i, mid, wndw_low, wndw_high)
    end
end


# Base.length(A::LinearOperator) = A.stencil_length
Base.ndims(A::LinearOperator) = 2
Base.size(A::LinearOperator) = (A.dimension, A.dimension)
Base.length(A::LinearOperator) = reduce(*, size(A))

#=
    Currently, for the evenly spaced grid we have a symmetric matrix
=#
Base.transpose(A::LinearOperator) = A
Base.ctranspose(A::LinearOperator) = A
Base.issymmetric(::AbstractLinearOperator) = true

function Base.full{T}(A::LinearOperator{T}, N::Int)
    @assert N >= A.stencil_length # stencil must be able to fit in the matrix
    mat = zeros(T, (N, N))
    v = zeros(T, N)
    for i=1:N
        v[i] = one(T)
        #=
            calculating the effect on a unit vector to get the matrix of transformation
            to get the vector in the new vector space.
        =#
        A_mul_B!(view(mat,:,i), A, v)
        v[i] = zero(T)
    end
    return mat
end

function sparse_full{T}(A::LinearOperator{T}, N::Int=A.dimension)
    @assert N >= A.stencil_length # stencil must be able to fit in the matrix
    mat = spzeros(T, N, N)
    v = zeros(T, N)
    for i=1:N
        v[i] = one(T)
        #=
            calculating the effect on a unit vector to get the matrix of transformation
            to get the vector in the new vector space.
        =#
        A_mul_B!(view(mat,:,i), A, v)
        v[i] = zero(T)
    end
    return mat
end
