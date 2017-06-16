function *(A::AbstractLinearOperator,x::AbstractVector)
    #=
        We will output a vector which is a supertype of the types of A and x
        to ensure numerical stability
    =#
    y = zeros(promote_type(eltype(A),eltype(x)), size(A,1))
    Base.A_mul_B!(y, A::AbstractLinearOperator, x::AbstractVector, BC = A.boundary_condition)
    return y
end

function *(func::Function, op)
    func(op)
end

immutable LinearOperator{T<:Real,S<:SVector} <: AbstractLinearOperator{T}
    derivative_order    :: Int
    approximation_order :: Int
    dimension           :: Int
    stencil_length      :: Int
    stencil_coefs       :: S
    boundary_point_count:: Int
    boundary_length     :: Int
    boundary_condition  :: Symbol
    low_boundary_coefs  :: Vector{Vector{T}}
    high_boundary_coefs :: Vector{Vector{T}}
    boundary_fn

    Base.@pure function LinearOperator{T,S}(derivative_order::Int, approximation_order::Int,
                                            dimension::Int, bndry_fn, bdc::Symbol=:D0) where {T<:Real,S<:SVector}
        # bdc == :D0 && !isa((bndry_fn[0]), Real) && error("Dirichlet accepts only constant valued boundaries")

        dimension            = dimension
        stencil_length       = derivative_order + approximation_order - 1
        boundary_length      = derivative_order + approximation_order
        boundary_point_count = stencil_length - div(stencil_length,2) + 1
        grid_step            = one(T)
        boundary_condition   = bdc
        low_boundary_coefs   = Vector{T}[]
        high_boundary_coefs  = Vector{T}[]
        stencil_coefs        = calculate_weights(derivative_order, zero(T),
                               grid_step .* collect(-div(stencil_length,2) : 1 : div(stencil_length,2)))
        boundary_fn          = bndry_fn

        for i in 1 : boundary_point_count
            push!(low_boundary_coefs, calculate_weights(derivative_order, (i-1)*grid_step, collect(zero(T) : grid_step : (boundary_length-1) * grid_step)))
            push!(high_boundary_coefs, reverse(low_boundary_coefs[i]))
            isodd(derivative_order) ? high_boundary_coefs = -high_boundary_coefs : nothing
        end
        new(derivative_order, approximation_order, dimension, stencil_length,
            stencil_coefs,
            boundary_point_count,
            boundary_length,
            boundary_condition,
            low_boundary_coefs,
            high_boundary_coefs,
            boundary_fn
        )
    end
    (::Type{LinearOperator{T}}){T<:Real}(dorder::Int, aorder::Int, dim::Int, bndry_fn, bdc::Symbol=:D0) =
    LinearOperator{T, SVector{dorder+aorder-1,T}}(dorder, aorder, dim, bndry_fn, bdc)
end

(L::LinearOperator)(t,u) = L*u
(L::LinearOperator)(t,u,du) = A_mul_B!(du,L,u)


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

# UnitRanges
@inline function getindex(A::LinearOperator, rng::UnitRange{Int}, cc::Colon)
    m = full(A)
    return m[rng, cc]
end


@inline function getindex(A::LinearOperator, rc::Colon, rng::UnitRange{Int})
    m = full(A)
    return m[rnd, cc]
end


@inline function getindex(A::LinearOperator, r::Int, rng::UnitRange{Int})
    m = A[r, :]
    return m[rng]
end


@inline function getindex(A::LinearOperator, rng::UnitRange{Int}, c::Int)
    m = A[:, c]
    return m[rng]
end


@inline function getindex{T}(A::LinearOperator{T}, rng::UnitRange{Int}, cng::UnitRange{Int})
    N = A.dimension
    if (rng[end] - rng[1]) > ((cng[end] - cng[1]))
        mat = zeros(T, (N, length(cng)))
        v = zeros(T, N)
        for i = cng
            v[i] = one(T)
            #=
                calculating the effect on a unit vector to get the matrix of transformation
                to get the vector in the new vector space.
            =#
            A_mul_B!(view(mat, :, i - cng[1] + 1), A, v)
            v[i] = zero(T)
        end
        return mat[rng, :]

    else
        mat = zeros(T, (length(rng), N))
        v = zeros(T, N)
        for i = rng
            v[i] = one(T)
            #=
                calculating the effect on a unit vector to get the matrix of transformation
                to get the vector in the new vector space.
            =#
            A_mul_B!(view(mat, i - rng[1] + 1, :), A, v)
            v[i] = zero(T)
        end
        return mat[:, cng]
    end
end

#############################################################
# Fornberg algorithm
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


function get_convolution_operator(A::AbstractLinearOperator, boundary_condition::Symbol)
    if boundary_condition == :D0
        return dirichlet_0!
    elseif boundary_condition == :periodic
        return periodic!
    # default
    else
        return dirichlet_0!
    end
end


function Base.A_mul_B!{T<:Real}(x_temp::AbstractVector{T}, A::AbstractLinearOperator{T}, x::AbstractVector{T}; BC=:D0)
    coeffs = A.stencil_coefs
    L = length(x)

    convolution_kernal! = get_convolution_operator(A, BC)
    Threads.@threads for i in 1 : length(x)
        convolution_kernal!(x_temp, x, coeffs, i)
    end
    # preparing the boundaries for dirichlet condition
    if A.boundary_condition == :D1
        x[1] += A.boundary_fn[1]
        x[end] += A.boundary_fn[2]
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

function Base.full{T}(A::LinearOperator{T}, N::Int=A.dimension)
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

function Base.sparse{T}(A::LinearOperator{T})
    N = A.dimension
    mat = spzeros(T, N, N)
    v = zeros(T, N)
    row = zeros(T, N)
    for i=1:N
        v[i] = one(T)
        #=
            calculating the effect on a unit vector to get the matrix of transformation
            to get the vector in the new vector space.
        =#
        A_mul_B!(row, A, v)
        copy!(view(mat,:,i), row)
        row .= 0.*row;
        v[i] = zero(T)
    end
    return mat
end
