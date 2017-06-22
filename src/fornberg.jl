function *(A::AbstractLinearOperator,x::AbstractVector)
    #=
        We will output a vector which is a supertype of the types of A and x
        to ensure numerical stability
    =#
    y = zeros(promote_type(eltype(A),eltype(x)), size(A,1))
    Base.A_mul_B!(y, A::AbstractLinearOperator, x::AbstractVector)
    return y
end

function negate!{T}(arr::T)
    if size(arr,2) == 1
        scale!(arr,-1)
        return nothing
    end
    for row in arr
        scale!(row,-1)
    end
end

immutable LinearOperator{T<:Real,S<:SVector,LBC,RBC} <: AbstractLinearOperator{T}
    derivative_order    :: Int
    approximation_order :: Int
    dimension           :: Int
    stencil_length      :: Int
    stencil_coefs       :: S
    boundary_point_count:: Int
    boundary_length     :: Int
    low_boundary_coefs  :: Vector{Vector{T}}
    high_boundary_coefs :: Vector{Vector{T}}
    boundary_fn         :: Tuple{T,T}

    Base.@pure function LinearOperator{T,S,LBC,RBC}(derivative_order::Int, approximation_order::Int,
                                            dimension::Int, bndry_fn) where {T<:Real,S<:SVector,LBC,RBC}
        # bdc == :D0 && !isa((bndry_fn[0]), Real) && error("Dirichlet accepts only constant valued boundaries")

        dimension            = dimension
        stencil_length       = derivative_order + approximation_order - 1 + (derivative_order+approximation_order)%2
        boundary_length      = derivative_order + approximation_order
        boundary_point_count = stencil_length - div(stencil_length,2) + 1
        grid_step            = one(T)
        low_boundary_coefs   = Vector{T}[]
        high_boundary_coefs  = Vector{T}[]
        stencil_coefs        = convert(SVector{stencil_length, T}, calculate_weights(derivative_order, zero(T),
                               grid_step .* collect(-div(stencil_length,2) : 1 : div(stencil_length,2))))

        l_fact, r_fact = initialize_boundaries!(low_boundary_coefs, high_boundary_coefs,derivative_order,grid_step,boundary_length,boundary_point_count,LBC,RBC)

        boundary_fn = (l_fact*bndry_fn[1], r_fact*bndry_fn[2])

        new(derivative_order, approximation_order, dimension, stencil_length,
            stencil_coefs,
            boundary_point_count,
            boundary_length,
            low_boundary_coefs,
            high_boundary_coefs,
            boundary_fn
        )
    end
    (::Type{LinearOperator{T}}){T<:Real}(dorder::Int, aorder::Int, dim::Int, LBC::Symbol, RBC::Symbol; bndry_fn=(0.0,0.0)) =
    LinearOperator{T, SVector{dorder+aorder-1+(dorder+aorder)%2,T}, LBC, RBC}(dorder, aorder, dim, bndry_fn)
end

function initialize_boundaries!{T}(low_boundary_coefs, high_boundary_coefs,
                                   derivative_order,grid_step::T,boundary_length,
                                   boundary_point_count,LBC,RBC)
    high_temp            = zeros(T,boundary_length)
    flag                 = derivative_order*boundary_point_count%2
    aorder               = boundary_length - 1
    first_order_coeffs   = zeros(T,boundary_length)
    original_coeffs      = zeros(T,boundary_length)
    l_diff = one(T)
    r_diff = one(T)

    if LBC == :Neumann
        first_order_coeffs = calculate_weights(1, (0)*grid_step, collect(zero(T) : grid_step : aorder* grid_step))
        original_coeffs =  calculate_weights(derivative_order, (0)*grid_step, collect(zero(T) : grid_step : (boundary_length-1) * grid_step))

        l_diff = original_coeffs[end]/first_order_coeffs[end]
        scale!(first_order_coeffs, original_coeffs[end]/first_order_coeffs[end])
        # scale!(original_coeffs, first_order_coeffs[end]/original_coeffs[end])
        @. original_coeffs = original_coeffs - first_order_coeffs
        # copy!(first_order_coeffs, first_order_coeffs[1:end-1])
        push!(low_boundary_coefs, original_coeffs[1:end-1])
    end

    if RBC == :Neumann
        copy!(first_order_coeffs, calculate_weights(1, (boundary_point_count-1)*grid_step, collect(zero(T) : grid_step : aorder * grid_step)))
        reverse!(first_order_coeffs)
        isodd(flag) ? negate!(first_order_coeffs) : nothing

        copy!(original_coeffs, calculate_weights(derivative_order, (boundary_point_count-1)*grid_step, collect(zero(T) : grid_step : (boundary_length-1) * grid_step)))
        reverse!(original_coeffs)
        isodd(flag) ? negate!(original_coeffs) : nothing

        r_diff = original_coeffs[1]/first_order_coeffs[1]
        scale!(first_order_coeffs, original_coeffs[1]/first_order_coeffs[1])
        # scale!(original_coeffs, first_order_coeffs[1]/original_coeffs[1])
        @. original_coeffs = original_coeffs - first_order_coeffs
        # copy!(first_order_coeffs, first_order_coeffs[1:end-1])
    end

    for i in 1 : boundary_point_count
        if LBC == :Neumann && i > 1
            push!(low_boundary_coefs, calculate_weights(derivative_order, (i-1)*grid_step, collect(zero(T) : grid_step : (boundary_length-1) * grid_step)))
        end

        if RBC == :Neumann && i < boundary_point_count
            copy!(high_temp, calculate_weights(derivative_order, (i-1)*grid_step, collect(zero(T) : grid_step : (boundary_length-1) * grid_step)))
            reverse!(high_temp)
            isodd(flag) ? negate!(high_temp) : nothing
            push!(high_boundary_coefs,high_temp)
        end
    end

    if RBC == :Neumann
        push!(high_boundary_coefs, original_coeffs[2:end])
    end
    return l_diff, r_diff
end


(L::LinearOperator)(t,u) = L*u
(L::LinearOperator)(t,u,du) = A_mul_B!(du,L,u)
get_LBC{A,B,C,D}(::LinearOperator{A,B,C,D}) = C
get_RBC{A,B,C,D}(::LinearOperator{A,B,C,D}) = D


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


# BandedMatrix{A,B,C,D}(A::LinearOperator{A,B,C,D}) = BandedMatrix(full(A, A.stencil_length), A.stencil_length, div(A.stencil_length,2), div(A.stencil_length,2))

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
function derivative{T<:Real}(y::Vector{T}, A::LinearOperator{T})
    dy = zeros(T, length(y))
    derivative!(dy, y, A)
    return dy
end


function derivative!{T<:Real}(dy::Vector{T}, y::Vector{T}, A::LinearOperator{T})
    N = length(y)
    #=
        Derivative is calculated in 3 parts:-
            1. For the initial boundary points
            2. For the middle points
            3. For the terminating boundary points
    =#
    @inbounds for i in 1 : A.boundary_point_count
        bc = A.low_boundary_coefs[i]
        tmp = zero(T)
        for j in 1 : A.boundary_length
            tmp += bc[j] * y[j]
        end
        dy[i] = tmp
    end

    d = div(A.stencil_length, 2)

    @inbounds for i in A.boundary_point_count+1 : N-A.boundary_point_count
        c = A.stencil_coefs
        tmp = zero(T)
        for j in 1 : A.stencil_length
            tmp += c[j] * y[i-d+j-1]
        end
        dy[i] = tmp
    end

    @inbounds for i in 1 : A.boundary_point_count
        bc = A.high_boundary_coefs[i]
        tmp = zero(T)
        for j in 1 : A.boundary_length
            tmp += bc[j] * y[N - A.boundary_length + j]
        end
        dy[N - i + 1] = tmp
    end
    return dy
end


function construct_differentiation_matrix{T<:Real}(N::Int, A::LinearOperator{T})
    #=
        This is for calculating the derivative in one go. But we are creating a function
        which can calculate the derivative by-passing the costly matrix multiplication.
    =#
    D = zeros(T, N, N)
    for i in 1 : A.boundary_point_count
        D[i, 1 : A.boundary_length] = A.low_boundary_coefs[i]
    end
    d = div(A.stencil_length, 2)
    for i in A.boundary_point_count + 1 : N - A.boundary_point_count
        D[i, i-d : i+d] = A.stencil_coefs
    end
    for i in 1 : A.boundary_point_count
        D[N - i + 1, N - A.boundary_length + 1 : N] = A.high_boundary_coefs[i]
    end
    return D
end


# immutable FiniteDifference <: AbstractLinearOperator
#     # TODO: the general case ie. with an uneven grid
# end


# This implements the Fornberg algorithm to obtain Finite Difference weights over arbitrary points to arbitrary order
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
    return _C
    # return C
end


function Base.A_mul_B!{T<:Real}(x_temp::AbstractVector{T}, A::LinearOperator{T}, x::AbstractVector{T})
    convolve_BC_left!(x_temp, x, A)
    convolve_interior!(x_temp, x, A)
    convolve_BC_right!(x_temp, x, A)
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
