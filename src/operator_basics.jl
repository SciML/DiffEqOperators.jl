function *(A::AbstractLinearOperator,x::AbstractVector)
    #=
        We will output a vector which is a supertype of the types of A and x
        to ensure numerical stability
    =#
    y = zeros(promote_type(eltype(A),eltype(x)), length(x))
    Base.A_mul_B!(y, A::AbstractLinearOperator, x::AbstractVector)
    return y
end


function *(A::AbstractLinearOperator,M::AbstractMatrix)
    #=
        We will output a vector which is a supertype of the types of A and x
        to ensure numerical stability
    =#
    y = zeros(promote_type(eltype(A),eltype(M)), size(M))
    Base.A_mul_B!(y, A::AbstractLinearOperator, M::AbstractMatrix)
    return y
end


function *(M::AbstractMatrix,A::AbstractLinearOperator)
    #=
        We will output a vector which is a supertype of the types of A and x
        to ensure numerical stability
    =#
    y = zeros(promote_type(eltype(A),eltype(M)), size(M))
    Base.A_mul_B!(y, A::AbstractLinearOperator, M::AbstractMatrix)
    return y
end


function *(A::AbstractLinearOperator,B::AbstractLinearOperator)
    # TODO: it will result in an operator which calculates
    #       the derivative of order A.dorder + B.dorder of
    #       approximation_order = min(approx_A, approx_B)
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
    dx                  :: T
    dimension           :: Int
    stencil_length      :: Int
    stencil_coefs       :: S
    boundary_point_count:: Int
    boundary_length     :: Int
    low_boundary_coefs  :: Vector{Vector{T}}
    high_boundary_coefs :: Vector{Vector{T}}
    boundary_fn         :: Tuple{Tuple{T,T,T},Tuple{T,T,T}}

    Base.@pure function LinearOperator{T,S,LBC,RBC}(derivative_order::Int, approximation_order::Int, dx::T,
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
    (::Type{LinearOperator{T}}){T<:Real}(dorder::Int,aorder::Int,dx::T,dim::Int,LBC::Symbol,RBC::Symbol;bndry_fn=(zero(T),zero(T),zero(T))) =
        LinearOperator{T, SVector{dorder+aorder-1+(dorder+aorder)%2,T}, LBC, RBC}(dorder, aorder, dx, dim, bndry_fn)
end


function initialize_left_boundary!{T}(low_boundary_coefs,stencil_coefs,bndry_fn,
                                   derivative_order,grid_step::T,boundary_length,dx,LBC)
    stencil_length = length(stencil_coefs)
    boundary_point_count = stencil_length - div(stencil_length,2) + 1

    if LBC == :None
        return (zero(T),zero(T),left_None_BC!(low_boundary_coefs,stencil_coefs,derivative_order,
                              grid_step,boundary_length)*bndry_fn[1]*dx)
    elseif LBC == :Neumann
        return (zero(T),one(T),left_Neumann_BC!(low_boundary_coefs,stencil_length,derivative_order,
                                 grid_step,boundary_length)*bndry_fn[1]*dx)
    elseif LBC == :Robin
        return (bndry_fn[1][1],bndry_fn[1][2],left_Robin_BC!(low_boundary_coefs,stencil_length,
                                                   bndry_fn[1],derivative_order,grid_step,
                                                   boundary_length,dx)*bndry_fn[1][3]*dx)
    elseif LBC == :Dirichlet0
        return (one(T),zero(T),bndry_fn[1])

    elseif LBC == :Dirichlet
        return (one(T),zero(T),bndry_fn[1])

    elseif LBC == :Neumann0
        return (zero(T),one(T),zero(T))

    else
        error("Unrecognized Boundary Type!")
    end
end


function initialize_right_boundary!{T}(high_boundary_coefs,stencil_coefs,bndry_fn,
                                   derivative_order,grid_step::T,boundary_length,dx,RBC)
    stencil_length = length(stencil_coefs)
    boundary_point_count = stencil_length - div(stencil_length,2) + 1

    if RBC == :None
        return (zero(T),zero(T),right_None_BC!(high_boundary_coefs,stencil_coefs,derivative_order,
                               grid_step,boundary_length)*bndry_fn[2]*dx)
    elseif RBC == :Neumann
        return (zero(T),one(T),right_Neumann_BC!(high_boundary_coefs,stencil_length,derivative_order,
                                  grid_step,boundary_length)*bndry_fn[2]*dx)
    elseif RBC == :Robin
        return (bndry_fn[2][1],bndry_fn[2][2],right_Robin_BC!(high_boundary_coefs,stencil_length,
                                                    bndry_fn[2],derivative_order,grid_step,
                                                    boundary_length,dx)*bndry_fn[2][3]*dx)
    elseif RBC == :Dirichlet0
        return (one(T),zero(T),bndry_fn[2])

    elseif RBC == :Dirichlet
        return (one(T),zero(T),bndry_fn[2])

    elseif RBC == :Neumann0
        return (zero(T),one(T),zero(T))

    else
        error("Unrecognized Boundary Type!")
    end
end


function left_None_BC!{T}(low_boundary_coefs,stencil_coefs,
                       derivative_order,grid_step::T,boundary_length)
    aorder               = boundary_length - 1
    stencil_length       = length(stencil_coefs)
    boundary_point_count = stencil_length - div(stencil_length,2) + 1
    l_diff               = zero(T)
    mid                  = div(stencil_length,2)
    for i in 1 : boundary_point_count
        # One-sided stencils require more points for same approximation order
        # TODO: I don't know if this is the correct stencil length for i > 1?

        if i < 1 + mid
            push!(low_boundary_coefs, calculate_weights(derivative_order, (i-1)*grid_step, collect(zero(T) : grid_step : (boundary_length-1)*grid_step)))
        else
            # FIXME: This "boundary point" should just be considered interior points for LBC = :None
            push!(low_boundary_coefs, stencil_coefs)
        end
    end
    return l_diff
end


function right_None_BC!{T}(high_boundary_coefs,stencil_coefs,
                        derivative_order,grid_step::T,boundary_length)
    stencil_length       = length(stencil_coefs)
    boundary_point_count = stencil_length - div(stencil_length,2) + 1
    high_temp            = zeros(T,boundary_length)
    flag                 = derivative_order*boundary_point_count%2
    aorder               = boundary_length - 1
    r_diff               = zero(T)
    for i in 1 : boundary_point_count
        # One-sided stencils require more points for same approximation order
        if i < 1 + div(stencil_length,2)
            push!(high_boundary_coefs, calculate_weights(derivative_order, -(i-1)*grid_step, reverse(collect(zero(T) : -grid_step : -(boundary_length-1)*grid_step))))
        else
            # FIXME: This "boundary point" should just be considered interior points for RBC = :None
            push!(high_boundary_coefs, stencil_coefs)
        end
    end
    return r_diff
end


function left_Neumann_BC!{T}(low_boundary_coefs,stencil_length,
                             derivative_order,grid_step::T,boundary_length)
    aorder               = boundary_length - 1
    boundary_point_count = stencil_length - div(stencil_length,2) + 1
    first_order_coeffs   = zeros(T,boundary_length)
    original_coeffs      = zeros(T,boundary_length)
    l_diff               = one(T)
    mid                  = div(stencil_length,2)+1

    first_order_coeffs = calculate_weights(1, (0)*grid_step, collect(zero(T) : grid_step : aorder* grid_step))
    original_coeffs =  calculate_weights(derivative_order, (0)*grid_step, collect(zero(T) : grid_step : (boundary_length-1) * grid_step))

    l_diff = original_coeffs[end]/first_order_coeffs[end]
    scale!(first_order_coeffs, original_coeffs[end]/first_order_coeffs[end])
    # scale!(original_coeffs, first_order_coeffs[end]/original_coeffs[end])
    @. original_coeffs = original_coeffs - first_order_coeffs
    # copy!(first_order_coeffs, first_order_coeffs[1:end-1])
    push!(low_boundary_coefs, original_coeffs[1:end-1])

    for i in 2 : boundary_point_count
        #=  this means that a stencil will suffice ie. we dont't need to worry about the boundary point
            being considered in the low_boundary_coefs
        =#
        if i > mid
            pos=i-1
            push!(low_boundary_coefs, calculate_weights(derivative_order, pos*grid_step, collect(zero(T) : grid_step : (boundary_length-1) * grid_step)))
        else
            pos=i-2
            push!(low_boundary_coefs, append!([zero(T)],calculate_weights(derivative_order, pos*grid_step, collect(zero(T) : grid_step : (boundary_length-1) * grid_step))))
        end
    end

    return l_diff
end


function right_Neumann_BC!{T}(high_boundary_coefs,stencil_length,
                           derivative_order,grid_step::T,boundary_length)
    boundary_point_count = stencil_length - div(stencil_length,2) + 1
    flag                 = derivative_order*boundary_point_count%2
    aorder               = boundary_length - 1
    original_coeffs      = zeros(T,boundary_length)
    r_diff               = one(T)
    mid                  = div(stencil_length,2)+1

    # this part is to incorporate the value of first derivative at the right boundary
    first_order_coeffs = calculate_weights(1, (boundary_point_count-1)*grid_step, collect(zero(T) : grid_step : aorder * grid_step))
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

    for i in 2 : boundary_point_count
        #=
            this means that a stencil will suffice ie. we dont't need to worry about the boundary point
            being considered in the high_boundary_coefs. Same code for low_boundary_coefs but reversed
            at the end
        =#
        if i > mid
            pos=i-1
            push!(high_boundary_coefs, calculate_weights(derivative_order, pos*grid_step, collect(zero(T) : grid_step : (boundary_length-1) * grid_step)))
        else
            pos=i-2
            push!(high_boundary_coefs, append!([zero(T)],calculate_weights(derivative_order, pos*grid_step, collect(zero(T) : grid_step : (boundary_length-1) * grid_step))))
        end
    end
    if flag == 1
        negate!(high_boundary_coefs)
    end
    reverse!(high_boundary_coefs)
    push!(high_boundary_coefs, original_coeffs[2:end])
    return r_diff
end


function left_Robin_BC!{T}(low_boundary_coefs,stencil_length,params,
                           derivative_order,grid_step::T,boundary_length,dx)
    aorder               = boundary_length - 1
    boundary_point_count = stencil_length - div(stencil_length,2) + 1
    first_order_coeffs   = zeros(T,boundary_length)
    original_coeffs      = zeros(T,boundary_length)
    l_diff               = one(T)
    mid                  = div(stencil_length,2)+1

    first_order_coeffs = params[2]*calculate_weights(1, (0)*grid_step, collect(zero(T) : grid_step : aorder* grid_step))
    first_order_coeffs[1] += dx*params[1]
    original_coeffs =  calculate_weights(derivative_order, (0)*grid_step, collect(zero(T) : grid_step : (boundary_length-1) * grid_step))

    l_diff = original_coeffs[end]/first_order_coeffs[end]
    scale!(first_order_coeffs, original_coeffs[end]/first_order_coeffs[end])
    # scale!(original_coeffs, first_order_coeffs[end]/original_coeffs[end])
    @. original_coeffs = original_coeffs - first_order_coeffs
    # copy!(first_order_coeffs, first_order_coeffs[1:end-1])
    push!(low_boundary_coefs, original_coeffs[1:end-1])

    for i in 2 : boundary_point_count
        #=  this means that a stencil will suffice ie. we dont't need to worry about the boundary point
            being considered in the low_boundary_coefs
        =#
        if i > mid
            pos=i-1
            push!(low_boundary_coefs, calculate_weights(derivative_order, pos*grid_step, collect(zero(T) : grid_step : (boundary_length-1) * grid_step)))
        else
            pos=i-2
            push!(low_boundary_coefs, append!([zero(T)],calculate_weights(derivative_order, pos*grid_step, collect(zero(T) : grid_step : (boundary_length-1) * grid_step))))
        end
    end

    return l_diff
end


function right_Robin_BC!{T}(high_boundary_coefs,stencil_length,params,
                           derivative_order,grid_step::T,boundary_length,dx)
    aorder               = boundary_length - 1
    boundary_point_count = stencil_length - div(stencil_length,2) + 1
    flag                 = derivative_order*boundary_point_count%2
    original_coeffs      = zeros(T,boundary_length)
    r_diff               = one(T)
    mid                  = div(stencil_length,2)+1

    first_order_coeffs = params[2]*calculate_weights(1, (boundary_point_count-1)*grid_step, collect(zero(T) : grid_step : aorder * grid_step))
    first_order_coeffs[end] += dx*params[1]
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

    for i in 2 : boundary_point_count
        #=
            this means that a stencil will suffice ie. we dont't need to worry about the boundary point
            being considered in the high_boundary_coefs. Same code for low_boundary_coefs but reversed
            at the end
        =#
        if i > mid
            pos=i-1
            push!(high_boundary_coefs, calculate_weights(derivative_order, pos*grid_step, collect(zero(T) : grid_step : (boundary_length-1) * grid_step)))
        else
            pos=i-2
            push!(high_boundary_coefs, append!([zero(T)],calculate_weights(derivative_order, pos*grid_step, collect(zero(T) : grid_step : (boundary_length-1) * grid_step))))
        end
    end
    if flag == 1
        negate!(high_boundary_coefs)
    end
    reverse!(high_boundary_coefs)
    push!(high_boundary_coefs, original_coeffs[2:end])
    return r_diff
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


function Base.A_mul_B!{T<:Real}(x_temp::AbstractVector{T}, A::LinearOperator{T}, x::AbstractVector{T})
    convolve_BC_left!(x_temp, x, A)
    convolve_interior!(x_temp, x, A)
    convolve_BC_right!(x_temp, x, A)
    scale!(x_temp, 1/(A.dx^A.derivative_order))
end


function Base.A_mul_B!{T<:Real}(x_temp::AbstractArray{T,2}, A::LinearOperator{T}, M::AbstractMatrix{T})
    if size(x_temp) == reverse(size(M))
        for i = 1:size(M,1)
            A_mul_B!(view(x_temp,i,:), A, M[i,:])
        end
    else
        for i = 1:size(M,2)
            A_mul_B!(view(x_temp,:,i), A, M[:,i])
        end
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
