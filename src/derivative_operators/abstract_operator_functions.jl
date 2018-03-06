# ~ bound checking functions ~
checkbounds(A::AbstractDerivativeOperator, k::Integer, j::Integer) =
    (0 < k ≤ size(A, 1) && 0 < j ≤ size(A, 2) || throw(BoundsError(A, (k,j))))

checkbounds(A::AbstractDerivativeOperator, kr::Range, j::Integer) =
    (checkbounds(A, first(kr), j); checkbounds(A,  last(kr), j))

checkbounds(A::AbstractDerivativeOperator, k::Integer, jr::Range) =
    (checkbounds(A, k, first(jr)); checkbounds(A, k,  last(jr)))

checkbounds(A::AbstractDerivativeOperator, kr::Range, jr::Range) =
    (checkbounds(A, kr, first(jr)); checkbounds(A, kr,  last(jr)))

checkbounds(A::AbstractDerivativeOperator, k::Colon, j::Integer) =
    (0 < j ≤ size(A, 2) || throw(BoundsError(A, (size(A,1),j))))

checkbounds(A::AbstractDerivativeOperator, k::Integer, j::Colon) =
    (0 < k ≤ size(A, 1) || throw(BoundsError(A, (k,size(A,2)))))


# BandedMatrix{A,B,C,D}(A::DerivativeOperator{A,B,C,D}) = BandedMatrix(full(A, A.stencil_length), A.stencil_length, div(A.stencil_length,2), div(A.stencil_length,2))

# ~~ getindex ~~
@inline function getindex(A::AbstractDerivativeOperator, i::Int, j::Int)
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
@inline getindex(A::AbstractDerivativeOperator, kr::Colon, jr::Colon) = full(A)

@inline function getindex(A::AbstractDerivativeOperator, rc::Colon, j)
    T = eltype(A.stencil_coefs)
    v = zeros(T, A.dimension)
    v[j] = one(T)
    copy!(v, A*v)
    return v
end


# symmetric right now
@inline function getindex(A::AbstractDerivativeOperator, i, cc::Colon)
    T = eltype(A.stencil_coefs)
    v = zeros(T, A.dimension)
    v[i] = one(T)
    copy!(v, A*v)
    return v
end


# UnitRanges
@inline function getindex(A::AbstractDerivativeOperator, rng::UnitRange{Int}, cc::Colon)
    m = full(A)
    return m[rng, cc]
end


@inline function getindex(A::AbstractDerivativeOperator, rc::Colon, rng::UnitRange{Int})
    m = full(A)
    return m[rnd, cc]
end

@inline function getindex(A::AbstractDerivativeOperator, r::Int, rng::UnitRange{Int})
    m = A[r, :]
    return m[rng]
end


@inline function getindex(A::AbstractDerivativeOperator, rng::UnitRange{Int}, c::Int)
    m = A[:, c]
    return m[rng]
end


@inline function getindex(A::AbstractDerivativeOperator{T}, rng::UnitRange{Int}, cng::UnitRange{Int}) where T
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

#=
    This the basic A_Mul_B! function which is broken up into 3 parts ie. handling the left
    boundary, handling the interior and handling the right boundary. Finally the vector is
    scaled appropriately according to the derivative order and the degree of discretizaiton.
    We also update the time stamp of the DerivativeOperator inside to manage time dependent
    boundary conditions.
=#
function Base.A_mul_B!(x_temp::AbstractVector{T}, A::Union{DerivativeOperator{T},UpwindOperator{T}}, x::AbstractVector{T}) where T<:Real
    convolve_BC_left!(x_temp, x, A)
    convolve_interior!(x_temp, x, A)
    convolve_BC_right!(x_temp, x, A)
    scale!(x_temp, 1./(A.dx^A.derivative_order))
    A.t[] += 1 # incrementing the internal time stamp
end

#=
    This definition of the A_mul_B! function makes it possible to apply the LinearOperator on
    a matrix and not just a vector. It basically transforms the rows one at a time.
=#
function Base.A_mul_B!(x_temp::AbstractArray{T,2}, A::AbstractDerivativeOperator{T}, M::AbstractMatrix{T}) where T<:Real
    if size(x_temp) == reverse(size(M))
        for i = 1:size(M,1)
            A_mul_B!(view(x_temp,i,:), A, view(M,i,:))
        end
    else
        for i = 1:size(M,2)
            A_mul_B!(view(x_temp,:,i), A, view(M,:,i))
        end
    end
end


# Base.length(A::AbstractDerivativeOperator) = A.stencil_length
Base.ndims(A::AbstractDerivativeOperator) = 2
Base.size(A::AbstractDerivativeOperator) = (A.dimension, A.dimension)
Base.length(A::AbstractDerivativeOperator) = reduce(*, size(A))


#=
    For the evenly spaced grid we have a symmetric matrix
=#
Base.transpose(A::Union{DerivativeOperator,UpwindOperator}) = A
Base.ctranspose(A::Union{DerivativeOperator,UpwindOperator}) = A
Base.issymmetric(::Union{DerivativeOperator,UpwindOperator}) = true

#=
    This function ideally should give us a matrix which when multiplied with the input vector
    returns the derivative. But the presence of different boundary conditons complicates the
    situation. It is not possible to incorporate the full information of the boundary conditions
    in the matrix as they are additive in nature. So in this implementation we will just
    return the inner stencil of the matrix transformation. The user will have to manually
    incorporate the BCs at the ends.
=#
function Base.full(A::AbstractDerivativeOperator{T}, N::Int=A.dimension) where T
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


#=
    This function ideally should give us a matrix which when multiplied with the input vector
    returns the derivative. But the presence of different boundary conditons complicates the
    situation. It is not possible to incorporate the full information of the boundary conditions
    in the matrix as they are additive in nature. So in this implementation we will just
    return the inner stencil of the matrix transformation. The user will have to manually
    incorporate the BCs at the ends.
=#
function Base.sparse(A::AbstractDerivativeOperator{T}) where T
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
