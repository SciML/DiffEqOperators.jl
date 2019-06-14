#= Worker functions=#
low(i::Int, mid::Int, bpc::Int) = Int(mid + (i-1)*(1-mid)/bpc)
high(i::Int, mid::Int, bpc::Int, slen::Int, L::Int) = Int(slen - (slen-mid)*(i-L+bpc)/(bpc))

# used in general dirichlet BC. To simulate a constant value beyond the boundary
limit(i, N) = N>=i>=1 ? i : (i<1 ? 1 : N)

# ~ bound checking functions ~
checkbounds(A::AbstractDerivativeOperator, k::Integer, j::Integer) =
    (0 < k ≤ size(A, 1) && 0 < j ≤ size(A, 2) || throw(BoundsError(A, (k,j))))

checkbounds(A::AbstractDerivativeOperator, kr::AbstractRange, j::Integer) =
    (checkbounds(A, first(kr), j); checkbounds(A,  last(kr), j))

checkbounds(A::AbstractDerivativeOperator, k::Integer, jr::AbstractRange) =
    (checkbounds(A, k, first(jr)); checkbounds(A, k,  last(jr)))

checkbounds(A::AbstractDerivativeOperator, kr::AbstractRange, jr::AbstractRange) =
    (checkbounds(A, kr, first(jr)); checkbounds(A, kr,  last(jr)))

checkbounds(A::AbstractDerivativeOperator, k::Colon, j::Integer) =
    (0 < j ≤ size(A, 2) || throw(BoundsError(A, (size(A,1),j))))

checkbounds(A::AbstractDerivativeOperator, k::Integer, j::Colon) =
    (0 < k ≤ size(A, 1) || throw(BoundsError(A, (k,size(A,2)))))

# ~~ getindex ~~
# @inline function getindex(A::AbstractDerivativeOperator, i::Int, j::Int)
#     @boundscheck checkbounds(A, i, j)
#     mid = div(A.stencil_length, 2) + 1
#     bpc = A.stencil_length - mid
#     l = max(1, low(j, mid, bpc))
#     h = min(A.stencil_length, high(j, mid, bpc, A.stencil_length, A.dimension))
#     slen = h - l + 1
#     if abs(i - j) > div(slen, 2)
#         return 0
#     else
#         return A.stencil_coefs[mid + j - i]
#     end
# end


@inline function getindex(A::AbstractDerivativeOperator, i::Int, j::Int)
    @boundscheck checkbounds(A, i, j)
    bpc = A.boundary_point_count
    N = A.dimension
    bsl = A.boundary_stencil_length
    slen = A.stencil_length
    if bpc > 0 && 1<=i<=bpc
        if j > bsl
            return 0
        else
            return A.low_boundary_coefs[i][j]
        end
    elseif bpc > 0 && (N-bpc)<i<=N
        if j < N+3-bsl
            return 0
        else
            return A.high_boundary_coefs[i-(N-1)][j-1]
        end
    else
        if j < i-bpc || j > i+slen-bpc-1
            return 0
        else
            return A.stencil_coefs[j-i + 1 + bpc]
        end
    end
end


# scalar - colon - colon
@inline getindex(A::AbstractDerivativeOperator, kr::Colon, jr::Colon) = convert(Array,A)

@inline function getindex(A::AbstractDerivativeOperator, rc::Colon, j)
    T = eltype(A.stencil_coefs)
    v = zeros(T, A.dimension)
    v[j] = one(T)
    copyto!(v, A*v)
    return v
end


# symmetric right now
@inline function getindex(A::AbstractDerivativeOperator, i, cc::Colon)
    T = eltype(A.stencil_coefs)
    v = zeros(T, A.dimension)
    v[i] = one(T)
    copyto!(v, A*v)
    return v
end


# UnitRanges
@inline function getindex(A::AbstractDerivativeOperator, rng::UnitRange{Int}, cc::Colon)
    m = convert(Array,A)
    return m[rng, cc]
end


@inline function getindex(A::AbstractDerivativeOperator, rc::Colon, rng::UnitRange{Int})
    m = convert(Array,A)
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
            mul!(view(mat, :, i - cng[1] + 1), A, v)
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
            mul!(view(mat, i - rng[1] + 1, :), A, v)
            v[i] = zero(T)
        end
        return mat[:, cng]
    end
end

#=
    This definition of the mul! function makes it possible to apply the LinearOperator on
    a matrix and not just a vector. It basically transforms the rows one at a time.
=#
function LinearAlgebra.mul!(x_temp::AbstractArray{T,2}, A::AbstractDerivativeOperator{T}, M::AbstractMatrix{T}) where T<:Real
    if size(x_temp) == reverse(size(M))
        for i = 1:size(M,1)
            mul!(view(x_temp,i,:), A, view(M,i,:))
        end
    else
        for i = 1:size(M,2)
            mul!(view(x_temp,:,i), A, view(M,:,i))
        end
    end
end


# Base.length(A::AbstractDerivativeOperator) = A.stencil_length
Base.ndims(A::AbstractDerivativeOperator) = 2
Base.size(A::AbstractDerivativeOperator) = (A.dimension, A.dimension + 2)
Base.size(A::AbstractDerivativeOperator,i::Integer) = size(A)[i]
Base.length(A::AbstractDerivativeOperator) = reduce(*, size(A))


#=
    For the evenly spaced grid we have a symmetric matrix
=#
Base.transpose(A::Union{DerivativeOperator,UpwindOperator}) = A
Base.adjoint(A::Union{DerivativeOperator,UpwindOperator}) = A
LinearAlgebra.issymmetric(::Union{DerivativeOperator,UpwindOperator}) = true

#=
    Fallback methods that use the full representation of the operator
=#
Base.exp(A::AbstractDerivativeOperator{T}) where T = exp(convert(Array,A))
Base.:\(A::AbstractVecOrMat, B::AbstractDerivativeOperator) = A \ convert(Array,B)
Base.:\(A::AbstractDerivativeOperator, B::AbstractVecOrMat) = convert(Array,A) \ B
Base.:/(A::AbstractVecOrMat, B::AbstractDerivativeOperator) = A / convert(Array,B)
Base.:/(A::AbstractDerivativeOperator, B::AbstractVecOrMat) = convert(Array,A) / B

########################################################################

# Are these necessary?

get_type(::AbstractDerivativeOperator{T}) where {T} = T

function *(A::AbstractDerivativeOperator,x::AbstractVector)
    #=
        We will output a vector which is a supertype of the types of A and x
        to ensure numerical stability
    =#
    get_type(A) != eltype(x) ? error("DiffEqOperator and array are not of same type!") : nothing
    y = zeros(promote_type(eltype(A),eltype(x)), length(x))
    LinearAlgebra.mul!(y, A::AbstractDerivativeOperator, x::AbstractVector)
    return y
end


function *(A::AbstractDerivativeOperator,M::AbstractMatrix)
    #=
        We will output a vector which is a supertype of the types of A and x
        to ensure numerical stability
    =#
    get_type(A) != eltype(M) ? error("DiffEqOperator and array are not of same type!") : nothing
    y = zeros(promote_type(eltype(A),eltype(M)), size(M))
    LinearAlgebra.mul!(y, A::AbstractDerivativeOperator, M::AbstractMatrix)
    return y
end


function *(M::AbstractMatrix,A::AbstractDerivativeOperator)
    #=
        We will output a vector which is a supertype of the types of A and x
        to ensure numerical stability
    =#
    get_type(A) != eltype(M) ? error("DiffEqOperator and array are not of same type!") : nothing
    y = zeros(promote_type(eltype(A),eltype(M)), size(M))
    LinearAlgebra.mul!(y, A::AbstractDerivativeOperator, M::AbstractMatrix)
    return y
end


function *(A::AbstractDerivativeOperator,B::AbstractDerivativeOperator)
    # TODO: it will result in an operator which calculates
    #       the derivative of order A.dorder + B.dorder of
    #       approximation_order = min(approx_A, approx_B)
end


function convert(::Type{Array}, A::AbstractDerivativeOperator{T}, N::Int=A.dimension) where T
    @assert N >= A.stencil_length # stencil must be able to fit in the matrix
    mat = zeros(T, (N, N+2))
    v = zeros(T, N+2)
    bpc = A.boundary_point_count
    bsl = A.boundary_stencil_length
    if bpc > 0
        #=
            Since the boundary stencils are centred at the respective boundary points,
            the first point is also included
        =#
        for i=1:bpc
            mat[i,1:A.boundary_stencil_length] .= A.low_boundary_coefs[i]
        end
        for i=1:bpc
            mat[N-bpc+i,N+2-bsl+1:N+2] .= A.high_boundary_coefs[i]
        end
    end

    for i=1+bpc:N-bpc
        #=
            Copy the stencil directly
        =#
        mat[i,i-bpc:i+A.stencil_length-bpc-1] .= A.stencil_coefs
    end
    return mat
end
