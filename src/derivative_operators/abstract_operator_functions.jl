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

@inline function getindex(A::AbstractDerivativeOperator, i::Int, j::Int)
    @boundscheck checkbounds(A, i, j)
    # TODO: Implement the lazy version
    if use_winding(A)
        return Array(A)[i,j]
    end

    bpc = A.boundary_point_count
    N = A.len
    bsl = A.boundary_stencil_length
    slen = A.stencil_length
    if bpc > 0 && 1<=i<=bpc
        if j > bsl
            return 0
        else
            return A.low_boundary_coefs[i][j]
        end
    elseif bpc > 0 && (N-bpc)<i<=N
        if j < N+2-bsl
            return 0
        else
            return A.high_boundary_coefs[bpc-(N-i)][bsl-(N+2-j)]
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
@inline getindex(A::AbstractDerivativeOperator, ::Colon, ::Colon) = Array(A)

@inline function getindex(A::AbstractDerivativeOperator, ::Colon, j)
    T = eltype(A.stencil_coefs)
    v = zeros(T, A.len)
    v[j] = one(T)
    copyto!(v, A*v)
    return v
end

@inline function getindex(A::AbstractDerivativeOperator, i, ::Colon)
    @boundscheck checkbounds(A, i, 1)
    T = eltype(A.stencil_coefs)
    v = zeros(T, A.len+2)

    bpc = A.boundary_point_count
    N = A.len
    bsl = A.boundary_stencil_length
    slen = A.stencil_length

    if bpc > 0 && 1<=i<=bpc
        v[1:bsl] .= A.low_boundary_coefs[i]
    elseif bpc > 0 && (N-bpc)<i<=N
         v[1:bsl]  .= A.high_boundary_coefs[i-(N-1)]
    else
        if use_winding(A)
            v[i-bpc+slen:i-bpc+2*slen-1] .= A.stencil_coefs
        else
            v[i-bpc:i-bpc+slen-1] .= A.stencil_coefs
        end
    end
    return v
end

# UnitRanges
@inline function getindex(A::AbstractDerivativeOperator, rng::UnitRange{Int}, ::Colon)
    m = BandedMatrix(A)
    return m[rng, cc]
end


@inline function getindex(A::AbstractDerivativeOperator, ::Colon, rng::UnitRange{Int})
    m = BandedMatrix(A)
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
    return BandedMatrix(A)[rng,cng]
end

# Base.length(A::AbstractDerivativeOperator) = A.stencil_length
Base.ndims(A::AbstractDerivativeOperator) = 2
Base.size(A::AbstractDerivativeOperator) = (A.len, A.len + 2)
Base.size(A::AbstractDerivativeOperator,i::Integer) = size(A)[i]
Base.length(A::AbstractDerivativeOperator) = reduce(*, size(A))

#=
    For the evenly spaced grid we have a symmetric matrix
=#
Base.transpose(A::DerivativeOperator) = A
Base.adjoint(A::DerivativeOperator) = A
LinearAlgebra.issymmetric(::DerivativeOperator) = true

#=
    Fallback methods that use the full representation of the operator
=#
Base.exp(A::AbstractDerivativeOperator{T}) where T = exp(convert(A))
Base.:\(A::AbstractVecOrMat, B::AbstractDerivativeOperator) = A \ convert(Array,B)
Base.:\(A::AbstractDerivativeOperator, B::AbstractVecOrMat) = Array(A) \ B
Base.:/(A::AbstractVecOrMat, B::AbstractDerivativeOperator) = A / convert(Array,B)
Base.:/(A::AbstractDerivativeOperator, B::AbstractVecOrMat) = Array(A) / B

#=
    The Inf opnorm can be calculated easily using the stencil coefficients, while other opnorms
    default to compute from the full matrix form.
=#
function LinearAlgebra.opnorm(A::DerivativeOperator, p::Real=2)
    if p == Inf
        sum(abs.(A.stencil_coefs)) / A.dx^A.derivative_order
    else
        opnorm(BandedMatrix(A), p)
    end
end

########################################################################

get_type(::AbstractDerivativeOperator{T}) where {T} = T

function *(M::AbstractMatrix,A::AbstractDerivativeOperator)
    y = zeros(promote_type(eltype(A),eltype(M)), size(M,1), size(A,2))
    LinearAlgebra.mul!(y, M, BandedMatrix(A))
    return y
end


function *(A::AbstractDerivativeOperator,B::AbstractDerivativeOperator)
    return BandedMatrix(A)*BandedMatrix(B)
end

################################################################################

function *(coeff_func::Function, A::DerivativeOperator{T,N,Wind}) where {T,N,Wind}
    coefficients = A.coefficients === nothing ? Vector{T}(undef,A.len) : A.coefficients
    DerivativeOperator{T,N,Wind,typeof(A.dx),typeof(A.stencil_coefs),
                       typeof(A.low_boundary_coefs),typeof(coefficients),
                       typeof(coeff_func)}(
        A.derivative_order, A.approximation_order,
        A.dx, A.len, A.stencil_length,
        A.stencil_coefs,
        A.boundary_stencil_length,
        A.boundary_point_count,
        A.low_boundary_coefs,
        A.high_boundary_coefs,coefficients,coeff_func
        )
end

################################################################################

function DiffEqBase.update_coefficients!(A::AbstractDerivativeOperator,u,p,t)
    if A.coeff_func !== nothing
        A.coeff_func(A.coefficients,u,p,t)
    end
end

################################################################################

(L::DerivativeOperator)(u,p,t) = L*u
(L::DerivativeOperator)(du,u,p,t) = mul!(du,L,u)
