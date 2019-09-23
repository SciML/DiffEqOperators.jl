function Base.copyto!(L::AbstractMatrix{T}, A::DerivativeOperator{T}, N::Int) where T
    bl = A.boundary_point_count
    stencil_length = A.stencil_length
    stencil_pivot = use_winding(A) ? (1 + stencil_length%2) : div(stencil_length,2)    
    bstl = A.boundary_stencil_length
    
    coeff   = A.coefficients
    get_coeff = if coeff isa AbstractVector
        i -> coeff[i]
    elseif coeff isa Number
        i -> coeff
    else
        i -> true
    end

    for i in 1:bl
        cur_coeff   = get_coeff(i)
        cur_stencil = use_winding(A) && cur_coeff < 0 ? reverse(A.low_boundary_coefs[i]) : A.low_boundary_coefs[i]
        L[i,1:bstl] = cur_coeff * cur_stencil
    end

    for i in bl+1:N-bl
        cur_coeff   = get_coeff(i)
        stencil     = eltype(A.stencil_coefs) <: AbstractVector ? A.stencil_coefs[i] : A.stencil_coefs
        cur_stencil = use_winding(A) && cur_coeff < 0 ? reverse(stencil) : stencil
        L[i,i+1-stencil_pivot:i-stencil_pivot+stencil_length] = cur_coeff * cur_stencil
    end

    for i in N-bl+1:N
        cur_coeff   = get_coeff(i)
        cur_stencil = use_winding(A) && cur_coeff < 0 ? reverse(A.high_boundary_coefs[i-N+bl]) : A.high_boundary_coefs[i-N+bl]
        L[i,N-bstl+3:N+2] = cur_coeff * cur_stencil
    end
    
    L
end

LinearAlgebra.Array(A::DerivativeOperator{T}, N::Int=A.len) where T =
    copyto!(zeros(T, N, N+2), A, N)

SparseArrays.SparseMatrixCSC(A::DerivativeOperator{T}, N::Int=A.len) where T =
    copyto!(spzeros(T, N, N+2), A, N)

SparseArrays.sparse(A::DerivativeOperator{T}, N::Int=A.len) where T =
    BandedMatrix(A,N)

function BandedMatrices.BandedMatrix(A::DerivativeOperator{T}, N::Int=A.len) where T
    stencil_length = A.stencil_length
    bstl = A.boundary_stencil_length
    L = BandedMatrix{T}(Zeros(N, N+2), (max(stencil_length-3,0,bstl),max(stencil_length-1,0,bstl)))
    copyto!(L, A, N)
end

Base.convert(::Type{Mat}, A::DerivativeOperator) where {Mat<:Union{Array,SparseMatrixCSC,BandedMatrix}} =
    Mat(A)

Base.convert(::Type{AbstractMatrix},A::DerivativeOperator) =
    BandedMatrix(A)

################################################################################
# Boundary Padded Array concretizations
################################################################################

function LinearAlgebra.Array(Q::BoundaryPaddedArray{T,D,N,M,V,B}) where {T,D,N,M,V,B}
    S = size(Q)
    out = zeros(T, S...)
    dim = D
    ulowview = selectdim(out, dim, 1)
    uhighview = selectdim(out, dim, S[dim])
    uview = selectdim(out, dim, 2:(S[dim]-1))
    ulowview .= Q.lower
    uhighview .= Q.upper
    uview .= Q.u
    return out
end

function LinearAlgebra.Array(Q::ComposedBoundaryPaddedArray{T,N,M,V,B}) where {T,N,M,V,B}
    S = size(Q)
    out = zeros(T, S...)
    dimset = 1:N
    uview = out
    for dim in dimset
        ulowview = selectdim(out, dim, 1)
        uhighview = selectdim(out, dim, S[dim])
        uview = selectdim(uview, dim, 2:(S[dim]-1))
        for (index, otherdim) in enumerate(setdiff(dimset, dim))
            ulowview = selectdim(ulowview, index, 2:(S[otherdim]-1))
            uhighview = selectdim(uhighview, index, 2:(S[otherdim]-1))
        end
        ulowview .= Q.lower[dim]
        uhighview .= Q.upper[dim]
    end
    uview .= Q.u
    return out
end

function Base.convert(::Type{AbstractArray}, A::AbstractBoundaryPaddedArray)
    Array(A)
end

################################################################################
# Boundary Condition Operator concretizations
################################################################################

# * Atomic BCs

# ** Affine BCs

function LinearAlgebra.Array(Q::AffineBC{T}, N::Int) where {T}
    Q_L = [transpose(Q.a_l) transpose(zeros(T, N-length(Q.a_l))); Diagonal(ones(T,N)); transpose(zeros(T, N-length(Q.a_r))) transpose(Q.a_r)]
    Q_b = [Q.b_l; zeros(T,N); Q.b_r]
    return (Array(Q_L), Q_b)
end

function SparseArrays.SparseMatrixCSC(Q::AffineBC{T}, N::Int) where {T}
    Q_L = [transpose(Q.a_l) transpose(zeros(T, N-length(Q.a_l))); Diagonal(ones(T,N)); transpose(zeros(T, N-length(Q.a_r))) transpose(Q.a_r)]
    Q_b = [Q.b_l; zeros(T,N); Q.b_r]
    return (Q_L, Q_b)
end

function BandedMatrices.BandedMatrix(Q::AffineBC{T}, N::Int) where {T}
    # We want the concrete matrix to have as small bandwidths as
    # possible, and we accomplish this by dropping all trailing
    # zeros. This way, we do not write outside the bands of the
    # BandedMatrix.
    a_r = Q.a_r[1:something(findlast(!iszero, Q.a_r), 0)]
    a_l = Q.a_l[1:something(findlast(!iszero, Q.a_l), 0)]

    # Compute bandwidths; the BC matrix should have the shape
    #
    # [a b c ...
    #  1
    #    1
    #      1
    #        .
    #          1
    #  ... x y z]
    #
    # where a,b,c,... and ...,x,y,z are determined by the boundary
    # conditions. If these coefficients are zero (Dirichlet0BC), then
    # the proper bandwidths are (l,u) = (1,-1).    
    l = max(count(!iszero, a_r)+1, 1)
    u = max(count(!iszero, a_l)-1, -1)

    Q_l = BandedMatrix((-1 => ones(T,N),), (N+2,N), (l, u))
    for (j,e) ∈ enumerate(a_l)
        BandedMatrices.inbands_setindex!(Q_l, e, 1, j)
    end
    for (j,e) ∈ enumerate(a_r)
        BandedMatrices.inbands_setindex!(Q_l, e, N+2, N-length(a_r)+j)
    end
    Q_b = [Q.b_l; zeros(T,N); Q.b_r]
    
    Q_l, Q_b
end

"""
    sparse(Q::AffineBC, N)

Since affine boundary conditions are representable by banded matrices,
that is the default sparse concretization; if you want a
`SparseMatrixCSC`, use `SparseMatrixCSC(Q, N)` instead.
"""
SparseArrays.sparse(Q::AffineBC, N::Int) = BandedMatrix(Q,N)

# ** Periodic BCs

LinearAlgebra.Array(Q::PeriodicBC{T}, N::Int) where T =
    ([transpose(zeros(T, N-1)) one(T)
      Diagonal(ones(T,N))
      one(T) transpose(zeros(T, N-1))],
     zeros(T,N+2))

SparseArrays.SparseMatrixCSC(Q::PeriodicBC{T}, N::Int) where T =
    (vcat(hcat(zeros(T, 1,N-1), one(T)),
          Diagonal(ones(T,N)),
          hcat(one(T), zeros(T, 1, N-1))),
     zeros(T,N+2))

SparseArrays.sparse(Q::PeriodicBC{T}, N::Int) where T = SparseMatrixCSC(Q,N)

BandedMatrices.BandedMatrix(::PeriodicBC, ::Int) =
    throw(ArgumentError("Periodic boundary conditions should be concretized as sparse matrices"))

LinearAlgebra.Array(Q::BoundaryPaddedVector) = [Q.l; Q.u; Q.r]

# Multi dimensional BC operators

"""
Returns a tuple, the first element of which is an array of the shape of the boundary,
filled with the linear operator parts of the respective Atomic BCs.
the second element is a simularly sized array of the affine parts.
"""
function LinearAlgebra.Array(Q::MultiDimDirectionalBC{T, B, D, N, K}, M) where {T, B, D,N,K}
    bc_tuples = Array.(Q.BCs, fill(M, size(Q.BCs)))
    Q_L = [bc_tuple[1] for bc_tuple in bc_tuples]
    inds = Array(1:N)
    inds[1], inds[D] = inds[D], inds[1]
    Q_b = [permutedims(add_dims(bc_tuple[2], N-1), inds) for bc_tuple in bc_tuples]

    return (Q_L, Q_b)
end

"""
Returns a tuple, the first element of which is a sparse array of the shape of the boundary,
filled with the linear operator parts of the respective Atomic BCs.
the second element is a simularly sized array of the affine parts.
"""
function SparseArrays.SparseMatrixCSC(Q::MultiDimDirectionalBC{T, B, D, N, K}, M) where {T, B, D,N,K}
    bc_tuples = sparse.(Q.BCs, fill(M, size(Q.BCs)))
    Q_L = [bc_tuple[1] for bc_tuple in bc_tuples]
    inds = Array(1:N)
    inds[1], inds[D] = inds[D], inds[1]
    Q_b = [permutedims(add_dims(bc_tuple[2], N-1), inds) for bc_tuple in bc_tuples]

    return (Q_L, Q_b)
end

SparseArrays.sparse(Q::MultiDimDirectionalBC, N) = SparseMatrixCSC(Q, N)

function BandedMatrices.BandedMatrix(Q::MultiDimDirectionalBC{T, B, D, N, K}, M) where {T, B, D,N,K}
    bc_tuples = BandedMatrix.(Q.BCs, fill(M, size(Q.BCs)))
    Q_L = [bc_tuple[1] for bc_tuple in bc_tuples]
    inds = Array(1:N)
    inds[1], inds[D] = inds[D], inds[1]
    Q_b = [permutedims(add_dims(bc_tuple[2], N-1),inds) for bc_tuple in bc_tuples]

    return (Q_L, Q_b)
end

"""
Returns a Tuple of MultiDimDirectionalBC Array concretizations, one for each dimension
"""
LinearAlgebra.Array(Q::ComposedMultiDimBC, Ns) = Tuple(Array.(Q.BCs, Ns))
SparseArrays.SparseMatrixCSC(Q::ComposedMultiDimBC, Ns...) = Tuple(sparse.(Q.BCs, Ns))
SparseArrays.sparse(Q::ComposedMultiDimBC, Ns) = SparseMatrixCSC(Q, Ns)
BandedMatrices.BandedMatrix(Q::ComposedMultiDimBC, Ns) = Tuple(BandedMatrix.(Q.BCs, Ns))

# HIgher Dimensional Concretizations. The following concretizations return two dimensional arrays
# which operate on flattened vectors. Mshape is the size of the unflattened array on which A is operating on.

function LinearAlgebra.Array(A::DerivativeOperator{T,N}, Mshape) where {T,N}
    # Case where A is not differentiating along the first dimension
    if N != 1
        n = 1
        for M_i in Mshape[1:N-1]
            n *= M_i
        end
        B = Kron(Array(A), Eye(n))
        if N != length(Mshape)
            n = 1
            for M_i in Mshape[N+1:end]
                n *= M_i
            end
            B = Kron(Eye(n), B)
        end

    # Case where A is differentiating along hte first dimension
    else
        n = 1
        for M_i in Mshape[2:end]
            n *= M_i
        end
        B = Kron(Eye(n), Array(A))
    end
    return Array(B)
end

function SparseArrays.SparseMatrixCSC(A::DerivativeOperator{T,N}, Mshape) where {T,N}
    # Case where A is not differentiating along the first dimension
    if N != 1
        n = 1
        for M_i in Mshape[1:N-1]
            n *= M_i
        end
        B = Kron(sparse(A), sparse(I,n,n))
        if N != length(Mshape)
            n = 1
            for M_i in Mshape[N+1:end]
                n *= M_i
            end
            B = Kron(sparse(I,n,n), B)
        end

    # Case where A is differentiating along hte first dimension
    else
        n = 1
        for M_i in Mshape[2:end]
            n *= M_i
        end
        B = Kron(sparse(I,n,n), sparse(A))
    end
    return sparse(B)
end

function SparseArrays.sparse(A::DerivativeOperator{T,N}, Mshape) where {T,N}
    return SparseMatrixCSC(A,Mshape)
end

function BandedMatrices.BandedMatrix(A::DerivativeOperator{T,N}, Mshape) where {T,N}
    # Case where A is not differentiating along the first dimension
    if N != 1
        n = 1
        for M_i in Mshape[1:N-1]
            n *= M_i
        end
        B = Kron(BandedMatrix(A), Eye(n))
        if N != length(Mshape)
            n = 1
            for M_i in Mshape[N+1:end]
                n *= M_i
            end
            B = Kron(Eye(n), B)
        end

    # Case where A is differentiating along hte first dimension
    else
        n = 1
        for M_i in Mshape[2:end]
            n *= M_i
        end
        B = Kron(BandedMatrix(Eye(n)), BandedMatrix(A))
    end
    return BandedMatrix(B)
end

function BlockBandedMatrices.BandedBlockBandedMatrix(A::DerivativeOperator{T,N}, Mshape) where {T,N}
    # Case where A is not differentiating along the first dimension
    if N != 1
        n = 1
        for M_i in Mshape[1:N-1]
            n *= M_i
        end
        B = Kron(BandedMatrix(A), Eye(n))
        if N != length(Mshape)
            n = 1
            for M_i in Mshape[N+1:end]
                n *= M_i
            end
            B = Kron(Eye(n), B)
        end

    # Case where A is differentiating along hte first dimension
    else
        n = 1
        for M_i in Mshape[2:end]
            n *= M_i
        end
        B = Kron(BandedMatrix(Eye(n)), BandedMatrix(A))
    end
    return BandedBlockBandedMatrix(B)
end
