function LinearAlgebra.Array(A::DerivativeOperator{T}, N::Int=A.len) where T
    L = zeros(T, N, N+2)
    bl = A.boundary_point_count
    stencil_length = A.stencil_length
    bstl = A.boundary_stencil_length
    coeff   = A.coefficients

    if use_winding(A)
        stencil_pivot = 1 + A.stencil_length%2
    else
        stencil_pivot = div(stencil_length,2)
    end

    for i in 1:bl
        cur_coeff   = typeof(coeff)   <: AbstractVector ? coeff[i] : coeff isa Number ? coeff : true
        cur_stencil = use_winding(A) && cur_coeff < 0 ? reverse(A.low_boundary_coefs[i]) : A.low_boundary_coefs[i]
        L[i,1:bstl] = cur_coeff * cur_stencil
    end

    for i in bl+1:N-bl
        cur_coeff   = typeof(coeff)   <: AbstractVector ? coeff[i] : coeff isa Number ? coeff : true
        stencil     = eltype(A.stencil_coefs) <: AbstractVector ? A.stencil_coefs[i] : A.stencil_coefs
        cur_stencil = use_winding(A) && cur_coeff < 0 ? reverse(stencil) : stencil
        L[i,i+1-stencil_pivot:i-stencil_pivot+stencil_length] = cur_coeff * cur_stencil
    end

    for i in N-bl+1:N
        cur_coeff   = typeof(coeff)   <: AbstractVector ? coeff[i] : coeff isa Number ? coeff : true
        cur_stencil = use_winding(A) && cur_coeff < 0 ? reverse(A.high_boundary_coefs[i-N+bl]) : A.high_boundary_coefs[i-N+bl]
        L[i,N-bstl+3:N+2] = cur_coeff * cur_stencil
    end
    return L
end

function SparseArrays.SparseMatrixCSC(A::DerivativeOperator{T}, N::Int=A.len) where T
    L = spzeros(T, N, N+2)
    bl = A.boundary_point_count
    stencil_length = A.stencil_length
    stencil_pivot = div(stencil_length,2)
    bstl = A.boundary_stencil_length
    coeff   = A.coefficients

    if use_winding(A)
        stencil_pivot = 1 + A.stencil_length%2
    else
        stencil_pivot = div(stencil_length,2)
    end

    for i in 1:A.boundary_point_count
        cur_coeff   = typeof(coeff)   <: AbstractVector ? coeff[i] : coeff isa Number ? coeff : true
        cur_stencil = use_winding(A) && cur_coeff < 0 ? reverse(A.low_boundary_coefs[i]) : A.low_boundary_coefs[i]
        L[i,1:bstl] = cur_coeff * cur_stencil
    end

    for i in bl+1:N-bl
        cur_coeff   = typeof(coeff)   <: AbstractVector ? coeff[i] : coeff isa Number ? coeff : true
        stencil     = eltype(A.stencil_coefs) <: AbstractVector ? A.stencil_coefs[i] : A.stencil_coefs
        cur_stencil = use_winding(A) && cur_coeff < 0 ? reverse(stencil) : stencil
        L[i,i+1-stencil_pivot:i-stencil_pivot+stencil_length] = cur_coeff * cur_stencil
    end

    for i in N-bl+1:N
        cur_coeff   = typeof(coeff)   <: AbstractVector ? coeff[i] : coeff isa Number ? coeff : true
        cur_stencil = use_winding(A) && cur_coeff < 0 ? reverse(A.high_boundary_coefs[i-N+bl]) : A.high_boundary_coefs[i-N+bl]
        L[i,N-bstl+3:N+2] = cur_coeff * cur_stencil
    end
    return L
end

function SparseArrays.sparse(A::DerivativeOperator{T}, N::Int=A.len) where T
    SparseMatrixCSC(A,N)
end

function BandedMatrices.BandedMatrix(A::DerivativeOperator{T}, N::Int=A.len) where T
    bl = A.boundary_point_count
    stencil_length = A.stencil_length
    bstl = A.boundary_stencil_length
    coeff   = A.coefficients
    if use_winding(A)
        stencil_pivot = 1 + A.stencil_length%2
    else
        stencil_pivot = div(stencil_length,2)
    end
    L = BandedMatrix{T}(Zeros(N, N+2), (max(stencil_length-3,0,bstl),max(stencil_length-1,0,bstl)))

    for i in 1:A.boundary_point_count
        cur_coeff   = typeof(coeff)   <: AbstractVector ? coeff[i] : coeff isa Number ? coeff : true
        cur_stencil = use_winding(A) && cur_coeff < 0 ? reverse(A.low_boundary_coefs[i]) : A.low_boundary_coefs[i]
        L[i,1:bstl] = cur_coeff * cur_stencil
    end
    for i in bl+1:N-bl
        cur_coeff   = typeof(coeff)   <: AbstractVector ? coeff[i] : coeff isa Number ? coeff : true
        stencil     = eltype(A.stencil_coefs) <: AbstractVector ? A.stencil_coefs[i] : A.stencil_coefs
        cur_stencil = use_winding(A) && cur_coeff < 0 ? reverse(stencil) : stencil
        L[i,i+1-stencil_pivot:i-stencil_pivot+stencil_length] = cur_coeff * cur_stencil
    end
    for i in N-bl+1:N
        cur_coeff   = typeof(coeff)   <: AbstractVector ? coeff[i] : coeff isa Number ? coeff : true
        cur_stencil = use_winding(A) && cur_coeff < 0 ? reverse(A.high_boundary_coefs[i-N+bl]) : A.high_boundary_coefs[i-N+bl]
        L[i,N-bstl+3:N+2] = cur_coeff * cur_stencil
    end
    return L
end

function Base.convert(::Type{Array},A::DerivativeOperator{T}) where T
    Array(A)
end

function Base.convert(::Type{SparseMatrixCSC},A::DerivativeOperator{T}) where T
    SparseMatrixCSC(A)
end

function Base.convert(::Type{BandedMatrix},A::DerivativeOperator{T}) where T
    BandedMatrix(A)
end

function Base.convert(::Type{AbstractMatrix},A::DerivativeOperator{T}) where T
    BandedMatrix(A)
end


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

#Atomic BCs
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
    Q_l = BandedMatrix{T}(Eye(N), (length(Q.a_r)-1, length(Q.a_l)-1))
    inbands_setindex!(Q_L, Q.a_l, 1, 1:length(Q.a_l))
    inbands_setindex!(Q_L, Q.a_r, N, (N-length(Q.a_r)+1):N)
    Q_b = [Q.b_l; zeros(T,N); Q.b_r]
    return (Q_L, Q_b)
end

function SparseArrays.sparse(Q::AffineBC{T}, N::Int) where {T}
    SparseMatrixCSC(Q,N)
end

LinearAlgebra.Array(Q::PeriodicBC{T}, N::Int) where T = (Array([transpose(zeros(T, N-1)) one(T); Diagonal(ones(T,N)); one(T) transpose(zeros(T, N-1))]), zeros(T, N))
SparseArrays.SparseMatrixCSC(Q::PeriodicBC{T}, N::Int) where T = ([transpose(zeros(T, N-1)) one(T); Diagonal(ones(T,N)); one(T) transpose(zeros(T, N-1))], zeros(T, N))
SparseArrays.sparse(Q::PeriodicBC{T}, N::Int) where T = SparseMatrixCSC(Q,N)
function BandedMatrices.BandedMatrix(Q::PeriodicBC{T}, N::Int) where T #Not reccomended!
    Q_array = BandedMatrix{T}(Eye(N), (N-1, N-1))
    Q_array[1, end] = one(T)
    Q_array[1, 1] = zero(T)
    Q_array[end, 1] = one(T)
    Q_array[end, end] = zero(T)

    return (Q_array, zeros(T, N))
end

function LinearAlgebra.Array(Q::BoundaryPaddedVector)
    return [Q.l; Q.u; Q.r]
end

function Base.convert(::Type{Array},A::AbstractBC{T}) where T
    Array(A)
end

function Base.convert(::Type{SparseMatrixCSC},A::AbstractBC{T}) where T
    SparseMatrixCSC(A)
end

function Base.convert(::Type{AbstractMatrix},A::AbstractBC{T}) where T
    SparseMatrixCSC(A)
end

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

################################################################################
# Upwind Operator Concretization
################################################################################

# TODO: Remove the generality of the non-uniform grid from this implementation
function LinearAlgebra.Array(A::DerivativeOperator{T,N,true}, len::Int=A.len) where {T,N}
    L = zeros(T, len, len+2)
    bpc = A.boundary_point_count
    stl = A.stencil_length
    bstl = A.boundary_stencil_length
    coeff   = A.coefficients

    # downwind stencils at low boundary
    downwind_stencils = A.low_boundary_coefs
    # upwind stencils at upper boundary
    upwind_stencils = A.high_boundary_coefs
    # interior stencils
    stencils = A.stencil_coefs

    for i in 1:bpc
        cur_coeff   = coeff[i]
        if cur_coeff >= 0
            cur_stencil = eltype(stencils) <: AbstractVector ? stencils[i] : stencils
            L[i,i+1:i+stl] = cur_coeff*cur_stencil
        else
            cur_stencil = downwind_stencils[i]
            L[i,1:bstl] = cur_coeff * cur_stencil
        end
    end

    for i in bpc+1:len-bpc
        cur_coeff   = coeff[i]
        cur_stencil = eltype(stencils) <: AbstractVector ? stencils[i-A.boundary_point_count] : stencils
        cur_stencil = cur_coeff >= 0 ? cur_stencil : A.derivative_order % 2 == 0 ? reverse(cur_stencil) : -1*reverse(cur_stencil)
        if cur_coeff >= 0
            L[i,i+1:i+stl] = cur_coeff * cur_stencil
        else
            L[i,i-stl+2:i+1] = cur_coeff * cur_stencil
        end
    end

    for i in len-bpc+1:len
        cur_coeff   = coeff[i]
        if cur_coeff < 0
            cur_stencil = eltype(stencils) <: AbstractVector ? stencils[i] : stencils # TODO, fix the indexing here for the vectors
            cur_stencil = ((-1)^A.derivative_order)*reverse(cur_stencil) #TODO make this consistent with above
            L[i,i-stl+2:i+1] = cur_coeff * cur_stencil
        else
            cur_stencil = upwind_stencils[i-len+bpc]
            L[i,len-bstl+3:len+2] = cur_coeff * cur_stencil
        end
    end
    return L
end

function LinearAlgebra.Array(A::DerivativeOperator{T,N,true,M}, len::Int=A.len) where {T,N,M<:AbstractArray{T}}
    L = zeros(T, len, len+2)
    bpc = A.boundary_point_count
    stl = A.stencil_length
    bstl = A.boundary_stencil_length
    coeff   = A.coefficients

    for i in 1:bpc
        cur_coeff   = coeff[i]
        if cur_coeff >= 0
            L[i,i+1:i+stl] = cur_coeff * A.low_boundary_coefs[1,i]
        else
            L[i,1:bstl] = cur_coeff * A.low_boundary_coefs[2,i]
        end
    end

    for i in bpc+1:len-bpc
        cur_coeff   = coeff[i]
        if cur_coeff >= 0
            L[i,i+1:i+stl] = cur_coeff * A.stencil_coefs[1,i-bpc]
        else
            L[i,i-stl+2:i+1] = cur_coeff * A.stencil_coefs[2,i-bpc]
        end
    end

    for i in len-bpc+1:len
        cur_coeff   = coeff[i]
        if cur_coeff < 0
            L[i,i-stl+2:i+1] = cur_coeff * A.high_boundary_coefs[2,i-len+bpc]
        else
            L[i,len-bstl+3:len+2] = cur_coeff * A.high_boundary_coefs[1,i-len+bpc]
        end
    end
    return L
end
