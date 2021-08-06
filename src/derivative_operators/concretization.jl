#
# Casting to normal matrix types.
#
# This implements the casts described in README.md and the cast from
# BoundaryPaddedArray to Array.
#

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
        stencil     = eltype(A.stencil_coefs) <: AbstractVector ? A.stencil_coefs[i-bl] : A.stencil_coefs
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

SparseArrays.sparse(A::DerivativeOperator{T}, N::Int=A.len) where T = SparseMatrixCSC(A,N)

function BandedMatrices.BandedMatrix(A::DerivativeOperator{T}, N::Int=A.len) where T
    stencil_length = A.stencil_length
    bstl = A.boundary_stencil_length
    L = BandedMatrix{T}(Zeros(N, N+2), (bstl-3,bstl-1))
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

#Atomic BCs
function LinearAlgebra.Array(Q::AffineBC{T}, N::Int) where {T}
    Q_L = [transpose(Q.a_l) transpose(zeros(T, N-length(Q.a_l))); Diagonal(ones(T,N)); transpose(zeros(T, N-length(Q.a_r))) transpose(Q.a_r)]
    Q_b = [Q.b_l; zeros(T,N); Q.b_r]
    return (Array(Q_L), Q_b)
end

function SparseArrays.SparseMatrixCSC(Q::AffineBC{T}, N::Int) where {T}
    Q_l = spzeros(T, N+2, N)
    for i in 1:N
        Q_l[i+1,i] = one(T)
    end

    a_r = Q.a_r[findfirst(!iszero, Q.a_r) isa Nothing ? something(end-1:end) : something(findfirst(!iszero, Q.a_r):end)]
    a_l = Q.a_l[findfirst(!iszero, Q.a_l) isa Nothing ? something(end-1:end) : something(findfirst(!iszero, Q.a_l):end)]
    for (j,e) ∈ enumerate(a_l)
        BandedMatrices.inbands_setindex!(Q_l, e, 1, j)
    end
    for (j,e) ∈ enumerate(a_r)
        BandedMatrices.inbands_setindex!(Q_l, e, N+2, N-length(a_r)+j)
    end
    
    Q_b = spzeros(T,N+2)
    Q_b[1] = Q.b_l
    Q_b[N+2] = Q.b_r
    return (Q_l,Q_b)
end

function BandedMatrices.BandedMatrix(Q::AffineBC{T}, N::Int) where {T}
    a_r = Q.a_r[findfirst(!iszero, Q.a_r) isa Nothing ? something(end-1:end) : something(findfirst(!iszero, Q.a_r):end)]
    a_l = Q.a_l[findfirst(!iszero, Q.a_l) isa Nothing ? something(end-1:end) : something(findfirst(!iszero, Q.a_l):end)]

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
    return (Q_l, Q_b)
end

function SparseArrays.sparse(Q::AffineBC{T}, N::Int) where {T}
    SparseMatrixCSC(Q,N)
end

LinearAlgebra.Array(Q::PeriodicBC{T}, N::Int) where T = (Array([transpose(zeros(T, N-1)) one(T); Diagonal(ones(T,N)); one(T) transpose(zeros(T, N-1))]), zeros(T, N+2))
function SparseArrays.SparseMatrixCSC(Q::PeriodicBC{T}, N::Int) where T
    Q_l = spzeros(T,N+2,N)
    for i in 1:N
       Q_l[i+1,i] = one(T)
    end
    Q_l[1, end] = one(T)
    Q_l[end, 1] = one(T)
    Q_b = spzeros(T,N+2)
    return (Q_l,Q_b)
end
SparseArrays.sparse(Q::PeriodicBC{T}, N::Int) where T = SparseMatrixCSC(Q,N)
function BandedMatrices.BandedMatrix(Q::PeriodicBC{T}, N::Int) where T #Not recommended!
    Q_array = BandedMatrix{T}((-1 => ones(T,N),), (N+2, N), (N+1,N+1))
    Q_array[1, end] = one(T)
    Q_array[end, 1] = one(T)
    return (Q_array, zeros(T, N+2))
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

# Multidimensional BC operators
_concretize(Q::MultiDimDirectionalBC, M) = _concretize(Q.BCs, M)

function _concretize(Q::AbstractArray{T,N}, M) where {T,N}
    return (stencil.(Q, fill(M,size(Q))), affine.(Q))
end

function LinearAlgebra.Array(Q::MultiDimDirectionalBC{T, B, D, N, L}, s::NTuple{N,G}) where {T, B, D,N,L, G<:Int}
    @assert size(Q.BCs) == perpindex(s, D) "The size of the BC array in Q, $(size(Q.BCs)) is incompatible with s, $s"
    blip = zeros(Int64, N)
    blip[D] = 2
    s_pad = s.+ blip # extend s in the right direction
    Q = _concretize(Q.BCs, s[D])
    ē = unit_indices(N)
    QL = zeros(T, prod(s_pad), prod(s))
    Qb = zeros(T, prod(s_pad))
    ranges = Union{typeof(1:10), Int64}[1:s[i] for i in 1:N]
    ranges[D] = ranges[D] .+ 1

    interior = CartesianIndices(Tuple(ranges))
    I1 = CartesianIndex(Tuple(ones(Int64, N)))
    for I in interior
        i = cartesian_to_linear(I, s_pad)
        j = cartesian_to_linear(I-ē[D], s)
        QL[i,j] = one(T)
    end
    ranges[D] = 1
    lower = CartesianIndices((Tuple(ranges)))
    ranges[D] = s_pad[D]
    upper = CartesianIndices((Tuple(ranges)))
    for K in CartesianIndices(upper)
        I = CartesianIndex(Tuple(K)[setdiff(1:N, D)])
        il = cartesian_to_linear(lower[K], s_pad)
        iu = cartesian_to_linear(upper[K], s_pad)
        Qb[il] = Q[2][I][1]
        Qb[iu] = Q[2][I][2]
        for k in 0:s[D]-1
            j = cartesian_to_linear(lower[K] + k*ē[D], s)
            QL[il, j] = Q[1][I][1][k+1]
            QL[iu, j] = Q[1][I][2][k+1]
        end
    end

    return (QL, Qb)
end

"""
This is confusing, but it does work.
"""
function LinearAlgebra.Array(Q::ComposedMultiDimBC{T, B, N,M} , s::NTuple{N,G}) where {T, B, N, M, G<:Int}
    for d in 1:N
        @assert size(Q.BCs[d]) == perpindex(s, d) "The size of the BC array in Q along dimension $d, $(size(Q.BCs[d])) is incompatible with s, $s"
    end
    s_pad = s.+2
    Q = Tuple(_concretize.(Q.BCs, s)) #essentially, finding the first and last rows of the matrix part and affine part for every atomic BC

    QL = zeros(T, prod(s_pad), prod(s))
    Qb = zeros(T, prod(s_pad))

    ranges = Union{typeof(1:10), Int64}[2:s_pad[i]-1 for i in 1:N] #Set up indices corresponding to the interior
    interior = CartesianIndices(Tuple(ranges))

    ē = unit_indices(N) #setup unit indices in each direction
    I1 = CartesianIndex(Tuple(ones(Int64, N))) #set up the ones index
    for I in interior #loop over interior
        i = cartesian_to_linear(I, s_pad) #find the index on the padded side
        j = cartesian_to_linear(I-I1, s)  #find the index on the unpadded side
        QL[i,j] = one(T)  #create a padded identity matrix
    end
    for dim in 1:N #Loop over boundaries
        r_ = deepcopy(ranges)
        r_[dim] = 1
        lower = CartesianIndices((Tuple(r_))) #set up upper and lower indices
        r_[dim] = s_pad[dim]
        upper = CartesianIndices((Tuple(r_)))
        for K in CartesianIndices(upper) #for every element of the boundaries
            I = CartesianIndex(Tuple(K)[setdiff(1:N, dim)]) #convert K to 2D index for indexing the BC arrays
            il = cartesian_to_linear(lower[K], s_pad) #Translate to linear indices
            iu = cartesian_to_linear(upper[K], s_pad) # ditto
            Qb[il] = Q[dim][2][I][1] #store the affine parts in indices corresponding with the lower index boundary
            Qb[iu] = Q[dim][2][I][2] #ditto with upper index
            for k in 1:s[dim] #loop over the direction orthogonal to the boundary
                j = cartesian_to_linear(lower[K] + k*ē[dim]-I1, s) #Find the linear index this element of the boundary stencil should be at on the unpadded side
                QL[il, j] = Q[dim][1][I][1][k]
                QL[iu, j] = Q[dim][1][I][2][k]
            end
        end
    end

    return (QL, Qb)
end

"""
See comments on the `Array` method for this type for an idea of what is going on.
"""
function SparseArrays.SparseMatrixCSC(Q::MultiDimDirectionalBC{T, B, D, N, L}, s::NTuple{N,G}) where {T, B, D,N,L, G<:Int}
    @assert size(Q.BCs) == perpindex(s, D) "The size of the BC array in Q, $(size(Q.BCs)) is incompatible with s, $s"
    blip = zeros(Int64, N)
    blip[D] = 2
    s_pad = s.+ blip # extend s in the right direction
    Q = _concretize(Q.BCs, s[D])
    ē = unit_indices(N)
    QL = spzeros(T, prod(s_pad), prod(s))
    Qb = spzeros(T, prod(s_pad))
    ranges = Union{typeof(1:10), Int64}[1:s[i] for i in 1:N]
    ranges[D] = ranges[D] .+ 1

    interior = CartesianIndices(Tuple(ranges))
    I1 = CartesianIndex(Tuple(ones(Int64, N)))
    for I in interior
        i = cartesian_to_linear(I, s_pad)
        j = cartesian_to_linear(I-ē[D], s)
        QL[i,j] = one(T)
    end
    ranges[D] = 1
    lower = CartesianIndices((Tuple(ranges)))
    ranges[D] = s_pad[D]
    upper = CartesianIndices((Tuple(ranges)))
    for K in CartesianIndices(upper)
        I = CartesianIndex(Tuple(K)[setdiff(1:N, D)])
        il = cartesian_to_linear(lower[K], s_pad)
        iu = cartesian_to_linear(upper[K], s_pad)
        Qb[il] = Q[2][I][1]
        Qb[iu] = Q[2][I][2]
        for k in 0:s[D]-1
            j = cartesian_to_linear(lower[K] + k*ē[D], s)
            QL[il, j] = Q[1][I][1][k+1]
            QL[iu, j] = Q[1][I][2][k+1]
        end
    end

    return (QL, Qb)
end


function SparseArrays.SparseMatrixCSC(Q::ComposedMultiDimBC{T, B, N,M}, s::NTuple{N,G}) where {T, B, N, M, G<:Int}
    for d in 1:N
        @assert size(Q.BCs[d]) == perpindex(s, d) "The size of the BC array in Q along dimension $d, $(size(Q.BCs[d])) is incompatible with s, $s"
    end
    s_pad = s.+2
    Q = Tuple(_concretize.(Q.BCs, s)) #essentially, finding the first and last rows of the matrix part and affine part for every atomic BC

    QL = spzeros(T, prod(s_pad), prod(s))
    Qb = spzeros(T, prod(s_pad))

    ranges = Union{typeof(1:10), Int64}[2:s_pad[i]-1 for i in 1:N] #Set up indices corresponding to the interior
    interior = CartesianIndices(Tuple(ranges))

    ē = unit_indices(N) #setup unit indices in each direction
    I1 = CartesianIndex(Tuple(ones(Int64, N))) #set up the ones index
    for I in interior #loop over interior
        i = cartesian_to_linear(I, s_pad) #find the index on the padded side
        j = cartesian_to_linear(I-I1, s)  #find the index on the unpadded side
        QL[i,j] = one(T)  #create a padded identity matrix
    end
    for dim in 1:N #Loop over boundaries
        r_ = deepcopy(ranges)
        r_[dim] = 1
        lower = CartesianIndices((Tuple(r_))) #set up upper and lower indices
        r_[dim] = s_pad[dim]
        upper = CartesianIndices((Tuple(r_)))
        for K in CartesianIndices(upper) #for every element of the boundaries
            I = CartesianIndex(Tuple(K)[setdiff(1:N, dim)]) #convert K to 2D index for indexing the BC arrays
            il = cartesian_to_linear(lower[K], s_pad) #Translate to linear indices
            iu = cartesian_to_linear(upper[K], s_pad) # ditto
            Qb[il] = Q[dim][2][I][1] #store the affine parts in indices corresponding with the lower index boundary
            Qb[iu] = Q[dim][2][I][2] #ditto with upper index
            for k in 1:s[dim] #loop over the direction orthogonal to the boundary
                j = cartesian_to_linear(lower[K] + k*ē[dim]-I1, s) #Find the linear index this element of the boundary stencil should be at on the unpadded side
                QL[il, j] = Q[dim][1][I][1][k]
                QL[iu, j] = Q[dim][1][I][2][k]
            end
        end
    end

    return (QL, Qb)
end

SparseArrays.sparse(Q::MultiDimDirectionalBC, s) = SparseMatrixCSC(Q, s)
SparseArrays.sparse(Q::ComposedMultiDimBC, s) = SparseMatrixCSC(Q, s)


function BandedMatrices.BandedMatrix(Q:: MultiDimensionalBC, M) where {T, B, D,N,K}
    throw("Banded Matrix cocnretization not yet supported for MultiDimensionalBCs")
end

################################################################################
# Higher-Dimensional DerivativeOperator Concretizations
################################################################################
# Higher-Dimensional Concretizations. The following concretizations return two-dimensional arrays
# which operate on flattened vectors. Mshape is the size of the unflattened array on which A is operating.

function LinearAlgebra.Array(A::DerivativeOperator{T,N}, Mshape) where {T,N}
    # Case where A is not differentiating along the first dimension
    if N != 1
        n = prod(Mshape[1:N-1])
        B = Kron(Array(A), Eye(n))
        if N != length(Mshape)
            n = prod(Mshape[N+1:end])
            B = Kron(Eye(n), B)
        end

    # Case where A is differentiating along the first dimension
    else
        n = prod(Mshape[2:end])
        B = Kron(Eye(n), Array(A))
    end
    return Array(B)
end

function SparseArrays.SparseMatrixCSC(A::DerivativeOperator{T,N}, Mshape) where {T,N}
    # Case where A is not differentiating along the first dimension
    if N != 1
        n = prod(Mshape[1:N-1])
        B = Kron(sparse(A), sparse(I,n,n))
        if N != length(Mshape)
            n = prod(Mshape[N+1:end])
            B = Kron(sparse(I,n,n), B)
        end

    # Case where A is differentiating along the first dimension
    else
        n = prod(Mshape[2:end])
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
        n = prod(Mshape[1:N-1])
        B = Kron(BandedMatrix(A), Eye(n))
        if N != length(Mshape)
            n = prod(Mshape[N+1:end])
            B = Kron(Eye(n), B)
        end

    # Case where A is differentiating along the first dimension
    else
        n = prod(Mshape[2:end])
        B = Kron(BandedMatrix(Eye(n)), BandedMatrix(A))
    end
    return BandedMatrix(B)
end

function BlockBandedMatrices.BandedBlockBandedMatrix(A::DerivativeOperator{T,N}, Mshape) where {T,N}
    # Case where A is not differentiating along the first dimension
    if N != 1
        n = prod(Mshape[1:N-1])
        B = Kron(BandedMatrix(A), Eye(n))
        if N != length(Mshape)
            n = prod(Mshape[N+1:end])
            B = Kron(Eye(n), B)
        end

    # Case where A is differentiating along the first dimension
    else
        n = prod(Mshape[2:end])
        B = Kron(BandedMatrix(Eye(n)), BandedMatrix(A))
    end
    return BandedBlockBandedMatrix(B)
end

################################################################################
# Upwind Operator Concretization
################################################################################

# Array Concretizations
# Uniform grid case
function LinearAlgebra.Array(A::DerivativeOperator{T,N,true}, len::Int=A.len) where {T,N}
    L = zeros(T, len, len+2)
    bpc = A.boundary_point_count
    stl = A.stencil_length
    bstl = A.boundary_stencil_length
    coeff   = A.coefficients
    offside = A.offside

    # downwind stencils at low boundary
    downwind_stencils = A.low_boundary_coefs
    # upwind stencils at upper boundary
    upwind_stencils = A.high_boundary_coefs
    # interior stencils
    stencils = A.stencil_coefs

    for i in 1:bpc
        cur_coeff   = coeff[i]
        if cur_coeff >= 0 && (i+stl <= len+2) && i >= offside
            cur_stencil = stencils
            L[i,i+1-offside:i+stl-offside] = cur_coeff*cur_stencil
        else
            cur_stencil = downwind_stencils[i]
            L[i,1:bstl] = cur_coeff * cur_stencil
        end
    end

    for i in bpc+1 : len-bpc-offside
        cur_coeff   = coeff[i]
        cur_stencil = stencils
        cur_stencil = cur_coeff >= 0 ? cur_stencil : ((-1)^A.derivative_order)*reverse(cur_stencil)
        if cur_coeff >= 0
            L[i,i+1-offside:i+stl-offside] = cur_coeff * cur_stencil
        else
            L[i,i-stl+2+offside:i+1+offside] = cur_coeff * cur_stencil
        end
    end

    for i in 1 : bpc+offside
        cur_coeff   = coeff[len-bpc+i-offside]
        if cur_coeff < 0 && (len+2-stl-bpc+i >= 1) && (i <= bpc+1)
            cur_stencil = stencils
            cur_stencil = ((-1)^A.derivative_order)*reverse(cur_stencil)
            L[len-bpc+i-offside,i+len-stl-bpc+2:i+len-bpc+1] = cur_coeff * cur_stencil
        elseif cur_coeff < 0 && (len+2-stl-bpc+i >= 1) && (i > bpc + 1)
            cur_stencil = upwind_stencils[bpc + offside + 1 - i]
            cur_stencil = ((-1)^A.derivative_order)*reverse(cur_stencil)
            L[len-bpc+i-offside ,len-bstl+3:len+2] = cur_coeff * cur_stencil
        elseif cur_coeff >=0 && i < offside + 1 
            cur_stencil = stencils
            L[len-bpc+i-offside,len-stl+i-offside+3:len+i-offside+2] = cur_coeff * cur_stencil
        else
            cur_stencil = upwind_stencils[i]
            L[len-bpc+i-offside,len-bstl+3:len+2] = cur_coeff * cur_stencil
        end
    end
    return L
end

# Non-uniform grid case
function LinearAlgebra.Array(A::DerivativeOperator{T,N,true,M}, len::Int=A.len) where {T,N,M<:AbstractArray{T}}
    L = zeros(T, len, len+2)
    bpc = A.boundary_point_count
    stl = A.stencil_length
    bstl = A.boundary_stencil_length
    coeff   = A.coefficients
    offside = A.offside

    for i in 1:bpc
        cur_coeff   = coeff[i]
        if cur_coeff >= 0 && offside == 0
            L[i,i+1:i+stl] = cur_coeff * A.low_boundary_coefs[1,i]
        elseif cur_coeff >= 0 && i < offside 
            L[i,1:stl] = cur_coeff * A.low_boundary_coefs[1,i]
        elseif cur_coeff >= 0 && i >= offside
            L[i,i+1-offside:i+stl-offside] = cur_coeff * A.low_boundary_coefs[1,i]
        else
            L[i,1:bstl] = cur_coeff * A.low_boundary_coefs[2,i]
        end
    end

    for i in bpc+1:len-bpc-offside
        cur_coeff   = coeff[i]
        if cur_coeff >= 0
            L[i,i+1-offside:i+stl-offside] = cur_coeff * A.stencil_coefs[1,i-bpc]
        else
            L[i,i-stl+2+offside:i+1+offside] = cur_coeff * A.stencil_coefs[2,i-bpc]
        end
    end

    for i in len-bpc+1-offside:len
        cur_coeff   = coeff[i]
        if cur_coeff < 0 && i <= len - offside
            L[i,i-stl+2+offside:i+1+offside] = cur_coeff * A.high_boundary_coefs[2,i-len+bpc+offside]
        elseif cur_coeff < 0
            L[i,len-stl+3:len+2] = cur_coeff * A.high_boundary_coefs[2,i-len+bpc+offside]
        elseif cur_coeff >=0 && i <= len - bpc
            L[i,i-bstl+2+offside:i+1+offside] = cur_coeff * A.stencil_coefs[1,i-bpc]
        else
            L[i,len-bstl+3:len+2] = cur_coeff * A.high_boundary_coefs[1,i-len+bpc+offside]
        end
    end
    return L
end

# Sparse Concretizations
# Uniform grid case
function SparseArrays.SparseMatrixCSC(A::DerivativeOperator{T,N,true}, len::Int=A.len) where {T,N}
    L = spzeros(T, len, len+2)
    bpc = A.boundary_point_count
    stl = A.stencil_length
    bstl = A.boundary_stencil_length
    coeff   = A.coefficients
    offside = A.offside

    # downwind stencils at low boundary
    downwind_stencils = A.low_boundary_coefs
    # upwind stencils at upper boundary
    upwind_stencils = A.high_boundary_coefs
    # interior stencils
    stencils = A.stencil_coefs

    for i in 1:bpc
        cur_coeff   = coeff[i]
        if cur_coeff >= 0 && (i+stl <= len+2) && i >= offside
            cur_stencil = stencils
            L[i,i+1-offside:i+stl-offside] = cur_coeff*cur_stencil
        else
            cur_stencil = downwind_stencils[i]
            L[i,1:bstl] = cur_coeff * cur_stencil
        end
    end

    for i in bpc+1 : len-bpc-offside
        cur_coeff   = coeff[i]
        cur_stencil = stencils
        cur_stencil = cur_coeff >= 0 ? cur_stencil : ((-1)^A.derivative_order)*reverse(cur_stencil)
        if cur_coeff >= 0
            L[i,i+1-offside:i+stl-offside] = cur_coeff * cur_stencil
        else
            L[i,i-stl+2+offside:i+1+offside] = cur_coeff * cur_stencil
        end
    end

    for i in 1 : bpc+offside
        cur_coeff   = coeff[len-bpc+i-offside]
        if cur_coeff < 0 && (len+2-stl-bpc+i >= 1) && (i <= bpc+1)
            cur_stencil = stencils
            cur_stencil = ((-1)^A.derivative_order)*reverse(cur_stencil)
            L[len-bpc+i-offside,i+len-stl-bpc+2:i+len-bpc+1 ] = cur_coeff * cur_stencil
        elseif cur_coeff < 0 && (len+2-stl-bpc+i >= 1) && (i > bpc + 1)
            cur_stencil = upwind_stencils[bpc + offside + 1 - i]
            cur_stencil = ((-1)^A.derivative_order)*reverse(cur_stencil)
            L[len-bpc+i-offside,len-bstl+3:len+2] = cur_coeff * cur_stencil
        elseif cur_coeff >=0 && i < offside + 1 
            cur_stencil = stencils
            L[len-bpc+i-offside,len-stl+i-offside+3:len+i-offside+2] = cur_coeff * cur_stencil
        else
            cur_stencil = upwind_stencils[i]
            L[len-bpc+i-offside,len-bstl+3:len+2] = cur_coeff * cur_stencil
        end
    end
    return L
end

# Non-uniform grid case
function SparseArrays.SparseMatrixCSC(A::DerivativeOperator{T,N,true,M}, len::Int=A.len) where {T,N,M<:AbstractArray{T}}
    L = spzeros(T, len, len+2)
    bpc = A.boundary_point_count
    stl = A.stencil_length
    bstl = A.boundary_stencil_length
    coeff   = A.coefficients
    offside = A.offside

    for i in 1:bpc
        cur_coeff   = coeff[i]
        if cur_coeff >= 0 && offside == 0
            L[i,i+1:i+stl] = cur_coeff * A.low_boundary_coefs[1,i]
        elseif cur_coeff >= 0 && i < offside 
            L[i,1:stl] = cur_coeff * A.low_boundary_coefs[1,i]
        elseif cur_coeff >= 0 && i >= offside
            L[i,i+1-offside:i+stl-offside] = cur_coeff * A.low_boundary_coefs[1,i]
        else
            L[i,1:bstl] = cur_coeff * A.low_boundary_coefs[2,i]
        end
    end

    for i in bpc+1:len-bpc-offside
        cur_coeff   = coeff[i]
        if cur_coeff >= 0
            L[i,i+1-offside:i+stl-offside] = cur_coeff * A.stencil_coefs[1,i-bpc]
        else
            L[i,i-stl+2+offside:i+1+offside] = cur_coeff * A.stencil_coefs[2,i-bpc]
        end
    end

    for i in len-bpc+1-offside:len
        cur_coeff   = coeff[i]
        if cur_coeff < 0 && i <= len - offside
            L[i,i-stl+2+offside:i+1+offside] = cur_coeff * A.high_boundary_coefs[2,i-len+bpc+offside]
        elseif cur_coeff < 0
            L[i,len-stl+3:len+2] = cur_coeff * A.high_boundary_coefs[2,i-len+bpc+offside]
        elseif cur_coeff >=0 && i <= len - bpc
            L[i,i-bstl+2+offside:i+1+offside] = cur_coeff * A.stencil_coefs[1,i-bpc]
        else
            L[i,len-bstl+3:len+2] = cur_coeff * A.high_boundary_coefs[1,i-len+bpc+offside]
        end
    end
    return L
end

# Banded Concretizations
# Uniform grid case
function BandedMatrices.BandedMatrix(A::DerivativeOperator{T,N,true}, len::Int=A.len) where {T,N}
    bpc = A.boundary_point_count
    stl = A.stencil_length
    bstl = A.boundary_stencil_length
    coeff   = A.coefficients
    offside = A.offside

    L = BandedMatrix{T}(Zeros(len, len+2), (stl-2, stl))

    # downwind stencils at low boundary
    downwind_stencils = A.low_boundary_coefs
    # upwind stencils at upper boundary
    upwind_stencils = A.high_boundary_coefs
    # interior stencils
    stencils = A.stencil_coefs

    for i in 1:bpc
        cur_coeff   = coeff[i]
        if cur_coeff >= 0 && (i+stl <= len+2) && i >= offside
            cur_stencil = stencils
            L[i,i+1-offside:i+stl-offside] = cur_coeff*cur_stencil
        else
            cur_stencil = downwind_stencils[i]
            L[i,1:bstl] = cur_coeff * cur_stencil
        end
    end

    for i in bpc+1 : len-bpc-offside
        cur_coeff   = coeff[i]
        cur_stencil = stencils
        cur_stencil = cur_coeff >= 0 ? cur_stencil : ((-1)^A.derivative_order)*reverse(cur_stencil)
        if cur_coeff >= 0
            L[i,i+1-offside:i+stl-offside] = cur_coeff * cur_stencil
        else
            L[i,i-stl+2+offside:i+1+offside] = cur_coeff * cur_stencil
        end
    end

    for i in 1 : bpc+offside
        cur_coeff   = coeff[len-bpc+i-offside]
        if cur_coeff < 0 && (len+2-stl-bpc+i >= 1) && (i <= bpc+1)
            cur_stencil = stencils
            cur_stencil = ((-1)^A.derivative_order)*reverse(cur_stencil)
            L[len-bpc+i-offside,i+len-stl-bpc+2:i+len-bpc+1] = cur_coeff * cur_stencil
        elseif cur_coeff < 0 && (len+2-stl-bpc+i >= 1) && (i > bpc + 1)
            cur_stencil = upwind_stencils[bpc + offside + 1 - i]
            cur_stencil = ((-1)^A.derivative_order)*reverse(cur_stencil)
            L[len-bpc+i-offside,len-bstl+3:len+2] = cur_coeff * cur_stencil
        elseif cur_coeff >=0 && i < offside + 1 
            cur_stencil = stencils
            L[len-bpc+i-offside,len-stl+i-offside+3:len+i-offside+2] = cur_coeff * cur_stencil
        else
            cur_stencil = upwind_stencils[i]
            L[len-bpc+i-offside,len-bstl+3:len+2] = cur_coeff * cur_stencil
        end
    end
    return L
end


# Non-uniform grid case
function BandedMatrices.BandedMatrix(A::DerivativeOperator{T,N,true,M}, len::Int=A.len) where {T,N,M<:AbstractArray{T}}
    bpc = A.boundary_point_count
    stl = A.stencil_length
    bstl = A.boundary_stencil_length
    coeff   = A.coefficients
    offside = A.offside
    L = BandedMatrix{T}(Zeros(len, len+2), (stl-2, stl))

    for i in 1:bpc
        cur_coeff   = coeff[i]
        if cur_coeff >= 0 && offside == 0
            L[i,i+1:i+stl] = cur_coeff * A.low_boundary_coefs[1,i]
        elseif cur_coeff >= 0 && i < offside 
            L[i,1:stl] = cur_coeff * A.low_boundary_coefs[1,i]
        elseif cur_coeff >= 0 && i >= offside
            L[i,i+1-offside:i+stl-offside] = cur_coeff * A.low_boundary_coefs[1,i]
        else
            L[i,1:bstl] = cur_coeff * A.low_boundary_coefs[2,i]
        end
    end

    for i in bpc+1:len-bpc-offside
        cur_coeff   = coeff[i]
        if cur_coeff >= 0
            L[i,i+1-offside:i+stl-offside] = cur_coeff * A.stencil_coefs[1,i-bpc]
        else
            L[i,i-stl+2+offside:i+1+offside] = cur_coeff * A.stencil_coefs[2,i-bpc]
        end
    end

    for i in len-bpc+1-offside:len
        cur_coeff   = coeff[i]
        if cur_coeff < 0 && i <= len - offside
            L[i,i-stl+2+offside:i+1+offside] = cur_coeff * A.high_boundary_coefs[2,i-len+bpc+offside]
        elseif cur_coeff < 0
            L[i,len-stl+3:len+2] = cur_coeff * A.high_boundary_coefs[2,i-len+bpc+offside]
        elseif cur_coeff >=0 && i <= len - bpc
            L[i,i-bstl+2+offside:i+1+offside] = cur_coeff * A.stencil_coefs[1,i-bpc]
        else
            L[i,len-bstl+3:len+2] = cur_coeff * A.high_boundary_coefs[1,i-len+bpc+offside]
        end
    end
    return L
end

# GhostDerivativeOperator Concretizations
################################################################################
function LinearAlgebra.Array(A::GhostDerivativeOperator{T, E, F},N::Int=A.L.len) where {T,E,F}
    Q_l, Q_b = Array(A.Q,N)
    return (Array(A.L,N)*Q_l, Array(A.L,N)*Q_b)
end

function LinearAlgebra.Array(A::GhostDerivativeOperator{T, E, F}, s::NTuple{N,I}) where {T,E,F,N,I<:Int}
    Q_l, Q_b = Array(A.Q,s)
    return (Array(A.L,s)*Q_l, Array(A.L,s)*Q_b)
end

function BandedMatrices.BandedMatrix(A::GhostDerivativeOperator{T, E, F},N::Int=A.L.len) where {T,E,F}
    Q_l, Q_b = BandedMatrix(A.Q,N)
    return (BandedMatrix(A.L,N)*Q_l, BandedMatrix(A.L,N)*Q_b)
end

function BandedMatrices.BandedMatrix(A::GhostDerivativeOperator{T, E, F}, s::NTuple{N,I}) where {T,E,F, N, I<:Int}
    Q_l,Q_b = BandedMatrix(A.Q,s)
    return (BandedMatrix(A.L,s)*Q_l, BandedMatrix(A.L,s)*Q_b)
end

function SparseArrays.SparseMatrixCSC(A::GhostDerivativeOperator{T, E, F},N::Int=A.L.len) where {T,E,F}
    Q_l, Q_b = SparseMatrixCSC(A.Q,N)
    return (SparseMatrixCSC(A.L,N)*Q_l, SparseMatrixCSC(A.L,N)*Q_b)
end

function SparseArrays.SparseMatrixCSC(A::GhostDerivativeOperator{T, E, F}, s::NTuple{N,I}) where {T,E,F,N,I<:Int}
    Q_l, Q_b = SparseMatrixCSC(A.Q,s)
    return (SparseMatrixCSC(A.L,s)*Q_l, SparseMatrixCSC(A.L,s)*Q_b)
end


function SparseArrays.sparse(A::GhostDerivativeOperator{T, E, F}, N::Int=A.L.len) where {T,E,F}
    return SparseMatrixCSC(A,N)
end


function SparseArrays.sparse(A::GhostDerivativeOperator{T, E, F}, s::NTuple{N,I}) where {T,E,F, N,I}
    return SparseMatrixCSC(A,s)
end

################################################################################
# Composite Operator Concretizations
################################################################################
Array(L::DiffEqScaledOperator, s) = L.coeff * Array(L.op, s)
Array(L::DiffEqOperatorCombination, s) = sum(Array.(L.ops, fill(s, length(L.ops))))
Array(L::DiffEqOperatorComposition, s) = prod(Array.(reverse(L.ops), fill(s, length(L.ops))))
