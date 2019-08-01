
abstract type MultiDimensionalBC{T, N} <: AbstractBC{T} end

"""
slicemul is the only limitation on the BCs here being used up to arbitrary dimension, an N dimensional implementation is needed.
"""
@inline function slicemul(A::Array{B,1}, u::AbstractArray{T, 2}, dim::Integer) where {T, B<:AtomicBC{T}}
    s = size(u)
    if dim == 1
        lower = zeros(T, s[2])
        upper = deepcopy(lower)
        for i in 1:(s[2])
            tmp = A[i]*u[:,i]
            lower[i], upper[i] = (tmp.l, tmp.r)
        end
    elseif dim == 2
        lower = zeros(T, s[1])
        upper = deepcopy(lower)
        for i in 1:(s[1])
            tmp = A[i]*u[i,:]
            lower[i], upper[i] = (tmp.l, tmp.r)
        end
    elseif dim == 3
        throw("The 3 dimensional Method should be being called, not this one. Check dispatch.")
    else
        throw("Dim greater than 3 not supported!")
    end
    return lower, upper
end


@inline function slicemul(A::Array{B,2}, u::AbstractArray{T, 3}, dim::Integer) where {T, B<:AtomicBC{T}}
    s = size(u)
    if dim == 1
        lower = zeros(T, s[2], s[3])
        upper = deepcopy(lower)
        for j in 1:s[3]
            for i in 1:s[2]
                tmp = A[i,j]*u[:,i,j]
                lower[i,j], upper[i,j] = (tmp.l, tmp.r)
            end
        end
    elseif dim == 2
        lower = zeros(T, s[1], s[3])
        upper = deepcopy(lower)
        for j in 1:s[3]
            for i in 1:s[1]
                tmp = A[i,j]*u[i,:,j]
                lower[i,j], upper[i,j] = (tmp.l, tmp.r)
            end
        end
    elseif dim == 3
        lower = zeros(T, s[1], s[2])
        upper = deepcopy(lower)
        for j in 1:s[2]
            for i in 1:s[1]
                tmp = A[i,j]*u[i,j,:]
                lower[i,j], upper[i,j] = (tmp.l, tmp.r)
            end
        end
    else
        throw("Dim greater than 3 not supported!")
    end
    return lower, upper
end


struct MultiDimDirectionalBC{T<:Number, B<:AtomicBC{T}, D, N, M} <: MultiDimensionalBC{T, N}
    BCs::Array{B,M} #dimension M=N-1 array of BCs to extend dimension D
end

struct ComposedMultiDimBC{T, B<:AtomicBC{T}, N,M} <: MultiDimensionalBC{T, N}
    BCs::Vector{Array{B, M}}
end

"""
A multiple dimensional BC, supporting arbitrary BCs at each boundary point.
To construct an arbitrary BC, pass an Array of BCs with dimension one less than that of your domain u - denoted N,
with a size of size(u)[setdiff(1:N, dim)], where dim is the dimension orthogonal to the boundary that you want to extend.

It is also possible to call
    Q_dim = MultiDimBC(YourBC, size(u), dim)
to use YourBC for the whole boundary orthogonal to that dimension.

Further, it is possible to call
Qx, Qy, Qz... = MultiDimBC(YourBC, size(u))
to use YourBC for the whole boundary for all dimensions. Valid for any number of dimensions greater than 1.
However this is only valid for Robin/General type BCs (including neummann/dirichlet) when the grid steps are equal in each dimension - including uniform grid case.

In the case where you want to extend the same Robin/GeneralBC to the whole boundary with a non unifrom grid, please use
    Qx, Qy, Qz... = RobinBC(l, r, (dx::Vector, dy::Vector, dz::Vector ...), approximation_order, size(u))
or
    Qx, Qy, Qz... = GeneralBC(αl, αr, (dx::Vector, dy::Vector, dz::Vector ...), approximation_order, size(u))

There are also constructors for NeumannBC, DirichletBC, Neumann0BC and Dirichlet0BC. Simply replace dx with the tuple as above, and append size(u) to the argument signature.
The order is a required argument in this case.

where dx, dy, and dz are vectors of grid steps.
"""
MultiDimBC(BC::Array{B,N}, dim::Integer) where {N, B} = MultiDimDirectionalBC{gettype(BC[1]), B, dim, N+1, N}(BC)
#s should be size of the domain
MultiDimBC(BC::B, s, dim::Integer) where  {B} = MultiDimDirectionalBC{gettype(BC), B, dim, length(s), length(s)-1}(fill(BC, s[setdiff(1:length(s), dim)]))

#Extra constructor to make a set of BC operators that extend an atomic BC Operator to the whole domain
#Only valid in the uniform grid case!
MultiDimBC(BC::B, s) where {B} = Tuple([MultiDimDirectionalBC{gettype(BC), B, dim, length(s), length(s)-1}(fill(BC, s[setdiff(1:length(s), dim)])) for dim in 1:length(s)])

# Additional constructors for cases when the BC is the same for all boundarties

PeriodicBC{T}(s) where T = MultiDimBC(PeriodicBC{T}(), s)

NeumannBC(α::AbstractVector{T}, dxyz, order, s) where T = RobinBC([zero(T), one(T), α[1]], [zero(T), one(T), α[2]], dxyz, order, s)
DirichletBC(αl::T, αr::T, s) where T = RobinBC([one(T), zero(T), αl], [one(T), zero(T), αr], [ones(T, si) for si in s], 2.0, s)

Dirichlet0BC(T::Type, s) = DirichletBC(zero(T), zero(T), s)
Neumann0BC(T::Type, dxyz, order, s) = NeumannBC([zero(T), zero(T)], dxyz, order, s)

RobinBC(l::AbstractVector{T}, r::AbstractVector{T}, dxyz, order, s) where {T} = Tuple([MultiDimDirectionalBC{T, RobinBC{T}, dim, length(s), length(s)-1}(fill(RobinBC(l, r, dxyz[dim], order), s[setdiff(1:length(s), dim)])) for dim in 1:length(s)])
GeneralBC(αl::AbstractVector{T}, αr::AbstractVector{T}, dxyz, order, s) where {T} = Tuple([MultiDimDirectionalBC{T, GeneralBC{T}, dim, length(s), length(s)-1}(fill(GeneralBC(αl, αr, dxyz[dim], order), s[setdiff(1:length(s), dim)])) for dim in 1:length(s)])


perpsize(A::AbstractArray{T,N}, dim::Integer) where {T,N} = size(A)[setdiff(1:N, dim)] #the size of A perpendicular to dim

# Constructors for Bridge BC to make it easier to join domains together. See docs on BrigeBC in BC_operators.jl for info on usage
function BridgeBC(u1::AbstractArray{T,2}, dim1::Int, hilo1::String, bc1::MultiDimDirectionalBC, u2::AbstractArray{T,2}, dim2::Int, hilo2::String, bc2::MultiDimDirectionalBC) where {T}
    @assert 1 ≤ dim1 ≤ 2 "dim1 must be 1≤dim1≤2, got dim1 = $dim1"
    @assert 1 ≤ dim1 ≤ 2 "dim2 must be 1≤dim1≤2, got dim1 = $dim1"
    s1 = perpsize(u1, dim1) #
    s2 = perpsize(u2, dim2)
    @assert s1 == s2 "Arrays must be same size along boundary to be joined, got boundary sizes u1 = $s1, u2 = $s2"
    if hilo1 == "low"
        view1 = selectdim(u1, dim1, 1)
        if hilo2 == "low"
            BC1 = Array{MixedBC{T, BridgeBC{T, 2, eltype(s1)}, getboundarytype(bc1)}}(undef, s1...)
            BC2 = Array{MixedBC{T, BridgeBC{T, 2, eltype(s2)}, getboundarytype(bc2)}}(undef, s2...)
            view2 = selectdim(u2, dim2, 1)
            for i in 1:s1[1]
                BC1[i] = MixedBC(BridgeBC{T, 2, eltype(s1)}(zeros(T, 1), view(view2, i), zeros(T, 1), view(view2, i)), bc1.BCs[i])
                BC2[i] = MixedBC(BridgeBC{T, 2, eltype(s1)}(zeros(T, 1), view(view1, i), zeros(T, 1), view(view1, i)), bc2.BCs[i])
            end
        elseif hilo2 == "high"
            BC1 = Array{MixedBC{T, BridgeBC{T, 2, eltype(s1)}, getboundarytype(bc1)}}(undef, s1...)
            BC2 = Array{MixedBC{T, getboundarytype(bc2), BridgeBC{T, 2, eltype(s2)}}}(undef, s2...)
            view2 = selectdim(u2, dim2, size(u2)[dim2])
            for i in 1:s1[1]
                BC1[i] = MixedBC(BridgeBC{T, 2, eltype(s1)}(zeros(T, 1), view(view2, i), zeros(T, 1), view(view2, i)), bc1.BCs[i])
                BC2[i] = MixedBC(bc2.BCs[i], BridgeBC{T, 2, eltype(s1)}(zeros(T, 1), view(view1, i), zeros(T, 1), view(view1, i)))
            end
        else
            throw("hilo2 not recognized, please use \"high\" to connect u1 to u2 along the upper index of dim2 of u2 or \"low\" to connect along the lower index end")
        end
    elseif hilo1 == "high"
        view1 = selectdim(u1, dim1, size(u1)[dim1])
        if hilo2 == "low"
            BC1 = Array{MixedBC{T, getboundarytype(bc1), BridgeBC{T, 2, eltype(s1)}}}(undef, s1...)
            BC2 = Array{MixedBC{T, BridgeBC{T, 2, eltype(s2)}, getboundarytype(bc2)}}(undef, s2...)
            view2 = selectdim(u2, dim2, 1)
            for i in 1:s1[1]
                BC1[i] = MixedBC(bc1.BCs[i], BridgeBC{T, 2, eltype(s1)}(zeros(T, 1), view(view2, i), zeros(T, 1), view(view2, i)))
                BC2[i] = MixedBC(BridgeBC{T, 2, eltype(s1)}(zeros(T, 1), view(view1, i), zeros(T, 1), view(view1, i)), bc2.BCs[i])
            end
        elseif hilo2 == "high"
            BC1 = Array{MixedBC{T, getboundarytype(bc1), BridgeBC{T, 2, eltype(s1)}}}(undef, s1...)
            BC2 = Array{MixedBC{T, getboundarytype(bc2), BridgeBC{T, 2, eltype(s2)}}}(undef, s2...)
            view2 = selectdim(u2, dim2, size(u2)[dim2])
            for i in 1:s1[1]
                BC1[i] = MixedBC(bc1.BCs[i], BridgeBC{T, 2, eltype(s1)}(zeros(T, 1), view(view2, i), zeros(T, 1), view(view2, i)))
                BC2[i] = MixedBC(bc2.BCs[i], BridgeBC{T, 2, eltype(s1)}(zeros(T, 1), view(view1, i), zeros(T, 1), view(view1, i)))
            end
        else
            throw("hilo2 not recognized, please use \"high\" to connect u1 to u2 along the upper index of dim2 of u2 or \"low\" to connect along the lower index end")
        end
    else
        throw("hilo1 not recognized, please use \"high\" to connect u1 to u2 along the upper index of dim1 of u1 or \"low\" to connect along the lower index end")
    end
    return (MultiDimBC(BC1, dim1), MultiDimBC(BC2, dim2))
end

function BridgeBC(u1::AbstractArray{T,3}, dim1::Int, hilo1::String, bc1::MultiDimDirectionalBC, u2::AbstractArray{T,3}, dim2::Int, hilo2::String, bc2::MultiDimDirectionalBC) where {T}
    @assert 1 ≤ dim1 ≤ 3 "dim1 must be 1≤dim1≤3, got dim1 = $dim1"
    @assert 1 ≤ dim1 ≤ 3 "dim2 must be 1≤dim1≤3, got dim1 = $dim1"
    s1 = perpsize(u1, dim1) #
    s2 = perpsize(u2, dim2)
    @assert s1 == s2 "Arrays must be same size along boundary to be joined, got boundary sizes u1 = $s1, u2 = $s2"
    if hilo1 == "low"
        view1 = selectdim(u1, dim1, 1)
        if hilo2 == "low"
            BC1 = Array{MixedBC{T, BridgeBC{T, 3, eltype(s1)}, getboundarytype(bc1)}}(undef, s1...)
            BC2 = Array{MixedBC{T, BridgeBC{T, 3, eltype(s2)}, getboundarytype(bc2)}}(undef, s2...)
            view2 = selectdim(u2, dim2, 1)
            for j in 1:s1[2], i in 1:s1[1]
                BC1[i, j] = MixedBC(BridgeBC{T, N, eltype(s1)}(zeros(T, 1), view(view2, i, j), zeros(T, 1), view(view2, i, j)), bc1.BCs[i,j])
                BC2[i, j] = MixedBC(BridgeBC{T, N, eltype(s1)}(zeros(T, 1), view(view1, i, j), zeros(T, 1), view(view1, i, j)), bc2.BCs[i,j])
            end
        elseif hilo2 == "high"
            BC1 = Array{MixedBC{T, BridgeBC{T, 3, eltype(s1)}, getboundarytype(bc1)}}(undef, s1...)
            BC2 = Array{MixedBC{T, getboundarytype(bc2), BridgeBC{T, 3, eltype(s2)}}}(undef, s2...)
            view2 = selectdim(u2, dim2, size(u2)[dim2])
            for j in 1:s1[2], i in 1:s1[1]
                BC1[i, j] = MixedBC(BridgeBC{T, N, eltype(s1)}(zeros(T, 1), view(view2, i, j), zeros(T, 1), view(view2, i, j)), bc1.BCs[i,j])
                BC2[i, j] = MixedBC(bc2.BCs[i,j], BridgeBC{T, N, eltype(s1)}(zeros(T, 1), view(view1, i, j), zeros(T, 1), view(view1, i, j)))
            end
        else
            throw("hilo2 not recognized, please use \"high\" to connect u1 to u2 along the upper index of dim2 of u2 or \"low\" to connect along the lower index end")
        end
    elseif hilo1 == "high"
        view1 = selectdim(u1, dim1, size(u1)[dim1])
        if hilo2 == "low"
            BC1 = Array{MixedBC{T, getboundarytype(bc1), BridgeBC{T, 3, eltype(s1)}}}(undef, s1...)
            BC2 = Array{MixedBC{T, BridgeBC{T, 3, eltype(s2)}, getboundarytype(bc2)}}(undef, s2...)
            view2 = selectdim(u2, dim2, 1)
            view2 = selectdim(u2, dim2, 1)
            for j in 1:s1[2], i in 1:s1[1]
                BC1[i, j] = MixedBC(bc1.BCs[i,j], BridgeBC{T, N, eltype(s1)}(zeros(T, 1), view(view2, i, j), zeros(T, 1), view(view2, i, j)))
                BC2[i, j] = MixedBC(BridgeBC{T, N, eltype(s1)}(zeros(T, 1), view(view1, i, j), zeros(T, 1), view(view1, i, j)), bc2.BCs[i,j])
            end
        elseif hilo2 == "high"
            BC1 = Array{MixedBC{T, getboundarytype(bc1), BridgeBC{T, 3, eltype(s1)}}}(undef, s1...)
            BC2 = Array{MixedBC{T, getboundarytype(bc2), BridgeBC{T, 3, eltype(s2)}}}(undef, s2...)
            view2 = selectdim(u2, dim2, size(u2)[dim2])
            for j in 1:s1[2], i in 1:s1[1]
                BC1[i, j] = MixedBC(bc1.BCs[i,j], BridgeBC{T, N, eltype(s1)}(zeros(T, 1), view(view2, i, j), zeros(T, 1), view(view2, i, j)))
                BC2[i, j] = MixedBC(bc2.BCs[i,j], BridgeBC{T, N, eltype(s1)}(zeros(T, 1), view(view1, i, j), zeros(T, 1), view(view1, i, j)))
            end
        else
            throw("hilo2 not recognized, please use \"high\" to connect u1 to u2 along the upper index of dim2 of u2 or \"low\" to connect along the lower index end")
        end
    else
        throw("hilo1 not recognized, please use \"high\" to connect u1 to u2 along the upper index of dim1 of u1 or \"low\" to connect along the lower index end")
    end
    return (MultiDimBC(BC1, dim1), MultiDimBC(BC2, dim2))
end
"""
Q = compose(BCs...)

-------------------------------------------------------------------------------------

Example:
Q = compose(Qx, Qy, Qz) # 3D domain
Q = compose(Qx, Qy) # 2D Domain

Creates a ComposedMultiDimBC operator, Q, that extends every boundary when applied to a `u` with compatible size and number of dimensions.

Qx Qy and Qz can be passed in any order, as long as there is exactly one BC operator that extends each dimension.
"""
function compose(BCs...)
    T = gettype(BCs[1])
    N = ndims(BCs[1])
    Ds = getaxis.(BCs)
    (length(BCs) == N) || throw("There must be enough BCs to cover every dimension - check that the number of MultiDimBCs == N")
    for D in Ds
        length(setdiff(Ds, D)) == (N-1) || throw("There are multiple boundary conditions that extend along $D - make sure every dimension has a unique extension")
    end
    BCs = BCs[sortperm([Ds...])]

    ComposedMultiDimBC{T, Union{eltype.(BCs)...}, N,N-1}([condition.BC for condition in BCs])
end

"""
Qx, Qy,... = decompose(Q::ComposedMultiDimBC{T,N,M})

-------------------------------------------------------------------------------------

Decomposes a ComposedMultiDimBC in to components that extend along each dimension individually
"""
decompose(Q::ComposedMultiDimBC) = Tuple([MultiDimBC(Q.BC[i], i) for i in 1:ndims(Q)])

getaxis(Q::MultiDimDirectionalBC{T, B, D, N, K}) where {T, B, D, N, K} = D
getboundarytype(Q::MultiDimDirectionalBC{T, B, D, N, K}) where {T, B, D, N, K} = B

Base.ndims(Q::MultiDimensionalBC{T,N}) where {T,N} = N

function Base.:*(Q::MultiDimDirectionalBC{T, B, D, N, K}, u::AbstractArray{T, N}) where {T, B, D, N, K}
    lower, upper = slicemul(Q.BCs, u, D)
    return BoundaryPaddedArray{T, D, N, K, typeof(u), typeof(lower)}(lower, upper, u)
end

function Base.:*(Q::ComposedMultiDimBC{T, B, N, K}, u::AbstractArray{T, N}) where {T, B, N, K}
    out = slicemul.(Q.BCs, fill(u, N), 1:N)
    return ComposedBoundaryPaddedArray{T, N, K, typeof(u), typeof(out[1][1])}([A[1] for A in out], [A[2] for A in out], u)
end
