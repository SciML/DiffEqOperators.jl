abstract type AbstractBC{T} <: AbstractDiffEqLinearOperator{T} end


abstract type AtomicBC{T} <: AbstractBC{T} end

abstract type MultiDimensionalBC{T, N} <: AbstractBC{T} end

"""
Robin, General, and in general Neumann, Dirichlet and Bridge BCs are all affine opeartors, meaning that they take the form Q*x = Qa*x + Qb.
"""
abstract type AffineBC{T} <: AtomicBC{T} end

"""
q = PeriodicBC{T}()

Qx, Qy, ... = PeriodicBC{T}(size(u)) #When all dimensions are to be extended with a periodic boundary condition.

-------------------------------------------------------------------------------------
Creates a periodic boundary condition, where the lower index end of some u is extended with the upper index end and vice versa.
It is not reccomended to concretize this BC type in to a BandedMatrix, since the vast majority of bands will be all 0s. SpatseMatrix concretization is reccomended.
"""
struct PeriodicBC{T} <: AtomicBC{T}
end

"""
  q = RobinBC(left_coefficients, right_coefficients, dx::T, approximation_order) where T # When this BC extends a dimension with a uniform step size

  q = RobinBC(left_coefficients, right_coefficients, dx::Vector{T}, approximation_order) where T # When this BC extends a dimension with a non uniform step size. dx should be the vector of step sizes for the whole dimension

-------------------------------------------------------------------------------------

  The variables in l are [αl, βl, γl], and correspond to a BC of the form αl*u(0) + βl*u'(0) = γl imposed on the lower index boundary.
  The variables in r are [αl, βl, γl], and correspond to an analagous boundary on the higher index end.
  Implements a robin boundary condition operator Q that acts on a vector to give an extended vector as a result
  Referring to (https://github.com/JuliaDiffEq/DiffEqOperators.jl/files/3267835/ghost_node.pdf)
  Write vector b̄₁ as a vertical concatanation with b0 and the rest of the elements of b̄ ₁, denoted b̄`₁, the same with ū into u0 and ū`. b̄`₁ = b̄`_2 = fill(β/Δx, length(stencil)-1)
  Pull out the product of u0 and b0 from the dot product. The stencil used to approximate u` is denoted s. b0 = α+(β/Δx)*s[1]
  Rearrange terms to find a general formula for u0:= -b̄`₁̇⋅ū`/b0 + γ/b0, which is dependent on ū` the robin coefficients and Δx.
  The non identity part of Qa is qa:= -b`₁/b0 = -β.*s[2:end]/(α+β*s[1]/Δx). The constant part is Qb = γ/(α+β*s[1]/Δx)
  do the same at the other boundary (amounts to a flip of s[2:end], with the other set of boundary coeffs)
"""
struct RobinBC{T} <: AffineBC{T}
    a_l::Vector{T}
    b_l::T
    a_r::Vector{T}
    b_r::T
    function RobinBC(l::AbstractVector{T}, r::AbstractVector{T}, dx::T, order = 1) where {T}
        αl, βl, γl = l
        αr, βr, γr = r

        s = calculate_weights(1, one(T), Array(one(T):convert(T,order+1))) #generate derivative coefficients about the boundary of required approximation order

        a_l = -s[2:end]./(αl*dx/βl + s[1])
        a_r = s[end:-1:2]./(αr*dx/βr - s[1]) # for other boundary stencil is flippedlr with *opposite sign*

        b_l = γl/(αl+βl*s[1]/dx)
        b_r = γr/(αr-βr*s[1]/dx)

        return new{T}(a_l, b_l, a_r, b_r)
    end
    function RobinBC(l::AbstractVector{T}, r::AbstractVector{T}, dx::AbstractVector{T}, order = 1) where {T}
        αl, βl, γl = l
        αr, βr, γr = r

        s_index = Array(one(T):convert(T,order+1))
        dx_l, dx_r = dx[1:length(s_index)], dx[(end-length(s_index)+1):end]

        s = calculate_weights(1, one(T), s_index) #generate derivative coefficients about the boundary of required approximation order
        denom_l = αl+βl*s[1]/dx_l[1]
        denom_r = αr-βr*s[1]/dx_r[end]

        a_l = -βl.*s[2:end]./(denom_l*dx_l[2:end])
        a_r = βr.*s[end:-1:2]./(denom_r*dx_r[1:(end-1)]) # for other boundary stencil is flippedlr with *opposite sign*

        b_l = γl/denom_l
        b_r = γr/denom_r

        return new{T}(a_l, b_l, a_r, b_r)
    end
end



"""
q = GeneralBC(α_leftboundary, α_rightboundary, dx::T, approximation_order)

-------------------------------------------------------------------------------------

Implements a generalization of the Robin boundary condition, where α is a vector of coefficients.
Represents a condition of the form α[1] + α[2]u[0] + α[3]u'[0] + α[4]u''[0]+... = 0
Implemented in a similar way to the RobinBC (see above).
This time there are multiple stencils for multiple derivative orders - these can be written as a matrix S.
All components that multiply u(0) are factored out, turns out to only involve the first colum of S, s̄0. The rest of S is denoted S`. the coeff of u(0) is s̄0⋅ᾱ[3:end] + α[2].
the remaining components turn out to be ᾱ[3:end]⋅(S`ū`) or equivalantly (transpose(ᾱ[3:end])*S`)⋅ū`. Rearranging, a stencil q_a to be dotted with ū` upon extension can readily be found, along with a constant component q_b
"""
struct GeneralBC{T} <:AffineBC{T}
    a_l::Vector{T}
    b_l::T
    a_r::Vector{T}
    b_r::T
    function GeneralBC(αl::AbstractVector{T}, αr::AbstractVector{T}, dx::T, order = 1) where {T}
        nl = length(αl)
        nr = length(αr)
        S_l = zeros(T, (nl-2, order+nl-2))
        S_r = zeros(T, (nr-2, order+nr-2))

        for i in 1:(nl-2)
            S_l[i,:] = [transpose(calculate_weights(i, one(T), Array(one(T):convert(T, order+i)))) transpose(zeros(T, Int(nl-2-i)))]./(dx^i) #am unsure if the length of the dummy_x is correct here
        end

        for i in 1:(nr-2)
            S_r[i,:] = [transpose(calculate_weights(i, convert(T, order+i), Array(one(T):convert(T, order+i)))) transpose(zeros(T, Int(nr-2-i)))]./(dx^i)
        end
        s0_l = S_l[:,1] ; Sl = S_l[:,2:end]
        s0_r = S_r[:,end] ; Sr = S_r[:,(end-1):-1:1]

        denoml = αl[2] .+ αl[3:end] ⋅ s0_l
        denomr = αr[2] .+ αr[3:end] ⋅ s0_r

        a_l = -transpose(transpose(αl[3:end]) * Sl) ./denoml
        a_r = -transpose(transpose(αr[3:end]) * Sr) ./denomr

        b_l = -αl[1]/denoml
        b_r = -αr[1]/denomr
        new{T}(a_l,b_l,reverse!(a_r),b_r)
    end

    function GeneralBC(αl::AbstractVector{T}, αr::AbstractVector{T}, dx::AbstractVector{T}, order = 1) where {T}

        nl = length(αl)
        nr = length(αr)
        dx_l, dx_r = (dx[1:(order+nl-2)], reverse(dx[(end-order-nr+3):end]))
        S_l = zeros(T, (nl-2, order+nl-2))
        S_r = zeros(T, (nr-2, order+nr-2))

        for i in 1:(nl-2)
            S_l[i,:] = [transpose(calculate_weights(i, one(T), Array(one(T):convert(T, order+i)))) transpose(zeros(T, Int(nl-2-i)))]./(dx_l.^i)
        end

        for i in 1:(nr-2)
            S_r[i,:] = [transpose(calculate_weights(i, convert(T, order+i), Array(one(T):convert(T, order+i)))) transpose(zeros(T, Int(nr-2-i)))]./(dx_r.^i)
        end
        s0_l = S_l[:,1] ; Sl = S_l[:,2:end]
        s0_r = S_r[:,end] ; Sr = S_r[:,(end-1):-1:1]

        denoml = αl[2] .+ αl[3:end] ⋅ s0_l
        denomr = αr[2] .+ αr[3:end] ⋅ s0_r

        a_l = -transpose(transpose(αl[3:end]) * Sl) ./denoml
        a_r = -transpose(transpose(αr[3:end]) * Sr) ./denomr

        b_l = -αl[1]/denoml
        b_r = -αr[1]/denomr
        new{T}(a_l,b_l,reverse!(a_r),b_r)
    end
end



"""
Quick and dirty way to allow mixed boundary types on each end of an array - may be cleaner and more versatile to split up left and right boundaries going forward
MixedBC(lowerBC, upperBC) is the interface.
"""

struct MixedBC{T, R, S} <: AtomicBC{T}
    lower::R
    upper::S
    function MixedBC(Qlower,Qupper)
        @assert Qlower isa AtomicBC
        @assert Qupper isa AtomicBC
         new{Union{gettype(Qlower), gettype(Qupper)}, typeof(Qlower), typeof(Qupper)}(Qlower, Qupper)
     end
end

function Base.:*(Q::MixedBC, u::AbstractVector)
    lower = Q.lower*u
    upper = Q.upper*u
    return BoundaryPaddedVector(lower.l, upper.r, u)
end

#implement Neumann and Dirichlet as special cases of RobinBC
NeumannBC(α::AbstractVector{T}, dx::Union{AbstractVector{T}, T}, order = 1) where T = RobinBC([zero(T), one(T), α[1]], [zero(T), one(T), α[2]], dx, order)
DirichletBC(αl::T, αr::T) where T = RobinBC([one(T), zero(T), αl], [one(T), zero(T), αr], 1.0, 2.0 )
#specialized constructors for Neumann0 and Dirichlet0
Dirichlet0BC(T::Type) = DirichletBC(zero(T), zero(T))
Neumann0BC(dx::Union{AbstractVector{T}, T}, order = 1) where T = NeumannBC([zero(T), zero(T)], dx, order)

# other acceptable argument signatures
#RobinBC(al::T, bl::T, cl::T, dx_l::T, ar::T, br::T, cr::T, dx_r::T, order = 1) where T = RobinBC([al,bl, cl], [ar, br, cr], dx_l, order)

"""
BridgeBC(u_low::AbstractArray{T,N}, u_up::AbstractArray{T,N}, indslow, indsup) # A different view in to 2 diffferent arrays on each end of the boundary, indslow is an iterable of indicies that index u_low, which extends the lower index end. Analogous for u_up and indsup with the upper boundary.

BridgeBC(u::AbstractArray{T,N}, inds) # The same view in to some array u at the index inds extends the boundary

-------------------------------------------------------------------------------------

Allows seperate domains governed by seperate equations to be bridged together with a boundary condition.
"""
struct BridgeBC{T,N,I} <: AffineBC{T}
    a_l::Vector{T} #Dummy vectors so that AffineBC methods still work
    b_l::SubArray{T,0,Array{T,N},NTuple{N,I},true}
    a_r::Vector{T}
    b_r::SubArray{T,0,Array{T,N},NTuple{N,I},true}
end

BridgeBC(u::AbstractArray, inds) = BridgeBC(u, inds, u, inds)

function BridgeBC(u_low::AbstractArray{T,N}, indslow, u_up::AbstractArray{T,N},  indsup) where {T, N}
    @assert length(indslow) == N
    @assert length(indsup) == N
    @assert mapreduce(x -> typeof(x) <: Integer, (&), indslow)
    @assert mapreduce(x -> typeof(x) <: Integer, (&), indsup)

    BridgeBC{T, length(indslow), eltype(indslow)}(zeros(T,1), view(u_low, indslow...), zeros(T,1), view(u_up, indsup...))
end

perpsize(A::AbstractArray{T,N}, dim::Integer) where {T,N} = size(A)[setdiff(1:N, dim)] #the size of A perpendicular to dim

"""
    Q1, Q2 = BridgeBC(u1::AbstractVector{T}, hilo1::String, bc1::AtomicBC{T}, u2::AbstractVector{T}, hilo2::AbstractVector{T}, bc2::AtomicBC{T})
-------------------------------------------------------------------------------------
Creates two BC operators that join array `u1` to `u2` at the `hilo1` end ("high" or "low" index end), and joins `u2` to `u1` with simalar settings given in `hilo2`.
The ends of `u1` and `u2` that are not connected will use the boundary conditions `bc1` and `bc2` respectively.

Use `Q1` to extend `u1` and `Q2` to extend `u2`.

When using these with a time/space stepping solve, please use elementwise equals on your u1 and u2 to avoid the need to create new BC operators each time, as follows:
    u_t1 .= L*Q*u_t0
-----------------------------------------------------------------------------------
Connecting two multi dimensional Arrays:
    Q1, Q2 = BridgeBC(u1::AbstractArray{T,N}, dim1::Int, hilo1::String, bc1, u2::AbstractArray{T,N}, dim2::Int, hilo2::String, bc2)
-----------------------------------------------------------------------------------

Creates two BC operators that join array `u1` to `u2` at the `hilo1` end ("high" or "low" index end) of dimension `dim1`, and joins `u2` to `u1` with simalar settings given in `hilo2` and `dim2`.
The ends of `u1` and `u2` that are not connected will use the boundary conditions `bc1` and `bc2` respectively.

Use `Q1` to extend `u1` and `Q2` to extend `u2`.

When using these with a time/space stepping solve, please use elementwise equals on your u1 and u2 to avoid the need to create new BC operators each time, as follows:
    u_t1 .= L*Q*u_t0
"""
function BridgeBC(u1::AbstractVector{T}, hilo1::String, bc1::AtomicBC{T}, u2::AbstractVector{T}, hilo2::AbstractVector{T}, bc2::AtomicBC{T}) where T
    if hilo1 == "low"
        view1 = view(u1, 1)
        if hilo2 == "low"
            view2 = view(u2, 1)
            BC1 = MixedBC(BridgeBC{T, 1, eltype(s1)}(zeros(T, 1), view2, zeros(T, 1), view2), bc1)
            BC2 = MixedBC(BridgeBC{T, 1, eltype(s1)}(zeros(T, 1), view1, zeros(T, 1), view1), bc2)
        elseif hilo2 == "high"
            view2 = view(u2, length(u2))
            BC1 = MixedBC(BridgeBC{T, 1, eltype(s1)}(zeros(T, 1), view2, zeros(T, 1), view2), bc1)
            BC2 = MixedBC(bc2, BridgeBC{T, 1, eltype(s1)}(zeros(T, 1), view1, zeros(T, 1), view1))
        else
            throw("hilo2 not recognized, please use \"high\" to connect u1 to u2 along the upper index of dim2 of u2 or \"low\" to connect along the lower index end")
        end
    elseif hilo1 == "high"
        view1 = view(u1, length(u1))
        if hilo2 == "low"
            view2 = view(u2, 1)
            BC1 = MixedBC(bc1, BridgeBC{T, 1, eltype(s1)}(zeros(T, 1), view2, zeros(T, 1), view2))
            BC2 = MixedBC(BridgeBC{T, 1, eltype(s1)}(zeros(T, 1), view1, zeros(T, 1), view1), bc2)
        elseif hilo2 == "high"
            view2 = view(u2, length(u2))
            BC1 = MixedBC(bc1, BridgeBC{T, 1, eltype(s1)}(zeros(T, 1), view2, zeros(T, 1), view2))
            BC2 = MixedBC(bc2, BridgeBC{T, 1, eltype(s1)}(zeros(T, 1), view1, zeros(T, 1), view1))
        else
            throw("hilo2 not recognized, please use \"high\" to connect u1 to u2 along the upper index of dim2 of u2 or \"low\" to connect along the lower index end")
        end
    else
        throw("hilo1 not recognized, please use \"high\" to connect u1 to u2 along the upper index of dim1 of u1 or \"low\" to connect along the lower index end")
    end
    return (BC1, BC2)
end

Base.:*(Q::BridgeBC{T,I,N}, u::AbstractVector{T}) where {T, I, N} = BoundaryPaddedVector{T, typeof(u)}(Q.b_l[1], Q.b_r[1], u)
Base.:*(Q::AffineBC, u::AbstractVector) = BoundaryPaddedVector(Q.a_l ⋅ u[1:length(Q.a_l)] + Q.b_l, Q.a_r ⋅ u[(end-length(Q.a_r)+1):end] + Q.b_r, u)
Base.:*(Q::PeriodicBC, u::AbstractVector) = BoundaryPaddedVector(u[end], u[1], u)

Base.size(Q::AtomicBC) = (Inf, Inf) #Is this nessecary?

gettype(Q::AbstractBC{T}) where T = T


#######################################################################
# Multidimensional
#######################################################################

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
