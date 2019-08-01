abstract type AbstractBC{T} <: AbstractDiffEqLinearOperator{T} end


abstract type AtomicBC{T} <: AbstractBC{T} end

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
