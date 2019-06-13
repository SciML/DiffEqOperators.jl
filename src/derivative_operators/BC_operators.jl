abstract type AbstractBC{T} end

# robin, general, and in general neumann BCs are all affine. neumann0 is not however; is a specialization needed?
abstract type AffineBC{T,V} <: AbstractBC{T} end

struct DirichletBC{T} <: AbstractBC{T}
    l::T
    r::T
end

struct NeumannBC{T, V<:AbstractArray{T}} <: AffineBC{T,V}
    a_l::V
    b_l::T
    a_r::V
    b_r::T
    function NeumannBC(l::T, r::T, dx::AbstractArray{T}, order::T = one(T)) where {T,V}
        s = calculate_weights(1, zero(T), zero(T):order) #generate derivative coefficients about the boundary of required approximation order

        a_l = -dx_l.*s[2:end]./s[1]
        a_r = -dx_r.*s[end:-1:2]./s[1]

        b_l = dx_l.*l./s[1]
        b_r = dx_r.*l./s[1]

        return new{T, typeof(a_l)}(a_l, b_l, a_r, b_r)
    end
end

struct PeriodicBC{T} <: AbstractBC{T}

end

"""
  Implements a robin boundary condition operator Q that acts on a vector to give an extended vector as a result
  Referring to (https://github.com/JuliaDiffEq/DiffEqOperators.jl/files/3267835/ghost_node.pdf)

  the variables in correspond to al*u(0) + bl*u'(0) = cl


  Write vector b̄_1 as a vertical concatanation with b0 and the rest of the elements of b̄_1, denoted b̄`_1, the same with ū into u0 and ū`. b̄` = fill(β/Δx, length(stencil)-1)
  Pull out the product of u0 and b0 from the dot product. The stencil used to approximate u` is denoted s. b0 = α+(β/Δx)*s[1]
  Rearrange terms to find a general formula for u0:= -b̄`_1̇⋅ū`/b0 + γ/b0, which is dependent on ū` the robin coefficients and Δx.
  The non identity part of Qa is qa:= -b`_1/b0 = -β.*s[2:end]/(α+β*s[1]/Δx). The constant part is Qb = γ/(α+β*s[1]/Δx)
  do the same at the other boundary (amounts to a flip in s with the right boundary coeffs)
"""

# For  condition, the variables correspond to al*u(0) + bl*u'(0) = cl
# this should not break
struct RobinBC{T, V<:AbstractVector{T}} <: AffineBC{T,V}
    a_l::V
    b_l::T
    a_r::V
    b_r::T
    function RobinBC(l::AbstractArray{T}, r::AbstractArray{T}, dx::AbstractArray{T}, order::T = one(T)) where {T,V}
        cl, al, bl = l
        cr, ar, br = r
        dx_l, dx_r = dx

        s = calculate_weights(1, zero(T), zero(T):order) #generate derivative coefficients about the boundary of required approximation order

        a_l = -bl.*s[2:end]./(al .+ bl*s[1]./dx_l)
        a_r = -br.*s[end:-1:2]./(ar .+ br*s[1]./dx_r)

        b_l = cl/(al+bl*s[1]/dx_l)
        b_r = cr/(ar+br*s[1]/dx_r)

        return new{T, typeof(a_l)}(a_l, b_l, a_r, b_r)
    end
end

"""
Implements a generalization of the Robin boundary condition, where α is a vector of coefficients.
Represents a condition of the form α[1] + α[2]u[0] + α[3]u'[0] + α[4]u''[0]+... = 0
Implemented in an analogous way to the RobinBC
"""
#If you know a more correct name for this kind of BC, please post an issue/PR
struct GeneralBC{T, V<:AbstractVector{T}} <:AffineBC{T,V}
    a_l::V
    b_l::T
    a_r::V
    b_r::T
    function GeneralBC{T,V}(αl::AbstractArray{T}, αr::AbstractArray{T}, dx::AbstractArray{T}, order::T = one(T)) where {T,V<:AbstractArray}
        dx_l, dx_r = dx
        nl = length(αl)
        nr = length(αr)
        S_l = zeros(T, (nl-2, order+nl-2))
        S_r = zeros(T, (nr-2, order+nr-2))

        for i in 1:(nl-2)
            S_l[i,:] = [transpose(calculate_weights(i, zero(T), zero(T):(order+i))) transpose(zeros(T, nl-2-i))] #am unsure if the length of the dummy_x is correct here
        end
        for i in 1:(nr-2)
            S_r[i,:] = [transpose(calculate_weights(i, zero(T), zero(T):(order+i))) transpose(zeros(T, nr-2-i))]
        end
        s0_l = S_l[:,1] ; Sl = S_l[2:end,:]
        s0_r = S_r[:,1] ; Sr = S_r[2:end,:]

        denoml = αl[2] .+ αl[3:end] ⋅ s0_l
        denomr = αr[2] .+ αr[3:end] ⋅ s0_r

        a_l = -transpose(αl) ⋅ Sl ./denoml
        a_r = -transpose(αr) ⋅ Sr ./denomr

        b_l = -αl[1]/denoml
        b_r = -αr[1]/denomr
        new{T, V}(a_l,b_l,reverse!(a_r),b_r)
    end
end

# other acceptable argument signatures
RobinBC(al::T, bl::T, cl::T, dx_l::T, ar::T, br::T, cr::T, dx_r::T) where T = RobinBC([al,bl,cl], [ar, br, cr], [dx_l, dx_r])
RobinBC(al::T, bl::T, cl::T, dx_l::T, ar::T, br::T, cr::T, dx_r::T, order::T) where T = RobinBC([al,bl,cl], [ar, br, cr], [dx_l, dx_r], order)

NeumannBC(l::T, dx_l::T, r::T, dx_r::T) where T = NeumannBC(l, r, [dx_l,dx_r])
NeumannBC(l::T, dx_l::T, r::T, dx_r::T, order::T) where T = NeumannBC(l, r, [dx_l,dx_r], order)
# this  is 'boundary padded vector' as opposed to 'boundary padded array' to distinguish it from the n dimensional implementation that will eventually be neeeded
struct BoundaryPaddedVector{T,T2 <: AbstractVector{T}}
    l::T
    r::T
    u::T2
end


Base.:*(Q::DirichletBC, u) = BoundaryPaddedVector(Q.l,Q.r,u)
Base.:*(Q::PeriodicBC, u) = BoundaryPaddedVector(u[end], u[1], u)
Base.:*(Q::AffineBC, u) = BoundaryPaddedVector(Q.a_l ⋅ u[1:length(Q.a_l)] + Q.b_l, Q.a_r ⋅ u[(end-length(Q.a_r)+1):end] + Q.b_r, u)

Base.size(Q::AbstractBC) = (Inf, Inf) #Is this nessecary?
Base.length(Q::BoundaryPaddedVector) = length(Q.u) + 2
Base.lastindex(Q::BoundaryPaddedVector) = Base.length(Q)

function Base.getindex(Q::BoundaryPaddedVector,i)
    @show i
    if i == 1
        return Q.l
    elseif i == length(Q)
        return Q.r
    else
        return Q.u[i-1]
    end
end

function LinearAlgebra.Array(Q::AffineBC, N::Int)
    Q_L = [transpose(Q.a_l) transpose(zeros(N-length(Q.a_l))); Diagonal(ones(N)); transpose(zeros(N-length(Q.a_r))) transpose(Q.a_r)]
    Q_b = [Q.b_l; zeros(N); Q.b_r]
    return (Q_L, Q_b)
end

function LinearAlgebra.Array(Q::BoundaryPaddedVector)
    return [Q.l; Q.u; Q.r]
end
