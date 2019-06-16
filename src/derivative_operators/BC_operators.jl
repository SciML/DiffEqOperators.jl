abstract type AbstractBC{T} end

"""
Robin, General, and in general Neumann and Dirichlet BCs are all affine opeartors, meaning that they take the form Qx = Qax + Qb.
"""
abstract type AffineBC{T,V} <: AbstractBC{T} end

struct PeriodicBC{T} <: AbstractBC{T}

end

"""
  The variables in l are [αl, βl, γl], and correspond to a BC of the form al*u(0) + bl*u'(0) = cl

  Implements a robin boundary condition operator Q that acts on a vector to give an extended vector as a result
  Referring to (https://github.com/JuliaDiffEq/DiffEqOperators.jl/files/3267835/ghost_node.pdf)

  Write vector b̄₁ as a vertical concatanation with b0 and the rest of the elements of b̄ ₁, denoted b̄`₁, the same with ū into u0 and ū`. b̄`₁ = b̄`_2 = fill(β/Δx, length(stencil)-1)
  Pull out the product of u0 and b0 from the dot product. The stencil used to approximate u` is denoted s. b0 = α+(β/Δx)*s[1]
  Rearrange terms to find a general formula for u0:= -b̄`₁̇⋅ū`/b0 + γ/b0, which is dependent on ū` the robin coefficients and Δx.
  The non identity part of Qa is qa:= -b`₁/b0 = -β.*s[2:end]/(α+β*s[1]/Δx). The constant part is Qb = γ/(α+β*s[1]/Δx)
  do the same at the other boundary (amounts to a flip of s[2:end], with the other set of boundary coeffs)
"""
struct RobinBC{T, V<:AbstractVector{T}} <: AffineBC{T,V}
    a_l::V
    b_l::T
    a_r::V
    b_r::T
    function RobinBC(l::AbstractArray{T}, r::AbstractArray{T}, dx::AbstractArray{T}, order = one(T)) where {T}
        αl, βl, γl = l
        αr, βr, γr = r
        dx_l, dx_r = dx

        s = calculate_weights(1, one(T), Array(one(T):convert(T,order+1))) #generate derivative coefficients about the boundary of required approximation order

        a_l = -s[2:end]./(αl*dx_l/βl + s[1])
        a_r = s[end:-1:2]./(αr*dx_r/βr - s[1]) # for other boundary stencil is flippedlr with *opposite sign*

        b_l = γl/(αl+βl*s[1]/dx_l)
        b_r = γr/(αr-βr*s[1]/dx_r)

        return new{T, typeof(a_l)}(a_l, b_l, a_r, b_r)
    end
end

"""
Implements a generalization of the Robin boundary condition, where α is a vector of coefficients.
Represents a condition of the form α[1] + α[2]u[0] + α[3]u'[0] + α[4]u''[0]+... = 0
Implemented in a similar way to the RobinBC (see above).

This time there are multiple stencils for multiple derivative orders - these can be written as a matrix S.
All components that multiply u(0) are factored out, turns out to only involve the first colum of S, s̄0. The rest of S is denoted S`. the coeff of u(0) is s̄0⋅ᾱ[3:end] + α[2].
the remaining components turn out to be ᾱ[3:end]⋅(S`ū`) or equivalantly (transpose(ᾱ[3:end])*S`)⋅ū`. Rearranging, a stencil q_a to be dotted with ū` upon extension can readily be found, along with a constant component q_b
"""
struct GeneralBC{T, V<:AbstractVector{T}} <:AffineBC{T,V}
    a_l::V
    b_l::T
    a_r::V
    b_r::T
    function GeneralBC(αl::AbstractArray{T}, αr::AbstractArray{T}, dx::AbstractArray{T}, order = 1) where {T}
        dx_l, dx_r = dx
        nl = length(αl)
        nr = length(αr)
        S_l = zeros(T, (nl-2, order+nl-2))
        S_r = zeros(T, (nr-2, order+nr-2))

        for i in 1:(nl-2)
            S_l[i,:] = [transpose(calculate_weights(i, one(T), Array(one(T):convert(T, order+i)))) transpose(zeros(T, nl-2-i-order))] #am unsure if the length of the dummy_x is correct here
        end
        for i in 1:(nr-2)
            S_r[i,:] = [transpose(calculate_weights(i, one(T), Array(one(T):convert(T, order+i)))) transpose(zeros(T, nr-2-i-order))]
        end
        s0_l = S_l[:,1] ; Sl = S_l[2:end,:]
        s0_r = -S_r[:,1] ; Sr = -S_r[2:end,:]

        denoml = αl[2] .+ αl[3:end] ⋅ s0_l
        denomr = αr[2] .+ αr[3:end] ⋅ s0_r

        a_l = -transpose(αl) * Sl ./denoml
        a_r = -transpose(αr) * Sr ./denomr

        b_l = -αl[1]/denoml
        b_r = -αr[1]/denomr
        new{T, typeof(a_l)}(a_l,b_l,reverse!(a_r),b_r)
    end
end

#implement Neumann and Dirichlet as special cases of RobinBC
NeumannBC(α::AbstractVector{T}, dx::AbstractVector{T}, order = 1) where T = RobinBC([zero(T), one(T), α[1]], [zero(T), one(T), α[2]], dx, order)
DirichletBC(α::AbstractVector{T}, dx::AbstractVector{T}, order = 1) where T = RobinBC([one(T), zero(T), α[1]], [one(T), zero(T), α[2]], dx, order)
# other acceptable argument signatures
RobinBC(al::T, bl::T, cl::T, dx_l::T, ar::T, br::T, cr::T, dx_r::T, order = 1) where T = RobinBC([al,bl, cl], [ar, br, cr], [dx_l, dx_r], order)

# this  is 'boundary padded vector' as opposed to 'boundary padded array' to distinguish it from the n dimensional implementation that will eventually be neeeded
struct BoundaryPaddedVector{T,T2 <: AbstractVector{T}}
    l::T
    r::T
    u::T2
end

Base.:*(Q::AffineBC, u) = BoundaryPaddedVector(Q.a_l ⋅ u[1:length(Q.a_l)] + Q.b_l, Q.a_r ⋅ u[(end-length(Q.a_r)+1):end] + Q.b_r, u)

Base.size(Q::AbstractBC) = (Inf, Inf) #Is this nessecary?
Base.length(Q::BoundaryPaddedVector) = length(Q.u) + 2
Base.size(Q::BoundaryPaddedVector) = (length(Q),)
Base.lastindex(Q::BoundaryPaddedVector) = Base.length(Q)

function Base.getindex(Q::BoundaryPaddedVector,i)
    if i == 1
        return Q.l
    elseif i == length(Q)
        return Q.r
    else
        return Q.u[i-1]
    end
end

function LinearAlgebra.Array(Q::AffineBC{T,V}, N::Int) where {T,V}
    Q_L = [transpose(Q.a_l) transpose(zeros(T, N-length(Q.a_l))); Diagonal(ones(T,N)); transpose(zeros(T, N-length(Q.a_r))) transpose(Q.a_r)]
    Q_b = [Q.b_l; zeros(T,N); Q.b_r]
    return (Array(Q_L), Q_b)
end

function SparseArrays.SparseMatrixCSC(Q::AffineBC{T,V}, N::Int) where {T,V}
    Q_L = [transpose(Q.a_l) transpose(zeros(T, N-length(Q.a_l))); Diagonal(ones(T,N)); transpose(zeros(T, N-length(Q.a_r))) transpose(Q.a_r)]
    Q_b = [Q.b_l; zeros(T,N); Q.b_r]
    return (Q_L, Q_b)
end

function SparseArrays.sparse(Q::AffineBC{T,V}, N::Int) where {T,V}
    SparseMatrixCSC(Q,N)
end

LinearAlgebra.Array(Q::PeriodicBC{T}, N::Int) where T = Array([transpose(zeros(T, N-1)) one(T); Diagonal(ones(T,N)); one(T) transpose(zeros(T, N-1))])
SparseArrays.SparseMatrixCSC(Q::PeriodicBC{T}, N::Int) where T = [transpose(zeros(T, N-1)) one(T); Diagonal(ones(T,N)); one(T) transpose(zeros(T, N-1))]
SparseArrays.sparse(Q::PeriodicBC{T}, N::Int) where T = SparseMatrixCSC(Q,N)

function LinearAlgebra.Array(Q::BoundaryPaddedVector)
    return [Q.l; Q.u; Q.r]
end

#TODO: Implement Sparse concretization
