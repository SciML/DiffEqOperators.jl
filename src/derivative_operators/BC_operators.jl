abstract type AbstractBC{T} <: AbstractDiffEqLinearOperator{T} end


abstract type AtomicBC{T} <: AbstractBC{T} end

"""
Robin, General, and in general Neumann, Dirichlet and Bridge BCs
are not necessarily linear operators.  Instead, they are affine
operators, with a constant term Q*x = Qa*x + Qb.
"""
abstract type AffineBC{T} <: AtomicBC{T} end

struct NeumannBC{N} end
struct Neumann0BC{N} end
struct DirichletBC{N} end
struct Dirichlet0BC{N} end
struct PeriodicBC{T} <: AtomicBC{T}
    PeriodicBC(T::Type) = new{T}()
end
struct RobinBC{T, V<:AbstractVector{T}} <: AffineBC{T}
    a_l::V
    b_l::T
    a_r::V
    b_r::T
    function RobinBC(l::NTuple{3,T}, r::NTuple{3,T}, dx::T, order = 1) where {T}
        αl, βl, γl = l
        αr, βr, γr = r

        s = calculate_weights(1, one(T), Array(one(T):convert(T,order+1))) #generate derivative coefficients about the boundary of required approximation order

        a_l = -s[2:end]./(αl*dx/βl + s[1])
        a_r = s[end:-1:2]./(αr*dx/βr - s[1]) # for other boundary stencil is flippedlr with *opposite sign*

        b_l = γl/(αl+βl*s[1]/dx)
        b_r = γr/(αr-βr*s[1]/dx)

        return new{T, typeof(a_l)}(a_l, b_l, a_r, b_r)
    end
    function RobinBC(l::Union{NTuple{3,T},AbstractVector{T}}, r::Union{NTuple{3,T},AbstractVector{T}}, dx::AbstractVector{T}, order = 1) where {T}
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

        return new{T, typeof(a_l)}(a_l, b_l, a_r, b_r)
    end
end



"""
q = GeneralBC(α_leftboundary, α_rightboundary, dx::T, approximation_order)

-------------------------------------------------------------------------------------

Implements a generalization of the Robin boundary condition, where α is a vector of coefficients.
Represents a condition of the form α[1] + α[2]u[0] + α[3]u'[0] + α[4]u''[0]+... = 0
Implemented in a similar way to the RobinBC (see above).
This time there are multiple stencils for multiple derivative orders - these can be written as a matrix S.
All components that multiply u(0) are factored out, turns out to only involve the first column of S, s̄0. The rest of S is denoted S`. the coeff of u(0) is s̄0⋅ᾱ[3:end] + α[2].
the remaining components turn out to be ᾱ[3:end]⋅(S`ū`) or equivalently (transpose(ᾱ[3:end])*S`)⋅ū`. Rearranging, a stencil q_a to be dotted with ū` upon extension can readily be found, along with a constant component q_b
"""
struct GeneralBC{T, L<:AbstractVector{T}, R<:AbstractVector{T}} <:AffineBC{T}
    a_l::L
    b_l::T
    a_r::R
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
        a_r = reverse(-transpose(transpose(αr[3:end]) * Sr) ./denomr)

        b_l = -αl[1]/denoml
        b_r = -αr[1]/denomr
        new{T, typeof(a_l), typeof(a_r)}(a_l,b_l,a_r,b_r)
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
        a_r = reverse(-transpose(transpose(αr[3:end]) * Sr) ./denomr)

        b_l = -αl[1]/denoml
        b_r = -αr[1]/denomr
        new{T,  typeof(a_l), typeof(a_r)}(a_l,b_l,a_r,b_r)
    end
end



#implement Neumann and Dirichlet as special cases of RobinBC
NeumannBC(α::NTuple{2,T}, dx::Union{AbstractVector{T}, T}, order = 1) where T = RobinBC((zero(T), one(T), α[1]), (zero(T), one(T), α[2]), dx, order)
DirichletBC(αl::T, αr::T) where T = RobinBC((one(T), zero(T), αl), (one(T), zero(T), αr), one(T), 2one(T) )
#specialized constructors for Neumann0 and Dirichlet0
Dirichlet0BC(T::Type) = DirichletBC(zero(T), zero(T))
Neumann0BC(dx::Union{AbstractVector{T}, T}, order = 1) where T = NeumannBC((zero(T), zero(T)), dx, order)

# other acceptable argument signatures
#RobinBC(al::T, bl::T, cl::T, dx_l::T, ar::T, br::T, cr::T, dx_r::T, order = 1) where T = RobinBC([al,bl, cl], [ar, br, cr], dx_l, order)

Base.:*(Q::AffineBC, u::AbstractVector) = BoundaryPaddedVector(Q.a_l ⋅ u[1:length(Q.a_l)] + Q.b_l, Q.a_r ⋅ u[(end-length(Q.a_r)+1):end] + Q.b_r, u)
Base.:*(Q::PeriodicBC, u::AbstractVector) = BoundaryPaddedVector(u[end], u[1], u)

Base.size(Q::AtomicBC) = (Inf, Inf) #Is this nessecary?

gettype(Q::AbstractBC{T}) where T = T
