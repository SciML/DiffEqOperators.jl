#############################################################
# Fornberg algorithm
function derivative{T<:Real}(y::Vector{T}, A::LinearOperator{T})
    dy = zeros(T, length(y))
    derivative!(dy, y, A)
    return dy
end


function derivative!{T<:Real}(dy::Vector{T}, y::Vector{T}, A::LinearOperator{T})
    N = length(y)
    #=
        Derivative is calculated in 3 parts:-
            1. For the initial boundary points
            2. For the middle points
            3. For the terminating boundary points
    =#
    @inbounds for i in 1 : A.boundary_point_count
        bc = A.low_boundary_coefs[i]
        tmp = zero(T)
        for j in 1 : A.boundary_length
            tmp += bc[j] * y[j]
        end
        dy[i] = tmp
    end

    d = div(A.stencil_length, 2)

    @inbounds for i in A.boundary_point_count+1 : N-A.boundary_point_count
        c = A.stencil_coefs
        tmp = zero(T)
        for j in 1 : A.stencil_length
            tmp += c[j] * y[i-d+j-1]
        end
        dy[i] = tmp
    end

    @inbounds for i in 1 : A.boundary_point_count
        bc = A.high_boundary_coefs[i]
        tmp = zero(T)
        for j in 1 : A.boundary_length
            tmp += bc[j] * y[N - A.boundary_length + j]
        end
        dy[N - i + 1] = tmp
    end
    return dy
end


function construct_differentiation_matrix{T<:Real}(N::Int, A::LinearOperator{T})
    #=
        This is for calculating the derivative in one go. But we are creating a function
        which can calculate the derivative by-passing the costly matrix multiplication.
    =#
    D = zeros(T, N, N)
    for i in 1 : A.boundary_point_count
        D[i, 1 : A.boundary_length] = A.low_boundary_coefs[i]
    end
    d = div(A.stencil_length, 2)
    for i in A.boundary_point_count + 1 : N - A.boundary_point_count
        D[i, i-d : i+d] = A.stencil_coefs
    end
    for i in 1 : A.boundary_point_count
        D[N - i + 1, N - A.boundary_length + 1 : N] = A.high_boundary_coefs[i]
    end
    return D
end


# immutable FiniteDifference <: AbstractLinearOperator
#     # TODO: the general case ie. with an uneven grid
# end


# This implements the Fornberg algorithm to obtain Finite Difference weights over arbitrary points to arbitrary order
function calculate_weights{T<:Real}(order::Int, x0::T, x::Vector{T})
    N = length(x)
    @assert order < N "Not enough points for the requested order."
    M = order
    c1 = one(T)
    c4 = x[1] - x0
    C = zeros(T, N, M+1)
    C[1,1] = 1
    @inbounds for i in 1 : N-1
        i1 = i + 1
        mn = min(i, M)
        c2 = one(T)
        c5 = c4
        c4 = x[i1] - x0
        for j in 0 : i-1
            j1 = j + 1
            c3 = x[i1] - x[j1]
            c2 *= c3
            if j == i-1
                for s in mn : -1 : 1
                    s1 = s + 1
                    C[i1,s1] = c1*(s*C[i,s] - c5*C[i,s1]) / c2
                end
                C[i1,1] = -c1*c5*C[i,1] / c2
           end
            for s in mn : -1 : 1
                s1 = s + 1
                C[j1,s1] = (c4*C[j1,s1] - s*C[j1,s]) / c3
            end
            C[j1,1] = c4 * C[j1,1] / c3
        end
        c1 = c2
    end
    #=
        This is to fix the problem of numerical instability which occurs when the sum of the stencil_coefficients is not
        exactly 0.
        https://scicomp.stackexchange.com/questions/11249/numerical-derivative-and-finite-difference-coefficients-any-update-of-the-fornb
        Stack Overflow answer on this issue.
        http://epubs.siam.org/doi/pdf/10.1137/S0036144596322507 - Modified Fornberg Algorithm
    =#
    _C = C[:,end]
    _C[div(N,2)+1] -= sum(_C)
    return _C
    # return C
end
