import LinearMaps: LinearMap, AbstractLinearMap
abstract AbstractLinearOperator{T} <: AbstractLinearMap{T}

immutable LinearOperator{T<:AbstractFloat} <: AbstractLinearOperator{T}
    derivative_order    :: Int
    approximation_order :: Int
    grid_step           :: T
    stencil_length      :: Int
    stencil_coefs       :: Vector{T}
    boundary_point_count:: Int
    boundary_length     :: Int
    low_boundary_coefs  :: Vector{Vector{T}}
    high_boundary_coefs :: Vector{Vector{T}}

    function LinearOperator(derivative_order::Int=1, approximation_order::Int=2, grid_step::T=one(T))
        stencil_length       = derivative_order + approximation_order - 1
        boundary_length      = derivative_order + approximation_order
        boundary_point_count = stencil_length - Int(ceil(stencil_length / 2))
        low_boundary_coefs   = Vector{T}[]
        high_boundary_coefs  = Vector{T}[]
        stencil_coefs        = calculate_weights(derivative_order, zero(T),
                               grid_step .* collect(-div(stencil_length,2) : 1 : div(stencil_length,2)))
        # for i in 1 : boundary_point_count
        #     push!(low_boundary_coefs, calculate_weights(derivative_order, (i-1)*grid_step, collect(zero(T) : grid_step : (boundary_length-1) * grid_step)))
        #     push!(high_boundary_coefs, reverse(low_boundary_coefs[i]))
        #     isodd(derivative_order) ? high_boundary_coefs = -high_boundary_coefs : nothing
        # end
        new(derivative_order, approximation_order, grid_step, stencil_length,
            calculate_weights(derivative_order, zero(T), grid_step .* collect(-div(stencil_length,2) : 1 : div(stencil_length,2))),
            boundary_point_count,
            boundary_length,
            low_boundary_coefs,
            high_boundary_coefs
        )
    end
end


function derivative{T<:AbstractFloat}(y::Vector{T}, fd::LinearOperator{T})
    dy = zeros(T, length(y))
    derivative!(dy, y, fd)
    return dy
end


function derivative!{T<:AbstractFloat}(dy::Vector{T}, y::Vector{T}, fd::LinearOperator{T})
    N = length(y)
    @inbounds for i in 1 : fd.boundary_point_count
        bc = fd.low_boundary_coefs[i]
        tmp = zero(T)
        for j in 1 : fd.boundary_length
            tmp += bc[j] * y[j]
        end
        dy[i] = tmp
    end

    d = div(fd.stencil_length, 2)

    @inbounds for i in fd.boundary_point_count+1 : N-fd.boundary_point_count
        c = fd.stencil_coefs
        tmp = zero(T)
        for j in 1 : fd.stencil_length
            tmp += c[j] * y[i-d+j-1]
        end
        dy[i] = tmp
    end

    @inbounds for i in 1 : fd.boundary_point_count
        bc = fd.high_boundary_coefs[i]
        tmp = zero(T)
        for j in 1 : fd.boundary_length
            tmp += bc[j] * y[N - fd.boundary_length + j]
        end
        dy[N - i + 1] = tmp
    end
    return dy
end


function construct_differentiation_matrix{T<:AbstractFloat}(N::Int, fd::LinearOperator{T})
    D = zeros(N, N)
    for i in 1 : fd.boundary_point_count
        D[i, 1 : fd.boundary_length] = fd.low_boundary_coefs[i]
    end
    d = div(fd.stencil_length, 2)
    for i in fd.boundary_point_count + 1 : N - fd.boundary_point_count
        D[i, i-d : i+d] = fd.stencil_coefs
    end
    for i in 1 : fd.boundary_point_count
        D[N - i + 1, N - fd.boundary_length + 1 : N] = fd.high_boundary_coefs[i]
    end
    return D
end


immutable FiniteDifference <: AbstractLinearOperator
    # TODO: the general case
end


# This implements the Fornberg algorithm to obtain FD weights over arbitrary points to arbitrary order
function calculate_weights{T<:AbstractFloat}(order::Int, x0::T, x::Vector{T})
    N = length(x)
    @assert order < N "Not enough points for the requested order."
    M = order
    c1 = one(T)
    c4 = x[1] - x0
    C = zeros(N, M+1)
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
    return C[:,end]
    # return C
end

