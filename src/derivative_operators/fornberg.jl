#############################################################
# Fornberg algorithm



immutable FiniteDifference{T<:Real,S<:SVector,LBC,RBC} <: AbstractDerivativeOperator{T}
    derivative_order    :: Int
    approximation_order :: Int
    dx                  :: Vector{T}
    dimension           :: Int
    stencil_length      :: Int
    stencil_coefs       :: Vector{S}
    boundary_point_count:: Tuple{Int,Int}
    boundary_length     :: Tuple{Int,Int}
    low_boundary_coefs  :: Ref{Vector{Vector{T}}}
    high_boundary_coefs :: Ref{Vector{Vector{T}}}
    boundary_condition  :: Ref{Tuple{Tuple{T,T,Any},Tuple{T,T,Any}}}
    t                   :: Ref{Int}

    Base.@pure function FiniteDifference{T,S,LBC,RBC}(derivative_order::Int, approximation_order::Int, dx::Vector{T},
                                            dimension::Int, BC) where {T<:Real,S<:SVector,LBC,RBC}
        dimension            = dimension
        dx                   = dx
        stencil_length       = derivative_order + approximation_order - 1 + (derivative_order+approximation_order)%2
        bl                   = derivative_order + approximation_order
        boundary_length      = (bl,bl)
        bpc                  = stencil_length - div(stencil_length,2) + 1
        bpc_array            = [bpc,bpc]
        grid_step            = dx
        x                    = [zero(T); cumsum(dx)]
        low_boundary_coefs   = Vector{T}[]
        high_boundary_coefs  = Vector{T}[]

        stl_2 = div(stencil_length,2)
        stencil_coefs        = [convert(SVector{stencil_length, T}, calculate_weights(derivative_order, x[idx],
                               x[idx-stl_2 : 1 : idx+stl_2])) for idx in stl_2+1:dimension-stl_2]

        left_bndry = initialize_left_boundary!(Val{:LO},low_boundary_coefs,stencil_coefs[1:stl_2],BC,derivative_order,
                                               grid_step[1:stl_2],bl,bpc_array,dx,LBC)

        right_bndry = initialize_right_boundary!(Val{:LO},high_boundary_coefs,stencil_coefs[end-stl_2+1:end],BC,derivative_order,
                                                 grid_step[end-stl_2+1:end],bl,bpc_array,dx,RBC)

        boundary_condition = (left_bndry, right_bndry)
        boundary_point_count = (bpc_array[1],bpc_array[2])

        t = 0

        new(derivative_order, approximation_order, dx, dimension, stencil_length,
            stencil_coefs,
            boundary_point_count,
            boundary_length,
            low_boundary_coefs,
            high_boundary_coefs,
            boundary_condition,
            t
            )
    end
    FiniteDifference{T}(dorder::Int,aorder::Int,dx::Vector{T},dim::Int,LBC::Symbol,RBC::Symbol;BC=(zero(T),zero(T))) where {T<:Real} =
        FiniteDifference{T, SVector{dorder+aorder-1+(dorder+aorder)%2,T}, LBC, RBC}(dorder, aorder, dx, dim, BC)
end

# This implements the Fornberg algorithm to obtain Finite Difference weights over arbitrary points to arbitrary order
function calculate_weights(order::Int, x0::T, x::Vector{T}) where T<:Real
    #=
        order: The derivative order for which we need the coefficients
        x0   : The point in the array 'x' for which we need the coefficients
        x    : A dummy array with relative coordinates, eg. central differences
               need coordinates centred at 0 while those at boundaries need
               coordinates starting from 0 to the end point
    =#
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
end
