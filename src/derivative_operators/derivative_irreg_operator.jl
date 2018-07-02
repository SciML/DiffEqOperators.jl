
struct FiniteDifference{T<:Real,S<:SVector,LBC,RBC} <: AbstractDerivativeOperator{T}
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
        # dimension            = dimension
        # dx                   = dx
        stencil_length       = derivative_order + approximation_order - 1 + (derivative_order+approximation_order)%2
        bl                   = derivative_order + approximation_order
        boundary_length      = (bl,bl)
        bpc                  = stencil_length - div(stencil_length,2) + 1
        bpc_array            = [bpc,bpc]
        grid_step            = dx #unnecessary

        if any(x->x<zero(T),dx)
            error("All grid steps must be greater than 0.0!")
        end
        x                    = [zero(T); cumsum(dx)]
        low_boundary_coefs   = Vector{T}[]
        high_boundary_coefs  = Vector{T}[]

        stl_2 = div(stencil_length,2)
        stencil_coefs        =[convert(SVector{stencil_length, T}, calculate_weights(derivative_order, zero(T),
                               x[i-stl_2 : i+stl_2]-x[i])) for i in stl_2+1:dimension-stl_2]

        left_bndry = initialize_left_boundary!(Val{:FD},low_boundary_coefs,stencil_coefs[1:bl],
                                                BC,derivative_order,grid_step[1:bl],bl,bpc_array,LBC)

        right_bndry = initialize_right_boundary!(Val{:FD},high_boundary_coefs,stencil_coefs[end-bl+1:end],
                                                 BC,derivative_order,grid_step[end-bl+1:end],bl,bpc_array,RBC)

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


function Base.A_mul_B!(x_temp::AbstractVector{T}, A::FiniteDifference{T}, x::AbstractVector{T}) where T<:Real
    convolve_BC_left!(x_temp, x, A)
    convolve_interior!(x_temp, x, A)
    convolve_BC_right!(x_temp, x, A)
    A.t[] += 1 # incrementing the internal time stamp
end


#=
    This function is used to update the boundary conditions especially if they evolve with
    time.
=#
function DiffEqBase.update_coefficients!(A::FiniteDifference{T,S,LBC,RBC};BC=nothing) where {T<:Real,S<:SVector,RBC,LBC}
    if BC != nothing
        LBC == :Robin ? (length(BC[1])==3 || error("Enter the new left boundary condition as a 1-tuple")) :
                        (length(BC[1])==1 || error("Robin BC needs a 3-tuple for left boundary condition"))

        RBC == :Robin ? length(BC[2])==3 || error("Enter the new right boundary condition as a 1-tuple") :
                        length(BC[2])==1 || error("Robin BC needs a 3-tuple for right boundary condition")

        left_bndry = initialize_left_boundary!(Val{:FD}, A.low_boundary_coefs[],A.stencil_coefs[1:A.boundary_length],BC,
                                               A.derivative_order,A.dx[1:A.boundary_length],A.boundary_length[1],LBC)

        right_bndry = initialize_right_boundary!(A.high_boundary_coefs[],A.stencil_coefs[end-A.boundary_length:end],BC,
                                                 A.derivative_order,A.dx[end-A.boundary_length:end],A.boundary_length[2],RBC)

        boundary_condition = (left_bndry, right_bndry)
        A.boundary_condition[] = boundary_condition
    end
end

#################################################################################################

#=
    This function sets us up to apply the boundary condition correctly. It returns a
    3-tuple which is basically the coefficients in the equation
                            a*u + b*du/dt = c(t)
    The RHS ie. 'c' can be a function of time as well and therefore it is implemented
    as an anonymous function.
=#
function initialize_left_boundary!(::Type{Val{:FD}},low_boundary_coefs,stencil_coefs,BC,derivative_order,grid_step::Vector{T},
                                   boundary_length,boundary_point_count,LBC) where T
    stencil_length = length(stencil_coefs[1])

    if LBC == :None
        #=
            Val{:FD} is type parametrization on symbols. There are two different right_None_BC!
            functions, one for LinearOperator and one for UpwindOperator. So to select the correct
            function without changing the name, this trick has been applied.
        =#
        boundary_point_count[1] = div(stencil_length,2)
        return (zero(T),zero(T),left_None_BC!(Val{:FD},low_boundary_coefs,stencil_length,derivative_order,
                                              grid_step,boundary_length)*BC[1])
    elseif LBC == :Neumann
        return (zero(T),one(T),left_Neumann_BC!(Val{:FD},low_boundary_coefs,stencil_length,derivative_order,
                                                grid_step,boundary_length)*BC[1])
    elseif LBC == :Robin
        error("LBC Robin not yet supported for irregular grid.")
        # return (BC[1][1],-BC[1][2],left_Robin_BC!(Val{:FD},low_boundary_coefs,stencil_length,
                                                   # BC[1],derivative_order,grid_step,
                                                   # boundary_length,dx)*BC[1][3])
    elseif LBC == :Dirichlet0
        return (one(T),zero(T),zero(T)*BC[1])

    elseif LBC == :Dirichlet
        typeof(BC[1]) <: Real ? ret = t->BC[1] : ret = BC[1]
        return (one(T),zero(T),ret)

    elseif LBC == :Neumann0
        return (zero(T),one(T),zero(T))

    elseif LBC == :periodic
        return (zero(T),zero(T),zero(T))

    else
        error("Unrecognized Boundary Type!")
    end
end


#=
    This function sets us up to apply the boundary condition correctly. It returns a
    3-tuple which is basically the coefficients in the equation
                            a*u + b*du/dt = c(t)
    The RHS ie. 'c' can be a function of time as well and therefore it is implemented
    as an anonymous function.
=#
function initialize_right_boundary!(::Type{Val{:FD}},high_boundary_coefs,stencil_coefs,BC,derivative_order,grid_step::Vector{T},
                                    boundary_length,boundary_point_count,RBC) where T
    stencil_length = length(stencil_coefs[1])

    if RBC == :None
        boundary_point_count[2] = div(stencil_length,2)
        #=
            Val{:FD} is type parametrization on symbols. There are two different right_None_BC!
            functions, one for LinearOperator and one for UpwindOperator. So to select the correct
            function without changing the name, this trick has been applied.
        =#
        return (zero(T),zero(T),right_None_BC!(Val{:FD},high_boundary_coefs,stencil_length,derivative_order,
                               grid_step,boundary_length)*BC[2])
    elseif RBC == :Neumann
        error("RBC Robin not yet supported for irregular grid.")
        # return (zero(T),one(T),right_Neumann_BC!(Val{:FD},high_boundary_coefs,stencil_length,derivative_order,
        #                           grid_step,boundary_length)*BC[2])
    elseif RBC == :Robin
        error("RBC Robin not yet supported for irregular grid.")
        # return (BC[2][1],BC[2][2],right_Robin_BC!(Val{:FD},high_boundary_coefs,stencil_length,
                                                    # BC[2],derivative_order,grid_step,
                                                    # boundary_length,dx)*BC[2][3])
    elseif RBC == :Dirichlet0
        return (one(T),zero(T),zero(T)*BC[2])

    elseif RBC == :Dirichlet
        typeof(BC[2]) <: Real ? ret = t->BC[2] : ret = BC[2]
        return (one(T),zero(T),ret)

    elseif RBC == :Neumann0
        return (zero(T),one(T),zero(T))

    elseif RBC == :periodic
        return (zero(T),zero(T),zero(T))

    else
        error("Unrecognized Boundary Type!")
    end
end


function left_None_BC!(::Type{Val{:FD}},low_boundary_coefs,stencil_length,derivative_order,
                       grid_step::Vector{T},boundary_length) where T
    # Fixes the problem excessive boundary points
    boundary_point_count = div(stencil_length,2)
    l_diff               = zero(T)
    mid                  = div(stencil_length,2)
    x = [0.0; cumsum(grid_step)]
    for i in 1 : boundary_point_count
        # One-sided stencils require more points for same approximation order
        # TODO: I don't know if this is the correct stencil length for i > 1?
        push!(low_boundary_coefs, calculate_weights(derivative_order, zero(T), x[1:boundary_length]-x[i]))
    end
    return l_diff
end


function right_None_BC!(::Type{Val{:FD}},high_boundary_coefs,stencil_length,derivative_order,
                        grid_step::Vector{T},boundary_length) where T
    boundary_point_count = div(stencil_length,2)
    high_temp            = zeros(T,boundary_length)
    flag                 = derivative_order*boundary_point_count%2
    aorder               = boundary_length - 1
    r_diff               = zero(T)
    x                    = [0.0; cumsum(grid_step)]

    for i in length(x) : -1 : length(x) - boundary_point_count + 1
        # One-sided stencils require more points for same approximation order
        push!(high_boundary_coefs, calculate_weights(derivative_order,x[i],x[end - boundary_length + 1 : end]))
    end
    return r_diff
end


function left_Neumann_BC!(::Type{Val{:FD}},low_boundary_coefs,stencil_length,derivative_order,
                          grid_step::Vector{T},boundary_length) where T
    boundary_point_count = stencil_length - div(stencil_length,2) + 1
    # first_order_coeffs   = zeros(T,boundary_length)
    # original_coeffs      = zeros(T,boundary_length)
    l_diff               = one(T)
    mid                  = div(stencil_length,2)+1
    x = [0.0;cumsum(grid_step)]

    first_order_coeffs = calculate_weights(1, zero(T), x[1:boundary_length])
    original_coeffs =  calculate_weights(derivative_order, zero(T), x[1:boundary_length])
    l_diff = original_coeffs[end]/first_order_coeffs[end]
    scale!(first_order_coeffs, original_coeffs[end]/first_order_coeffs[end])
    # scale!(original_coeffs, first_order_coeffs[end]/original_coeffs[end])
    @. original_coeffs = original_coeffs - first_order_coeffs
    # copy!(first_order_coeffs, first_order_coeffs[1:end-1])
    push!(low_boundary_coefs, original_coeffs[1:end-1])

    for i in 2 : boundary_point_count
        #=  this means that a stencil will suffice ie. we dont't need to worry about the boundary point
            being considered in the low_boundary_coefs
        =#

        if i > mid
            pos=i
            push!(low_boundary_coefs, calculate_weights(derivative_order, x[pos], x[1:boundary_length]))
        else
            pos=i-1
            push!(low_boundary_coefs, append!([zero(T)],calculate_weights(derivative_order, x[pos], x[1:boundary_length])))
        end
    end

    return l_diff
end


#TODO:::
# function right_Neumann_BC!(::Type{Val{:FD}},high_boundary_coefs,stencil_length,derivative_order,
#                            grid_step::Vector{T},boundary_length) where T
#     boundary_point_count = stencil_length - div(stencil_length,2) + 1
#     flag                 = derivative_order*boundary_point_count%2
#     # original_coeffs      = zeros(T,boundary_length)
#     r_diff               = one(T)
#     mid                  = div(stencil_length,2)+1
#     x                    = [0.0; cumsum(grid_step)]
#     @show flag
#
#     # this part is to incorporate the value of first derivative at the right boundary
#     first_order_coeffs = calculate_weights(1, x[end-boundary_point_count+1], x[end - boundary_length + 1 : end])
#     original_coeffs = calculate_weights(derivative_order, x[end-boundary_point_count+1], x[end - boundary_length + 1 : end])
#
#     r_diff = original_coeffs[1]/first_order_coeffs[1]
#     scale!(first_order_coeffs, original_coeffs[1]/first_order_coeffs[1])
#     # scale!(original_coeffs, first_order_coeffs[1]/original_coeffs[1])
#     @. original_coeffs = original_coeffs - first_order_coeffs
#     # copy!(first_order_coeffs, first_order_coeffs[1:end-1])
#     push!(high_boundary_coefs, original_coeffs[2:end])
#     #TODO doesnt work... i don't get it.
#     for i in length(x) - boundary_point_count+2 : length(x)
#         # push!(high_boundary_coefs,ones(T,boundary_length)*i)
#         # @show i
#         # @show length(x)-i+2
#         if i > length(x)-mid+1
#             pos=i
#             push!(high_boundary_coefs, calculate_weights(derivative_order, x[pos],x[end - boundary_length + 1 : end]))
#         else
#             pos=i-1
#             push!(high_boundary_coefs, append!([zero(T)], calculate_weights(derivative_order, x[pos],x[end - boundary_length  : end-1])))
#         end
#     end
#
#     reverse!(high_boundary_coefs)
#
#
#     return r_diff
# end


# function left_Robin_BC!(::Type{Val{:FD}},low_boundary_coefs,stencil_length,params,
#                         derivative_order,grid_step::Vector{T},boundary_length,dx) where T
#     boundary_point_count = stencil_length - div(stencil_length,2) + 1
#     first_order_coeffs   = zeros(T,boundary_length)
#     original_coeffs      = zeros(T,boundary_length)
#     l_diff               = one(T)
#     mid                  = div(stencil_length,2)+1
#
#     # in Robin BC the left boundary has opposite sign by convention
#     first_order_coeffs = -params[2]*calculate_weights(1, (0)*grid_step, collect(zero(T) : grid_step : (boundary_length-1)* grid_step))
#     first_order_coeffs[1] += dx*params[1]
#     original_coeffs =  calculate_weights(derivative_order, (0)*grid_step, collect(zero(T) : grid_step : (boundary_length-1) * grid_step))
#
#     l_diff = original_coeffs[end]/first_order_coeffs[end]
#     scale!(first_order_coeffs, original_coeffs[end]/first_order_coeffs[end])
#     # scale!(original_coeffs, first_order_coeffs[end]/original_coeffs[end])
#     @. original_coeffs = original_coeffs - first_order_coeffs
#     # copy!(first_order_coeffs, first_order_coeffs[1:end-1])
#     push!(low_boundary_coefs, original_coeffs[1:end-1])
#
#     for i in 2 : boundary_point_count
#         #=
#             this means that a stencil will suffice ie. we dont't need to worry about the boundary point
#             being considered in the low_boundary_coefs
#         =#
#         if i > mid
#             pos=i-1
#             push!(low_boundary_coefs, calculate_weights(derivative_order, pos*grid_step, collect(zero(T) : grid_step : (boundary_length-1) * grid_step)))
#         else
#             pos=i-2
#             push!(low_boundary_coefs, append!([zero(T)],calculate_weights(derivative_order, pos*grid_step, collect(zero(T) : grid_step : (boundary_length-1) * grid_step))))
#         end
#     end
#
#     return l_diff
# end


# function right_Robin_BC!(::Type{Val{:FD}},high_boundary_coefs,stencil_length,params,
#                         derivative_order,grid_step::Vector{T},boundary_length,dx) where T
#     boundary_point_count = stencil_length - div(stencil_length,2) + 1
#     flag                 = derivative_order*boundary_point_count%2
#     original_coeffs      = zeros(T,boundary_length)
#     r_diff               = one(T)
#     mid                  = div(stencil_length,2)+1
#
#     first_order_coeffs = params[2]*calculate_weights(1, (boundary_point_count-1)*grid_step,
#                                                      collect(zero(T) : grid_step : (boundary_length-1) * grid_step))
#     first_order_coeffs[end] += dx*params[1]
#     reverse!(first_order_coeffs)
#     isodd(flag) ? negate!(first_order_coeffs) : nothing
#
#     copy!(original_coeffs, calculate_weights(derivative_order, (boundary_point_count-1)*grid_step,
#                                              collect(zero(T) : grid_step : (boundary_length-1) * grid_step)))
#     reverse!(original_coeffs)
#     isodd(flag) ? negate!(original_coeffs) : nothing
#
#     r_diff = original_coeffs[1]/first_order_coeffs[1]
#     scale!(first_order_coeffs, original_coeffs[1]/first_order_coeffs[1])
#     # scale!(original_coeffs, first_order_coeffs[1]/original_coeffs[1])
#     @. original_coeffs = original_coeffs - first_order_coeffs
#     # copy!(first_order_coeffs, first_order_coeffs[1:end-1])
#
#     for i in 2 : boundary_point_count
#         #=
#             this means that a stencil will suffice ie. we dont't need to worry about the boundary point
#             being considered in the high_boundary_coefs. Same code for low_boundary_coefs but reversed
#             at the end
#         =#
#         if i > mid
#             pos=i-1
#             push!(high_boundary_coefs, calculate_weights(derivative_order, pos*grid_step,
#                                                          collect(zero(T) : grid_step : (boundary_length-1) * grid_step)))
#         else
#             pos=i-2
#             push!(high_boundary_coefs, append!([zero(T)],
#                                                calculate_weights(derivative_order,pos*grid_step,
#                                                                  collect(zero(T) : grid_step : (boundary_length-1) * grid_step))))
#         end
#     end
#     if flag == 1
#         negate!(high_boundary_coefs)
#     end
#     reverse!(high_boundary_coefs)
#     push!(high_boundary_coefs, original_coeffs[2:end])
#     return r_diff
# end

#################################################################################################


(L::FiniteDifference)(u,p,t) = L*u
(L::FiniteDifference)(du,u,p,t) = A_mul_B!(du,L,u)
# get_LBC(::FiniteDifference{A,B,C,D}) where {A,B,C,D} = C
# get_RBC(::FiniteDifference{A,B,C,D}) where {A,B,C,D} = D
