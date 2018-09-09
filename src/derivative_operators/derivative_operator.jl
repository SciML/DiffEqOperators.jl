get_type(::AbstractDerivativeOperator{T}) where {T} = T

function *(A::AbstractDerivativeOperator,x::AbstractVector)
    #=
        We will output a vector which is a supertype of the types of A and x
        to ensure numerical stability
    =#
    get_type(A) != eltype(x) ? error("DiffEqOperator and array are not of same type!") : nothing
    y = zeros(promote_type(eltype(A),eltype(x)), length(x))
    LinearAlgebra.mul!(y, A::AbstractDerivativeOperator, x::AbstractVector)
    return y
end


function *(A::AbstractDerivativeOperator,M::AbstractMatrix)
    #=
        We will output a vector which is a supertype of the types of A and x
        to ensure numerical stability
    =#
    get_type(A) != eltype(M) ? error("DiffEqOperator and array are not of same type!") : nothing
    y = zeros(promote_type(eltype(A),eltype(M)), size(M))
    LinearAlgebra.mul!(y, A::AbstractDerivativeOperator, M::AbstractMatrix)
    return y
end


function *(M::AbstractMatrix,A::AbstractDerivativeOperator)
    #=
        We will output a vector which is a supertype of the types of A and x
        to ensure numerical stability
    =#
    get_type(A) != eltype(M) ? error("DiffEqOperator and array are not of same type!") : nothing
    y = zeros(promote_type(eltype(A),eltype(M)), size(M))
    LinearAlgebra.mul!(y, A::AbstractDerivativeOperator, M::AbstractMatrix)
    return y
end


function *(A::AbstractDerivativeOperator,B::AbstractDerivativeOperator)
    # TODO: it will result in an operator which calculates
    #       the derivative of order A.dorder + B.dorder of
    #       approximation_order = min(approx_A, approx_B)
end


function negate!(arr::T) where T
    if size(arr,2) == 1
        rmul!(arr,-one(eltype(arr[1]))) #fix right neumann bc, eltype(Vector{T}) doesnt work.
    else
        for row in arr
            rmul!(row,-one(eltype(arr[1])))
        end
    end
end


struct DerivativeOperator{T<:Real,S<:SVector,LBC,RBC} <: AbstractDerivativeOperator{T}
    derivative_order    :: Int
    approximation_order :: Int
    dx                  :: T
    dimension           :: Int
    stencil_length      :: Int
    stencil_coefs       :: S
    boundary_point_count:: Tuple{Int,Int}
    boundary_length     :: Tuple{Int,Int}
    low_boundary_coefs  :: Ref{Vector{Vector{T}}}
    high_boundary_coefs :: Ref{Vector{Vector{T}}}
    boundary_condition  :: Ref{Tuple{Tuple{T,T,Any},Tuple{T,T,Any}}}
    t                   :: Ref{Int}

    function DerivativeOperator{T,S,LBC,RBC}(derivative_order::Int,
                                             approximation_order::Int, dx::T,
                                             dimension::Int, BC) where
                                             {T<:Real,S<:SVector,LBC,RBC}
        dimension            = dimension
        dx                   = dx
        stencil_length       = derivative_order + approximation_order - 1 + (derivative_order+approximation_order)%2
        bl                   = derivative_order + approximation_order
        boundary_length      = (bl,bl)
        bpc                  = stencil_length - div(stencil_length,2) + 1
        bpc_array            = [bpc,bpc]
        grid_step            = one(T)
        low_boundary_coefs   = Vector{T}[]
        high_boundary_coefs  = Vector{T}[]
        stencil_coefs        = convert(SVector{stencil_length, T}, calculate_weights(derivative_order, zero(T),
                               grid_step .* collect(-div(stencil_length,2) : 1 : div(stencil_length,2))))

        left_bndry = initialize_left_boundary!(Val{:LO},low_boundary_coefs,stencil_coefs,BC,derivative_order,
                                               grid_step,bl,bpc_array,dx,LBC)

        right_bndry = initialize_right_boundary!(Val{:LO},high_boundary_coefs,stencil_coefs,BC,derivative_order,
                                                 grid_step,bl,bpc_array,dx,RBC)

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
    DerivativeOperator{T}(dorder::Int,aorder::Int,dx::T,dim::Int,LBC::Symbol,RBC::Symbol;BC=(zero(T),zero(T))) where {T<:Real} =
        DerivativeOperator{T, SVector{dorder+aorder-1+(dorder+aorder)%2,T}, LBC, RBC}(dorder, aorder, dx, dim, BC)
end

#=
    This function is used to update the boundary conditions especially if they evolve with
    time.
=#
function DiffEqBase.update_coefficients!(A::DerivativeOperator{T,S,LBC,RBC};BC=nothing) where {T<:Real,S<:SVector,RBC,LBC}
    if BC != nothing
        LBC == :Robin ? (length(BC[1])==3 || error("Enter the new left boundary condition as a 1-tuple")) :
                        (length(BC[1])==1 || error("Robin BC needs a 3-tuple for left boundary condition"))

        RBC == :Robin ? length(BC[2])==3 || error("Enter the new right boundary condition as a 1-tuple") :
                        length(BC[2])==1 || error("Robin BC needs a 3-tuple for right boundary condition")

        left_bndry = initialize_left_boundary!(A.low_boundary_coefs[],A.stencil_coefs,BC,
                                               A.derivative_order,one(T),A.boundary_length[1],A.dx,LBC)

        right_bndry = initialize_right_boundary!(A.high_boundary_coefs[],A.stencil_coefs,BC,
                                                 A.derivative_order,one(T),A.boundary_length[2],A.dx,RBC)

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

    update: change of boundary conditions takes care of stencil out of boundary issue, but
    the current implementation removes the ability to have a time dependent c(t).
=#

function initialize_left_boundary!(::Type{Val{:LO}},low_boundary_coefs,stencil_coefs,BC,derivative_order,grid_step::T,
                                   boundary_length,boundary_point_count,dx,LBC) where T
    stencil_length = length(stencil_coefs)

    if LBC == :None
        #=
            Val{:LO} is type parametrization on symbols. There are two different right_None_BC!
            functions, one for LinearOperator and one for UpwindOperator. So to select the correct
            function without changing the name, this trick has been applied.
        =#
        boundary_point_count[1] = div(stencil_length,2)
        return (zero(T),zero(T),left_None_BC!(Val{:LO},low_boundary_coefs,stencil_length,derivative_order,
                                              grid_step,boundary_length)*BC[1]*dx)
    elseif LBC == :Neumann
        return (zero(T),one(T),left_Neumann_BC!(Val{:LO},low_boundary_coefs,stencil_length,derivative_order,
                                                grid_step,boundary_length)*BC[1]*dx)
    elseif LBC == :Robin
        return (BC[1][1],-BC[1][2],left_Robin_BC!(Val{:LO},low_boundary_coefs,stencil_length,
                                                   BC[1],derivative_order,grid_step,
                                                   boundary_length,dx)*BC[1][3]*dx)
    elseif LBC == :Dirichlet0
        boundary_point_count[1] = div(stencil_length,2)
        return (one(T),zero(T),left_Dirichlet0_BC!(Val{:LO},low_boundary_coefs,stencil_length,derivative_order,
                                              grid_step,boundary_length)*BC[1]*dx)

    elseif LBC == :Dirichlet
        # typeof(BC[1]) <: Real ? ret = t->BC[1] : ret = BC[1]
        boundary_point_count[1] = div(stencil_length,2)
        return (one(T),zero(T),left_Dirichlet_BC!(Val{:LO},low_boundary_coefs,stencil_length,derivative_order,
                                              grid_step,boundary_length)*BC[1]*dx)

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
function initialize_right_boundary!(::Type{Val{:LO}},high_boundary_coefs,stencil_coefs,BC,derivative_order,grid_step::T,
                                    boundary_length,boundary_point_count,dx,RBC) where T
    stencil_length = length(stencil_coefs)

    if RBC == :None
        boundary_point_count[2] = div(stencil_length,2)
        #=
            Val{:LO} is type parametrization on symbols. There are two different right_None_BC!
            functions, one for LinearOperator and one for UpwindOperator. So to select the correct
            function without changing the name, this trick has been applied.
        =#
        return (zero(T),zero(T),right_None_BC!(Val{:LO},high_boundary_coefs,stencil_length,derivative_order,
                               grid_step,boundary_length)*BC[2]*dx)
    elseif RBC == :Neumann
        return (zero(T),one(T),right_Neumann_BC!(Val{:LO},high_boundary_coefs,stencil_length,derivative_order,
                                  grid_step,boundary_length)*BC[2]*dx)
    elseif RBC == :Robin
        return (BC[2][1],BC[2][2],right_Robin_BC!(Val{:LO},high_boundary_coefs,stencil_length,
                                                    BC[2],derivative_order,grid_step,
                                                    boundary_length,dx)*BC[2][3]*dx)
    elseif RBC == :Dirichlet0
        boundary_point_count[2] = div(stencil_length,2)
        return (one(T),zero(T),right_Dirichlet0_BC!(Val{:LO},high_boundary_coefs,stencil_length,derivative_order,
                               grid_step,boundary_length)*BC[2]*dx)

    elseif RBC == :Dirichlet
        boundary_point_count[2] = div(stencil_length,2)
        # typeof(BC[2]) <: Real ? ret = t->BC[2] : ret = BC[2]
        return (one(T),zero(T),right_Dirichlet_BC!(Val{:LO},high_boundary_coefs,stencil_length,derivative_order,
                               grid_step,boundary_length)*BC[2]*dx)

    elseif RBC == :Neumann0
        return (zero(T),one(T),zero(T))

    elseif RBC == :periodic
        return (zero(T),zero(T),zero(T))

    else
        error("Unrecognized Boundary Type!")
    end
end


function left_None_BC!(::Type{Val{:LO}},low_boundary_coefs,stencil_length,derivative_order,
                       grid_step::T,boundary_length) where T
    # Fixes the problem excessive boundary points
    boundary_point_count = div(stencil_length,2)
    l_diff               = zero(T)
    mid                  = div(stencil_length,2)

    for i in 1 : boundary_point_count
        # One-sided stencils require more points for same approximation order
        # TODO: I don't know if this is the correct stencil length for i > 1?
        push!(low_boundary_coefs, calculate_weights(derivative_order, (i-1)*grid_step, collect(zero(T) : grid_step : (boundary_length-1)*grid_step)))
    end
    return l_diff
end


function right_None_BC!(::Type{Val{:LO}},high_boundary_coefs,stencil_length,derivative_order,
                        grid_step::T,boundary_length) where T
    boundary_point_count = div(stencil_length,2)
    high_temp            = zeros(T,boundary_length)
    flag                 = derivative_order*boundary_point_count%2
    aorder               = boundary_length - 1
    r_diff               = zero(T)

    for i in 1 : boundary_point_count
        # One-sided stencils require more points for same approximation order
        push!(high_boundary_coefs, calculate_weights(derivative_order, -(i-1)*grid_step, reverse(collect(zero(T) : -grid_step : -(boundary_length-1)*grid_step))))
    end
    return r_diff
end

function left_Dirichlet0_BC!(::Type{Val{:LO}},low_boundary_coefs,stencil_length,derivative_order,
                       grid_step::T,boundary_length) where T

    boundary_point_count = div(stencil_length,2)
    l_diff               = zero(T)
    mid                  = div(stencil_length,2)

    for i in 1 : boundary_point_count
        push!(low_boundary_coefs, calculate_weights(derivative_order, i*grid_step, collect(zero(T) : grid_step : (boundary_length-1)*grid_step))[2:end])
    end
    return l_diff
end

function right_Dirichlet0_BC!(::Type{Val{:LO}},high_boundary_coefs,stencil_length,derivative_order,
                        grid_step::T,boundary_length) where T
    boundary_point_count = div(stencil_length,2)
    high_temp            = zeros(T,boundary_length)
    flag                 = derivative_order*boundary_point_count%2
    aorder               = boundary_length - 1
    r_diff               = zero(T)

    for i in 1 : boundary_point_count
        push!(high_boundary_coefs, calculate_weights(derivative_order, -i*grid_step, reverse(collect(zero(T) : -grid_step : -(boundary_length-1)*grid_step)))[1:end-1])
    end
    return r_diff
end

function left_Dirichlet_BC!(::Type{Val{:LO}},low_boundary_coefs,stencil_length,derivative_order,
                       grid_step::T,boundary_length) where T

    boundary_point_count = div(stencil_length,2)
    l_diff               = zero(T)
    mid                  = div(stencil_length,2)
    push!(low_boundary_coefs,[one(T);zeros(T,stencil_length-1)])
    for i in 2 : boundary_point_count
        push!(low_boundary_coefs, calculate_weights(derivative_order, (i-1)*grid_step, collect(zero(T) : grid_step : (boundary_length-1)*grid_step)))
    end
    return l_diff
end

function right_Dirichlet_BC!(::Type{Val{:LO}},high_boundary_coefs,stencil_length,derivative_order,
                        grid_step::T,boundary_length) where T
    boundary_point_count = div(stencil_length,2)
    high_temp            = zeros(T,boundary_length)
    flag                 = derivative_order*boundary_point_count%2
    aorder               = boundary_length - 1
    r_diff               = zero(T)

    push!(high_boundary_coefs,[zeros(T,stencil_length-1);one(T)])
    for i in 2 : boundary_point_count
        push!(high_boundary_coefs, calculate_weights(derivative_order, -(i-1)*grid_step, reverse(collect(zero(T) : -grid_step : -(boundary_length-1)*grid_step))))
    end
    return r_diff
end

function left_Neumann_BC!(::Type{Val{:LO}},low_boundary_coefs,stencil_length,derivative_order,
                          grid_step::T,boundary_length) where T
    boundary_point_count = stencil_length - div(stencil_length,2) + 1
    first_order_coeffs   = zeros(T,boundary_length)
    original_coeffs      = zeros(T,boundary_length)
    l_diff               = one(T)
    mid                  = div(stencil_length,2)+1

    first_order_coeffs = calculate_weights(1, (0)*grid_step, collect(zero(T) : grid_step : (boundary_length-1)*grid_step))
    original_coeffs =  calculate_weights(derivative_order, (0)*grid_step, collect(zero(T) : grid_step : (boundary_length-1) * grid_step))
    l_diff = original_coeffs[end]/first_order_coeffs[end]
    rmul!(first_order_coeffs, original_coeffs[end]/first_order_coeffs[end])
    # rmul!(original_coeffs, first_order_coeffs[end]/original_coeffs[end])
    @. original_coeffs = original_coeffs - first_order_coeffs
    # copyto!(first_order_coeffs, first_order_coeffs[1:end-1])
    push!(low_boundary_coefs, original_coeffs[1:end-1])

    for i in 2 : boundary_point_count
        #=  this means that a stencil will suffice ie. we dont't need to worry about the boundary point
            being considered in the low_boundary_coefs
        =#
        if i > mid
            pos=i-1
            push!(low_boundary_coefs, calculate_weights(derivative_order, pos*grid_step, collect(zero(T) : grid_step : (boundary_length-1) * grid_step)))
        else
            pos=i-2
            push!(low_boundary_coefs, append!([zero(T)],calculate_weights(derivative_order, pos*grid_step, collect(zero(T) : grid_step : (boundary_length-1) * grid_step))))
        end
    end

    return l_diff
end


function right_Neumann_BC!(::Type{Val{:LO}},high_boundary_coefs,stencil_length,derivative_order,
                           grid_step::T,boundary_length) where T
    boundary_point_count = stencil_length - div(stencil_length,2) + 1
    flag                 = derivative_order*boundary_point_count%2
    original_coeffs      = zeros(T,boundary_length)
    r_diff               = one(T)
    mid                  = div(stencil_length,2)+1

    # this part is to incorporate the value of first derivative at the right boundary
    first_order_coeffs = calculate_weights(1, (boundary_point_count-1)*grid_step,
                                           collect(zero(T) : grid_step : (boundary_length-1) * grid_step))
    reverse!(first_order_coeffs)
    isodd(flag) ? negate!(first_order_coeffs) : nothing

    copyto!(original_coeffs, calculate_weights(derivative_order, (boundary_point_count-1)*grid_step,
                                             collect(zero(T) : grid_step : (boundary_length-1) * grid_step)))

    reverse!(original_coeffs)
    isodd(flag) ? negate!(original_coeffs) : nothing

    r_diff = original_coeffs[1]/first_order_coeffs[1]
    rmul!(first_order_coeffs, original_coeffs[1]/first_order_coeffs[1])
    # rmul!(original_coeffs, first_order_coeffs[1]/original_coeffs[1])
    @. original_coeffs = original_coeffs - first_order_coeffs
    # copyto!(first_order_coeffs, first_order_coeffs[1:end-1])

    for i in 2 : boundary_point_count
        #=
            this means that a stencil will suffice ie. we dont't need to worry about the boundary point
            being considered in the high_boundary_coefs. Same code for low_boundary_coefs but reversed
            at the end
        =#
        if i > mid
            pos=i-1
            push!(high_boundary_coefs, calculate_weights(derivative_order, pos*grid_step,
                                                         collect(zero(T) : grid_step : (boundary_length-1) * grid_step)))
        else
            pos=i-2
            push!(high_boundary_coefs, append!([zero(T)],
                                               calculate_weights(derivative_order, pos*grid_step,
                                               collect(zero(T) : grid_step : (boundary_length-1) * grid_step))))
        end
    end
    if flag == 1
        negate!(high_boundary_coefs)
    end
    reverse!(high_boundary_coefs)
    push!(high_boundary_coefs, original_coeffs[2:end])
    return r_diff
end


function left_Robin_BC!(::Type{Val{:LO}},low_boundary_coefs,stencil_length,params,
                        derivative_order,grid_step::T,boundary_length,dx) where T
    boundary_point_count = stencil_length - div(stencil_length,2) + 1
    first_order_coeffs   = zeros(T,boundary_length)
    original_coeffs      = zeros(T,boundary_length)
    l_diff               = one(T)
    mid                  = div(stencil_length,2)+1

    # in Robin BC the left boundary has opposite sign by convention
    first_order_coeffs = -params[2]*calculate_weights(1, (0)*grid_step, collect(zero(T) : grid_step : (boundary_length-1)* grid_step))
    first_order_coeffs[1] += dx*params[1]
    original_coeffs =  calculate_weights(derivative_order, (0)*grid_step, collect(zero(T) : grid_step : (boundary_length-1) * grid_step))

    l_diff = original_coeffs[end]/first_order_coeffs[end]
    rmul!(first_order_coeffs, original_coeffs[end]/first_order_coeffs[end])
    # rmul!(original_coeffs, first_order_coeffs[end]/original_coeffs[end])
    @. original_coeffs = original_coeffs - first_order_coeffs
    # copyto!(first_order_coeffs, first_order_coeffs[1:end-1])
    push!(low_boundary_coefs, original_coeffs[1:end-1])

    for i in 2 : boundary_point_count
        #=
            this means that a stencil will suffice ie. we dont't need to worry about the boundary point
            being considered in the low_boundary_coefs
        =#
        if i > mid
            pos=i-1
            push!(low_boundary_coefs, calculate_weights(derivative_order, pos*grid_step, collect(zero(T) : grid_step : (boundary_length-1) * grid_step)))
        else
            pos=i-2
            push!(low_boundary_coefs, append!([zero(T)],calculate_weights(derivative_order, pos*grid_step, collect(zero(T) : grid_step : (boundary_length-1) * grid_step))))
        end
    end

    return l_diff
end


function right_Robin_BC!(::Type{Val{:LO}},high_boundary_coefs,stencil_length,params,
                        derivative_order,grid_step::T,boundary_length,dx) where T
    boundary_point_count = stencil_length - div(stencil_length,2) + 1
    flag                 = derivative_order*boundary_point_count%2
    original_coeffs      = zeros(T,boundary_length)
    r_diff               = one(T)
    mid                  = div(stencil_length,2)+1

    first_order_coeffs = params[2]*calculate_weights(1, (boundary_point_count-1)*grid_step,
                                                     collect(zero(T) : grid_step : (boundary_length-1) * grid_step))
    first_order_coeffs[end] += dx*params[1]
    reverse!(first_order_coeffs)
    isodd(flag) ? negate!(first_order_coeffs) : nothing

    copyto!(original_coeffs, calculate_weights(derivative_order, (boundary_point_count-1)*grid_step,
                                             collect(zero(T) : grid_step : (boundary_length-1) * grid_step)))
    reverse!(original_coeffs)
    isodd(flag) ? negate!(original_coeffs) : nothing

    r_diff = original_coeffs[1]/first_order_coeffs[1]
    rmul!(first_order_coeffs, original_coeffs[1]/first_order_coeffs[1])
    # rmul!(original_coeffs, first_order_coeffs[1]/original_coeffs[1])
    @. original_coeffs = original_coeffs - first_order_coeffs
    # copyto!(first_order_coeffs, first_order_coeffs[1:end-1])

    for i in 2 : boundary_point_count
        #=
            this means that a stencil will suffice ie. we dont't need to worry about the boundary point
            being considered in the high_boundary_coefs. Same code for low_boundary_coefs but reversed
            at the end
        =#
        if i > mid
            pos=i-1
            push!(high_boundary_coefs, calculate_weights(derivative_order, pos*grid_step,
                                                         collect(zero(T) : grid_step : (boundary_length-1) * grid_step)))
        else
            pos=i-2
            push!(high_boundary_coefs, append!([zero(T)],
                                               calculate_weights(derivative_order,pos*grid_step,
                                                                 collect(zero(T) : grid_step : (boundary_length-1) * grid_step))))
        end
    end
    if flag == 1
        negate!(high_boundary_coefs)
    end
    reverse!(high_boundary_coefs)
    push!(high_boundary_coefs, original_coeffs[2:end])
    return r_diff
end

#################################################################################################


(L::DerivativeOperator)(u,p,t) = L*u
(L::DerivativeOperator)(du,u,p,t) = mul!(du,L,u)
get_LBC(::DerivativeOperator{A,B,C,D}) where {A,B,C,D} = C
get_RBC(::DerivativeOperator{A,B,C,D}) where {A,B,C,D} = D

#=
    The Inf opnorm can be calculated easily using the stencil coeffiicents, while other opnorms
    default to compute from the full matrix form.
=#
function LinearAlgebra.opnorm(A::DerivativeOperator{T,S,LBC,RBC}, p::Real=2) where {T,S,LBC,RBC}
    if p == Inf && LBC in [:Dirichlet0, :Neumann0, :periodic] && RBC in [:Dirichlet0, :Neumann0, :periodic]
        sum(abs.(A.stencil_coefs)) / A.dx^A.derivative_order
    else
        opnorm(convert(Array,A), p)
    end
end
