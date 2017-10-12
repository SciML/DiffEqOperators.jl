struct UpwindOperator{T<:Real,S<:SVector,LBC,RBC} <: AbstractDerivativeOperator{T}
    derivative_order    :: Int
    approximation_order :: Int
    dx                  :: T
    dimension           :: Int
    directions          :: Ref{BitArray{1}}
    stencil_length      :: Int
    up_stencil_coefs    :: S
    down_stencil_coefs  :: S
    boundary_point_count:: Tuple{Int,Int}
    boundary_length     :: Tuple{Int,Int}
    low_boundary_coefs  :: Ref{Vector{Vector{T}}}
    high_boundary_coefs :: Ref{Vector{Vector{T}}}
    boundary_condition  :: Ref{Tuple{Tuple{T,T,Any},Tuple{T,T,Any}}}
    t                   :: Ref{Int}

    Base.@pure function UpwindOperator{T,S,LBC,RBC}(derivative_order::Int, approximation_order::Int, dx::T,
                                            dimension::Int, directions::BitArray{1}, bndry_fn) where {T<:Real,S<:SVector,LBC,RBC}
        dimension            = dimension
        dx                   = dx
        directions           = directions
        stencil_length       = derivative_order + approximation_order
        bl                   = derivative_order + approximation_order

        boundary_length      = (bl,bl)
        grid_step            = one(T)
        low_boundary_coefs   = Vector{T}[]
        high_boundary_coefs  = Vector{T}[]

        #=
            We are implementing biased Upwind Operators which use a point from the other direction
            also to ensure a more stable solution.
            http://ac.els-cdn.com/S0378475401002889/1-s2.0-S0378475401002889-main.pdf?_tid=534e0818-8b0b-11e7-8b12-00000aab0f01&acdnat=1503826821_3bf9422abe7aa2d3613c5b644b6e258f - page 11
        =#
        up_stencil_coefs        = convert(SVector{stencil_length, T}, calculate_weights(derivative_order,convert(T,(stencil_length+1)%2),
                                          grid_step .* collect(zero(T) : grid_step : stencil_length-1)))

        down_stencil_coefs      = reverse(up_stencil_coefs)

        # no need to flip the sign of downwind operator, doesn't converge that way
        # derivative_order%2 == 1 ? negate!(down_stencil_coefs) : nothing

        bpc_array            = [length(up_stencil_coefs)-1,length(down_stencil_coefs)-1]

        left_bndry = initialize_left_boundary!(Val{:UO},low_boundary_coefs,bndry_fn,derivative_order,grid_step,
                                                   bl,bpc_array,directions,dx,LBC)
        right_bndry = initialize_right_boundary!(Val{:UO},high_boundary_coefs,bndry_fn,derivative_order,grid_step,
                                                     bl,bpc_array,directions,dx,RBC)

        boundary_condition = (left_bndry, right_bndry)
        boundary_point_count = (bpc_array[1],bpc_array[2])
        t = 0

        new(derivative_order, approximation_order, dx, dimension, directions,
            stencil_length,
            up_stencil_coefs,
            down_stencil_coefs,
            boundary_point_count,
            boundary_length,
            low_boundary_coefs,
            high_boundary_coefs,
            boundary_condition,
            t
            )
    end
    UpwindOperator{T}(dorder::Int,aorder::Int,dx::T,dim::Int,direction::BitArray{1},LBC::Symbol,RBC::Symbol;BC=(zero(T),zero(T))) where {T<:Real} = UpwindOperator{T, SVector{dorder+aorder,T}, LBC, RBC}(dorder, aorder, dx, dim, direction, BC)
end


(L::UpwindOperator)(t,u) = L*u
(L::UpwindOperator)(t,u,du) = A_mul_B!(du,L,u)

function update_coefficients!(A::UpwindOperator{T,S,LBC,RBC};BC=nothing, directions=nothing) where {T<:Real,S<:SVector,RBC,LBC}
    if BC != nothing
        LBC == :Robin ? (length(BC[1])==3 || error("Enter the new left boundary condition as a 1-tuple")) :
                        (length(BC[1])==1 || error("Robin BC needs a 3-tuple for left boundary condition"))

        RBC == :Robin ? length(BC[2])==3 || error("Enter the new right boundary condition as a 1-tuple") :
                        length(BC[2])==1 || error("Robin BC needs a 3-tuple for right boundary condition")

        left_bndry = initialize_left_boundary!(A.low_boundary_coefs[],A.stencil_coefs,BC,
                                               A.derivative_order,one(T),A.boundary_length,A.dx,A.directions,LBC)

        right_bndry = initialize_right_boundary!(A.high_boundary_coefs[],A.stencil_coefs,BC,
                                                 A.derivative_order,one(T),A.boundary_length,A.dx,A.directions,RBC)

        boundary_condition = (left_bndry, right_bndry)
        A.boundary_condition[] = boundary_condition
    end
    if directions != nothing
        A.directions[] = directions
    end
end


function initialize_left_boundary!(::Type{Val{:UO}},low_boundary_coefs,BC,derivative_order,grid_step::T,
                                   boundary_length,boundary_point_count,directions,dx,LBC) where T
    approximation_order = boundary_length - derivative_order
    up_stencil_length = boundary_length

    if LBC == :None
        # up_stencil_length%2 == 1 ? boundary_point_count[1] = 1 : boundary_point_count[1] = 0
        return (zero(T),zero(T),left_None_BC!(Val{:UO},low_boundary_coefs,up_stencil_length,derivative_order,
                                              grid_step,boundary_length)*BC[1]*dx)

    elseif LBC == :Neumann
        return (zero(T),one(T),left_Neumann_BC!(Val{:UO},low_boundary_coefs,up_stencil_length,derivative_order,
                                                grid_step,boundary_length)*BC[1]*dx)

    elseif LBC == :Robin
        return (BC[1][1],-BC[1][2],left_Robin_BC!(Val{:UO},low_boundary_coefs,up_stencil_length,
                                                   BC[1],derivative_order,grid_step,
                                                   boundary_length,dx)*BC[1][3]*dx)

    elseif LBC == :nothing
        return (zero(T),zero(T),left_nothing_BC!(Val{:UO},low_boundary_coefs,up_stencil_length,derivative_order,
                                              grid_step,boundary_length)*BC[1]*dx)

    elseif LBC == :Dirichlet0
        return (one(T),zero(T),zero(T)*BC[1])

    elseif LBC == :Dirichlet
        return (one(T),zero(T),one(T)*BC[1])

    elseif LBC == :Neumann0
        return (zero(T),one(T),zero(T))

    elseif LBC == :periodic
        return (zero(T),zero(T),zero(T))

    else
        error("Unrecognized Boundary Type!")
    end
end

# well it says that we have to use somewhere inside the function definition
function initialize_right_boundary!(::Type{Val{:UO}},high_boundary_coefs,BC,derivative_order,grid_step::T,
                                    boundary_length,boundary_point_count,directions,dx,RBC) where T
    approximation_order = boundary_length - derivative_order
    down_stencil_length = boundary_length

    if RBC == :None
        # down_stencil_length%2 == 1 ? boundary_point_count[2] = 1 : boundary_point_count[2] = 0
        return (zero(T),zero(T),right_None_BC!(Val{:UO},high_boundary_coefs,down_stencil_length,derivative_order,
                               grid_step,boundary_length)*BC[2]*dx)

    elseif RBC == :Neumann
        return (zero(T),one(T),right_Neumann_BC!(Val{:UO},high_boundary_coefs,down_stencil_length,derivative_order,
                                  grid_step,boundary_length)*BC[2]*dx)

    elseif RBC == :Robin
        return (BC[2][1],BC[2][2],right_Robin_BC!(Val{:UO},high_boundary_coefs,down_stencil_length,
                                                    BC[2],derivative_order,grid_step,
                                                    boundary_length,dx)*BC[2][3]*dx)

    elseif RBC == :nothing
        return (zero(T),zero(T),right_nothing_BC!(Val{:UO},high_boundary_coefs,down_stencil_length,derivative_order,
                               grid_step,boundary_length)*BC[2]*dx)

    elseif RBC == :Dirichlet0
        return (one(T),zero(T),zero(T)*BC[2])

    elseif RBC == :Dirichlet
        return (one(T),zero(T),one(T)*BC[2])

    elseif RBC == :Neumann0
        return (zero(T),one(T),zero(T))

    elseif RBC == :periodic
        return (zero(T),zero(T),zero(T))

    else
        error("Unrecognized Boundary Type!")
    end
end


function left_None_BC!(::Type{Val{:UO}},low_boundary_coefs,up_stencil_length,derivative_order,
                       grid_step::T,boundary_length) where T
    # Fixes the problem excessive boundary points
    boundary_point_count = up_stencil_length
    l_diff               = zero(T)

    for i in  1:boundary_point_count
        # One-sided stencils require more points for same approximation order
        # TODO: I don't know if this is the correct stencil length for i > 1?
        push!(low_boundary_coefs, negate!(calculate_weights(derivative_order, (i-1)*grid_step, collect(zero(T) : grid_step : (boundary_length-1)*grid_step))))
    end
    return l_diff
end


function right_None_BC!(::Type{Val{:UO}},high_boundary_coefs,down_stencil_length,derivative_order,
                        grid_step::T,boundary_length) where T
    boundary_point_count = down_stencil_length
    high_temp            = zeros(T,boundary_length)
    aorder               = boundary_length - 1
    r_diff               = zero(T)

    for i in 1 : boundary_point_count
        push!(high_boundary_coefs, negate!(calculate_weights(derivative_order, -(i-1)*grid_step, reverse(collect(zero(T) : -grid_step : -(boundary_length-1)*grid_step)))))
    end
    return r_diff
end


function left_nothing_BC!(::Type{Val{:UO}},low_boundary_coefs,up_stencil_length,derivative_order,
                       grid_step::T,boundary_length) where T
    # Fixes the problem excessive boundary points
    boundary_point_count = up_stencil_length
    l_diff               = zero(T)

    push!(low_boundary_coefs, negate!(calculate_weights(derivative_order, (0)*grid_step, collect(zero(T) : grid_step : (boundary_length-1)*grid_step))))
    return l_diff
end


function right_nothing_BC!(::Type{Val{:UO}},high_boundary_coefs,down_stencil_length,derivative_order,
                        grid_step::T,boundary_length) where T
    boundary_point_count = down_stencil_length
    r_diff               = zero(T)

    push!(high_boundary_coefs, negate!(calculate_weights(derivative_order, -(0)*grid_step, reverse(collect(zero(T) : -grid_step : -(boundary_length-1)*grid_step)))))
    return r_diff
end
