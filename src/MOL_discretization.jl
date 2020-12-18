using ModelingToolkit: operation
# Method of lines discretization scheme
struct MOLFiniteDifference{T} <: DiffEqBase.AbstractDiscretization
    dxs::T
    upwind_order::Int
    centered_order::Int
end

# Constructors. If no order is specified, both upwind and centered differences will be 2nd order
MOLFiniteDifference(dxs::T, order) where T =
    MOLFiniteDifference{T}(dxs, order, order)

function MOLFiniteDifference(dxs::T; order = 2, upwind_order = nothing,
                            centered_order = nothing) where T
    MOLFiniteDifference(
        dxs,
        upwind_order === nothing ? order : upwind_order,
        centered_order === nothing ? order : centered_order
    )
end

function throw_bc_err(bc)
    throw(BoundaryConditionError(
        "Could not read boundary condition '$(bc.lhs) ~ $(bc.rhs)'"
    ))
end
# Get boundary conditions from an array
# The returned boundary conditions will be the three coefficients (α, β, γ)
# for a RobinBC of the form α*u(0) + β*u'(0) = γ
# These are created automatically from the form of the boundary condition that is
# passed, i.e.
# u(t, 0) ~ γ                           ---> (1, 0, γ)
# Dx(u(t, 0)) ~ γ                       ---> (0, 1, γ)
# α * u(t, 0)) + β * Dx(u(t, 0)) ~ γ    ---> (α, β, γ)
# In general, all of α, β, and γ could be Expr (i.e. functions of t)
function get_bcs(bcs,tdomain,domain)
    lhs_deriv_depvars_bcs = Dict()
    num_bcs = size(bcs,1)
    for i = 1:num_bcs
        lhs = bcs[i].lhs
        # Extract the variable from the lhs
        if operation(lhs) isa Sym
            # Dirichlet boundary condition
            var = nameof(operation(lhs))
            α = 1.0
            β = 0.0
            bc_args = lhs.args
        elseif operation(lhs) isa Differential
            # Neumann boundary condition
            # Check that we don't have a second-order derivative in the
            # boundary condition, by checking that the argument is a Sym
            @assert operation(lhs.args[1]) isa Sym throw_bc_err(bcs[i])
            var = nameof(operation(lhs.args[1]))
            α = 0.0
            β = 1.0
            bc_args = lhs.args[1].args
        elseif operation(lhs) isa typeof(+)
            # Robin boundary condition
            lhs_l, lhs_r = lhs.args
            # Left side of the expression should be Sym or α * Sym
            if operation(lhs_l) isa Sym
                α = 1.0
                var_l = nameof(operation(lhs_l))
                bc_args_l = lhs_l.args
            elseif operation(lhs_l) isa typeof(*)
                α = lhs_l.args[1]
                # Convert α to a Float64 if it is an Int, leave unchanged otherwise
                α = α isa Int ? Float64(α) : α
                @assert operation(lhs_l.args[2]) isa Sym throw_bc_err(bcs[i])
                var_l = nameof(operation(lhs_l.args[2]))
                bc_args_l = lhs_l.args[2].args
            else
                throw_bc_err(bcs[i])
            end
            # Right side of the expression should be Differential or β * Differential
            if operation(lhs_r) isa Differential
                # Check that we don't have a second-order derivative in the
                # boundary condition
                @assert operation(lhs_r.args[1]) isa Sym throw_bc_err(bcs[i])
                β = 1.0
                var_r = nameof(operation(lhs_r.args[1]))
                bc_args_r = lhs_r.args[1].args
            elseif operation(lhs_r) isa typeof(*)
                β = lhs_r.args[1]
                # Convert β to a Float64 if it is an Int, leave unchanged otherwise
                β = β isa Int ? Float64(β) : β
                # Check that the bc is a derivative
                @assert operation(lhs_r.args[2]) isa Differential throw_bc_err(bcs[i])
                # But not second order (argument should be a Sym)
                @assert operation(lhs_r.args[2].args[1]) isa Sym throw_bc_err(bcs[i])
                var_r = nameof(operation(lhs_r.args[2].args[1]))
                bc_args_r = lhs_r.args[2].args[1].args
            else
                throw_bc_err(bcs[i])
            end
            # Check var and bc_args are the same in lhs and rhs, and if so assign 
            # the unique value
            @assert var_l == var_r throw(BoundaryConditionError(
                "mismatched variables '$var_l' and '$var_r' "
                * "in Robin BC '$(bcs[i].lhs) ~ $(bcs[i].rhs)'"
            ))
            var = var_l
            @assert bc_args_l == bc_args_r throw(BoundaryConditionError(
                "mismatched args $bc_args_l and $bc_args_r "
                * "in Robin BC '$(bcs[i].lhs) ~ $(bcs[i].rhs)'"
            ))
            bc_args = bc_args_l
        else
            throw_bc_err(bcs[i])
        end
        if !haskey(lhs_deriv_depvars_bcs,var)
            # Initialize dict of boundary conditions for this variable
            lhs_deriv_depvars_bcs[var] = Dict()
        end
        # Create key
        if isequal(bc_args[1],tdomain.lower) # u(t=0,x)
            key = "ic"
        elseif isequal(bc_args[2],domain.lower) # u(t,x=x_init)
            key = "left bc"
        elseif isequal(bc_args[2],domain.upper) # u(t,x=x_final)
            key = "right bc"
        else
            throw(BoundaryConditionError(
                "Boundary condition '$(bcs[i].lhs) ~ $(bcs[i].rhs)' could not be read. "
                * "BCs should be applied at t=$(tdomain.lower), "
                * "x=$(domain.lower), or x=$(domain.upper)"
            ))
        end
        # Create value
        γ = toexpr(bcs[i].rhs)
        # Assign
        if key == "ic"
            # Initial conditions always take the form u(0, x) ~ γ
            lhs_deriv_depvars_bcs[var][key] = γ
        else
            # Boundary conditions can be more general
            lhs_deriv_depvars_bcs[var][key] = (toexpr(α), toexpr(β), γ)
        end
    end
    # Check each variable got all boundary conditions
    for var in keys(lhs_deriv_depvars_bcs)
        for key in ["ic", "left bc", "right bc"]
            if !haskey(lhs_deriv_depvars_bcs[var], key)
                throw(BoundaryConditionError("missing $key for $var"))
            end
        end
    end
    return lhs_deriv_depvars_bcs
end


# Recursively traverses the input expression (rhs), replacing derivatives by
# finite difference schemes. It returns a time dependent expression (expr)
# that will be evaluated in the "f" ODE function (in DiffEqBase.discretize),
# Note: 'non-derived' dependent variables are inserted into the differential equations
#       E.g., Dx(u(t,x))=v(t,x)*Dx(u(t,x)), v(t,x)=t*x
#            =>  Dx(u(t,x))=t*x*Dx(u(t,x))

function discretize_2(input,deriv_order,upwind_order,centered_order,dx,X,len_of_indep_vars,
                      deriv_var,dep_var_idx,indep_var_idx)
    if !(input isa ModelingToolkit.Symbolic)
        return :($(input))
    else
        if input isa Sym || (input isa Term && operation(input) isa Sym)
            expr = :(0.0)
            var = nameof(input isa Sym ? input : operation(input))
            if haskey(indep_var_idx,var) # ind. var.
                if var != :(t)
                    i = indep_var_idx[var]
                    expr = :($X[$i][2:$len_of_indep_vars[$i]-1])
                else
                    expr = :(t)
                end
            else # dep. var.
                # TODO: time and cross derivatives terms
                i = indep_var_idx[deriv_var[1]]
                j = dep_var_idx[var]
                if deriv_order == 0
                    expr = :(u[:,$j])
                elseif deriv_order == 1
                    # TODO: approx_order and forward/backward should be
                    #       input parameters of each derivative
                    L = UpwindDifference(deriv_order,upwind_order,dx[i],len_of_indep_vars[i]-2,-1)
                    expr = :(-1*($L*Q[$j]*u[:,$j]))
                elseif deriv_order == 2
                    L = CenteredDifference(deriv_order,centered_order,dx[i],len_of_indep_vars[i]-2)
                    expr = :($L*Q[$j]*u[:,$j])
                end
            end
            return expr
        elseif input isa Term && operation(input) isa Differential
            var = nameof(input.op.x)
            push!(deriv_var,var)
            return discretize_2(input.args[1],deriv_order+1,upwind_order,centered_order,dx,X,
                                len_of_indep_vars,deriv_var,dep_var_idx,indep_var_idx)
        else
            name = nameof(operation(input))
            if size(input.args,1) == 1
                aux = discretize_2(input.args[1],deriv_order,upwind_order,centered_order,dx,X,
                                   len_of_indep_vars,deriv_var,dep_var_idx,indep_var_idx)
                return :(broadcast($name, $aux))
            else
                aux_1 = discretize_2(input.args[1],deriv_order,upwind_order,centered_order,dx,X,
                                     len_of_indep_vars,deriv_var,dep_var_idx,indep_var_idx)
                aux_2 = discretize_2(input.args[2],deriv_order,upwind_order,centered_order,dx,X,
                                     len_of_indep_vars,deriv_var,dep_var_idx,indep_var_idx)
                return :(broadcast($name, $aux_1, $aux_2))
            end
        end
    end
end

# Convert a PDE problem into an ODE problem
function DiffEqBase.discretize(pdesys::PDESystem,discretization::MOLFiniteDifference)

    # TODO: discretize the following cases
    #
    #   1) PDE System
    #        1.a) Transient
    #                There is more than one indep. variable, including  't'
    #                E.g., du/dt = d2u/dx2 + d2u/dy2 + f(t,x,y)
    #        1.b) Stationary
    #                There is more than one indep. variable, 't' is not included
    #                E.g., 0 = d2u/dx2 + d2u/dy2 + f(x,y)
    #   2) ODE System
    #        't' is the only independent variable
    #        The ODESystem is packed inside a PDESystem
    #        E.g., du/dt = f(t)
    #
    #   Note: regarding input format, lhs must be "du/dt" or "0".
    #

    # The following code deals with 1.a case for 1-D,
    # i.e., only considering 't' and 'x'.


    ### Declare and define independent-variable data structures ###############

    tdomain = 0.0
    indep_var_idx = Dict()
    num_indep_vars = size(pdesys.domain,1)
    domain = Array{Any}(undef,num_indep_vars)
    dx = Array{Any}(undef,num_indep_vars)
    X = Array{Any}(undef,num_indep_vars)
    len_of_indep_vars = Array{Any}(undef,num_indep_vars)
    k = 0
    for i = 1:num_indep_vars
        var = nameof(pdesys.domain[i].variables)
        indep_var_idx[var] = i
        domain[i] = pdesys.domain[i].domain
        if var != :(t)
            dx[i] = discretization.dxs[i-k]
            X[i] = domain[i].lower:dx[i]:domain[i].upper
            len_of_indep_vars[i] = size(X[i],1)
        else
            dx[i] = 0.0
            X[i] = 0.0
            len_of_indep_vars[i] = 0.0
            tdomain = pdesys.domain[1].domain
            @assert tdomain isa IntervalDomain
            k = 1
        end
    end

    ### Declare and define dependent-variable data structures #################

    # TODO: specify order for each derivative
    upwind_order = discretization.upwind_order
    centered_order = discretization.centered_order

    lhs_deriv_depvars = Dict()
    dep_var_idx = Dict()
    dep_var_disc = Dict() # expressions evaluated in the ODE function (f)

    # if there is only one equation
    if pdesys.eq isa Equation
        eqs = [pdesys.eq]
    else
        eqs = pdesys.eq
    end
    num_dep_vars = size(eqs,1)
    for j = 1:num_dep_vars
        input = eqs[j].lhs
        op = operation(input)
        if op isa Sym
            var = nameof(op)
        else #var isa Differential
            var = nameof(operation(input.args[1]))
            lhs_deriv_depvars[var] = j
        end
        dep_var_idx[var] = j
    end
    for (var,j) in dep_var_idx
        aux = discretize_2( eqs[j].rhs,0,upwind_order,centered_order,dx,X,len_of_indep_vars,
                            [],dep_var_idx,indep_var_idx)
        dep_var_disc[var] = @RuntimeGeneratedFunction(:((Q,u,t) -> $aux))
    end

    ### Declare and define boundary conditions ################################

    # TODO: extend to Neumann BCs and Robin BCs
    lhs_deriv_depvars_bcs = get_bcs(pdesys.bcs,tdomain,domain[2])
    u_ic = Array{Float64}(undef,len_of_indep_vars[2]-2,num_dep_vars)
    robin_bc_func = Array{Any}(undef,num_dep_vars)
    Q = Array{RobinBC}(undef,num_dep_vars)

    for var in keys(dep_var_idx)
        j = dep_var_idx[var]
        bcs = lhs_deriv_depvars_bcs[var]

        # Initial condition depends on space but not time
        ic = @RuntimeGeneratedFunction(:(x -> $(bcs["ic"])))
        u_ic[:,j] = ic.(X[2][2:len_of_indep_vars[2]-1])

        # Boundary conditions depend on time and so will be evaluated at t within the
        # ODE function for the discretized PDE (below)
        # We use @RuntimeGeneratedFunction to do this efficiently
        αl, βl, γl = bcs["left bc"]
        αr, βr, γr = bcs["right bc"]
        # Right now there is only one independent variable, so dx is always dx[2]
        # i.e. the spatial variable
        robin_bc_func[j] = @RuntimeGeneratedFunction(:(t -> begin
            RobinBC(($(αl), $(βl), $(γl)), ($(αr), $(βr), $(γr)), $(dx[2]))
        end))
    end

    ### Define the discretized PDE as an ODE function #########################

    function f(du,u,p,t)

        # Boundary conditions can vary with respect to time (but not space)
        for j in 1:num_dep_vars
            Q[j] = robin_bc_func[j](t)
        end

        for (var,disc) in dep_var_disc
            j = dep_var_idx[var]
            res = disc(Q,u,t)
            if haskey(lhs_deriv_depvars,var)
                du[:,j] = res
            else
                u[:,j] .= res
            end
        end

    end

    # Return problem ##########################################################
    # The second entry, robin_bc_func, is stored as the "extrapolation" in the
    # PDEProblem. Since this can in general be time-dependent, it must be evaluated
    # at the correct time when used, e.g.
    #   prob.extrapolation[1](t[i])*sol[:,1,i]
    return PDEProblem(
        ODEProblem(f,u_ic,(tdomain.lower,tdomain.upper),nothing),
        robin_bc_func,
        X
    )
end
