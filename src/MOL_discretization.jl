using ModelingToolkit: operation
# Method of lines discretization scheme
struct MOLFiniteDifference{T} <: DiffEqBase.AbstractDiscretization
    dxs::T
    order::Int
    MOLFiniteDifference(args...;order=2) = new{typeof(args[1])}(args[1],order)
end

# Get boundary conditions from an array
function get_bcs(bcs,tdomain,domain)
    lhs_deriv_depvars_bcs = Dict()
    no_bcs = size(bcs,1)
    for i = 1:no_bcs
        var = operation(bcs[i].lhs)
        if var isa Sym
            var = var.name
            if !haskey(lhs_deriv_depvars_bcs,var)
                lhs_deriv_depvars_bcs[var] = Array{Expr}(undef,3)
            end
            j = 0
            if isequal(bcs[i].lhs.args[1],tdomain.lower) # u(t=0,x)
                j = 1
            elseif isequal(bcs[i].lhs.args[2],domain.lower) # u(t,x=x_init)
                j = 2
            elseif isequal(bcs[i].lhs.args[2],domain.upper) # u(t,x=x_final)
                j = 3
            end
            if !(bcs[i].rhs isa ModelingToolkit.Symbolic)
                lhs_deriv_depvars_bcs[var][j] = :(var=$(bcs[i].rhs))
            else
                lhs_deriv_depvars_bcs[var][j] = toexpr(bcs[i].rhs)
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

function discretize_2(input,deriv_order,approx_order,dx,X,len,
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
                    expr = :($X[$i][2:$len[$i]-1])
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
                    approx_order = 1
                    L = UpwindDifference(deriv_order,approx_order,dx[i],len[i]-2,-1)
                    expr = :(-1*($L*Q[$j]*u[:,$j]))
                elseif deriv_order == 2
                    L = CenteredDifference(deriv_order,approx_order,dx[i],len[i]-2)
                    expr = :($L*Q[$j]*u[:,$j])
                end
            end
            return expr
        elseif input isa Term && operation(input) isa Differential
            var = nameof(input.op.x)
            push!(deriv_var,var)
            return discretize_2(input.args[1],deriv_order+1,approx_order,dx,X,
                                len,deriv_var,dep_var_idx,indep_var_idx)
            pop!(deriv_var,var)
        else
            name = nameof(operation(input))
            if size(input.args,1) == 1
                aux = discretize_2(input.args[1],deriv_order,approx_order,dx,X,
                                   len,deriv_var,dep_var_idx,indep_var_idx)
                return :(broadcast($name, $aux))
            else
                aux_1 = discretize_2(input.args[1],deriv_order,approx_order,dx,X,
                                     len,deriv_var,dep_var_idx,indep_var_idx)
                aux_2 = discretize_2(input.args[2],deriv_order,approx_order,dx,X,
                                     len,deriv_var,dep_var_idx,indep_var_idx)
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
    no_indep_vars = size(pdesys.domain,1)
    domain = Array{Any}(undef,no_indep_vars)
    dx = Array{Any}(undef,no_indep_vars)
    X = Array{Any}(undef,no_indep_vars)
    len = Array{Any}(undef,no_indep_vars)
    k = 0
    for i = 1:no_indep_vars
        var = nameof(pdesys.domain[i].variables)
        indep_var_idx[var] = i
        domain[i] = pdesys.domain[i].domain
        if var != :(t)
            dx[i] = discretization.dxs[i-k]
            X[i] = domain[i].lower:dx[i]:domain[i].upper
            len[i] = size(X[i],1)
        else
            dx[i] = 0.0
            X[i] = 0.0
            len[i] = 0.0
            tdomain = pdesys.domain[1].domain
            @assert tdomain isa IntervalDomain
            k = 1
        end
    end

    ### Declare and define dependent-variable data structures #################

    # TODO: specify order for each derivative
    approx_order = discretization.order

    lhs_deriv_depvars = Dict()
    dep_var_idx = Dict()
    dep_var_disc = Dict() # expressions evaluated in the ODE function (f)

    # if there is only one equation
    if pdesys.eq isa Equation
        eqs = [pdesys.eq]
    else
        eqs = pdesys.eq
    end
    no_dep_vars = size(eqs,1)
    for j = 1:no_dep_vars
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
        aux = discretize_2( eqs[j].rhs,0,approx_order,dx,X,len,
                            [],dep_var_idx,indep_var_idx)
        # TODO: is there a better way to convert an Expr into a Function?
        dep_var_disc[var] = @eval (Q,u,t) -> $aux
    end

    ### Declare and define boundary conditions ################################

    # TODO: extend to Neumann BCs and Robin BCs
    lhs_deriv_depvars_bcs = get_bcs(pdesys.bcs,tdomain,domain[2])
    t = 0.0
    u_t0 = Array{Float64}(undef,len[2]-2,no_dep_vars)
    u_x0 = Array{Any}(undef,no_dep_vars)
    u_x1 = Array{Any}(undef,no_dep_vars)
    Q = Array{RobinBC}(undef,no_dep_vars)

    for var in keys(dep_var_idx)
        j = dep_var_idx[var]
        bcs = lhs_deriv_depvars_bcs[var]

        g = eval(:((x,t) -> $(bcs[1])))
        u_t0[:,j] = @eval $g.($(X[2][2:len[2]-1]),$t)

        u_x0[j] = @eval (x,t) -> $(bcs[2])
        u_x1[j] = @eval (x,t) -> $(bcs[3])

        a = Base.invokelatest(u_x0[j],X[2][1],0.0)
        b = Base.invokelatest(u_x1[j],last(X[2]),0.0)
        Q[j] = DirichletBC(a,b)
    end

    ### Define the discretized PDE as an ODE function #########################

    function f(du,u,p,t)

        # Boundary conditions can vary with respect to time
        for j in 1:no_dep_vars
            a = Base.invokelatest(u_x0[j],X[2][1],t)
            b = Base.invokelatest(u_x1[j],last(X[2]),t)
            Q[j] = DirichletBC(a,b)
        end

        for (var,disc) in dep_var_disc
            j = dep_var_idx[var]
            res = Base.invokelatest(disc,Q,u,t)
            if haskey(lhs_deriv_depvars,var)
                du[:,j] = res
            else
                u[:,j] .= res
            end
        end

    end

    # Return problem ##########################################################
    return PDEProblem(ODEProblem(f,u_t0,(tdomain.lower,tdomain.upper),nothing),Q,X)
end
