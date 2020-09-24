#using Reduce

# Method of lines discretization scheme
struct MOLFiniteDifference{T} <: DiffEqBase.AbstractDiscretization
    dxs::T
    order::Int
    MOLFiniteDifference(args...;order=2) = new{typeof(args[1])}(args[1],order)
end

# Get boundary conditions from an array
function get_bcs(bcs,tdomain,domain)
    lhs_deriv_depvars_bcs = Dict()
    n = size(bcs,1)
    for i = 1:n
        var = bcs[i].lhs.op
        if var isa Variable
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
            if bcs[i].rhs isa ModelingToolkit.Constant
                lhs_deriv_depvars_bcs[var][j] = :(var=$(bcs[i].rhs.value))
            else    
                lhs_deriv_depvars_bcs[var][j] = Expr(bcs[i].rhs)
            end
        end
    end
    return lhs_deriv_depvars_bcs
end


# Recursively traverses the input expression (rhs), replacing derivatives by
# finite difference schemes. It returns a time dependent expression (expr)
# that will be evaluated in the "f" ODE function (in DiffEqBase.discretize),
# Note: 'non-derived' dependent variables are inserted into the diff. equations
#       E.g. Dx(u(t,x))=v(t,x)*Dx(u(t,x)), v(t,x)=t*x
#            =>  Dx(u(t,x))=t*x*Dx(u(t,x))
                            
function discretize_2(input,deriv_order,approx_order,dx,X,len,index)
    if input isa ModelingToolkit.Constant
        return :($(input.value)) 
    elseif input isa Operation
        if input.op isa Variable
            expr = :(0.0)
            if !haskey(index,input.op) # ind. var.
                expr = :($X)
            else # dep. var.
                j = index[input.op]
                if deriv_order == 0
                    expr = :(u[:,$j])
                elseif deriv_order == 1
                    approx_order = 1
                    L = UpwindDifference(deriv_order,approx_order,dx[1],len,-1)
                    expr = :(-1*($L*Q[$j]*u[:,$j]))
                elseif deriv_order == 2
                    L = CenteredDifference(deriv_order,approx_order,dx[1],len)
                    expr = :($L*Q[$j]*u[:,$j])
                end
            end
            return expr
        elseif input.op isa Differential
            return discretize_2(input.args[1],deriv_order+1,approx_order,dx,X,len,index)
        else
            if size(input.args,1) == 1
                aux = discretize_2(input.args[1],deriv_order,approx_order,dx,X,len,index)
                return :(broadcast($(input.op), $aux))
            else
                aux_1 = discretize_2(input.args[1],deriv_order,approx_order,dx,X,len,index)
                aux_2 = discretize_2(input.args[2],deriv_order,approx_order,dx,X,len,index)
                return :(broadcast($(input.op), $aux_1, $aux_2))
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
    #                E.g. du/dt = d2u/dx2 + d2u/dy2 + f(t,x,y)
    #        1.b) Stationary
    #                There is more than one indep. variable, 't' is not included
    #                E.g. 0 = d2u/dx2 + d2u/dy2 + f(x,y)
    #   2) ODE System
    #        't' is the only independent variable
    #        The ODESystem is packed inside a PDESystem
    #        E.g. du/dt = f(t)
    #
    #   Note: regarding input format, lhs must be "du/dt" or "0".
    #

    # The following code deals with 1.a case for 1D,
    # i.e. only considering 't' and 'x'

    ### Get domains (typically temporal and spatial) ###########################
    # TODO: here it is assumed that the time domain is the first in the array.
    #       It can be in any order.

    tdomain = pdesys.domain[1].domain
    @assert tdomain isa IntervalDomain

    no_iv = size(pdesys.domain,1)
    domain = []
    dx = []
    X = []
    xx = []
    for i = 1:no_iv-1
        domain = vcat(domain,pdesys.domain[i+1].domain)
        dx = vcat(dx,discretization.dxs)
        X = vcat(X,domain[i].lower:dx[i]:domain[i].upper)
        xx = vcat(xx,size(X,1)-2)
    end
    interior = domain[1].lower+dx[1]:dx[1]:domain[1].upper-dx[1]

    # TODO: specify order for each derivative
    approx_order = discretization.order

    ### Calculate discretization expression ####################################
    # The discretization is an expression which is then evaluated
    # in the ODE function (f)

    discretization = Dict()
    lhs_deriv_depvars = Dict()
    index = Dict()

    # if there is only one equation
    if pdesys.eq isa Equation
        eqs = [pdesys.eq]
    else
        eqs = pdesys.eq
    end

    n_eqs = size(eqs,1)
    for j = 1:n_eqs
        input = eqs[j].lhs
        if input.op isa Variable
            var = input.op
        else #var isa Differential
            var = input.args[1].op
            lhs_deriv_depvars[var] = j
        end
        index[var] = j
    end
    for (var,j) in index
        aux = discretize_2( eqs[j].rhs,0,approx_order,
                            dx[1],interior,xx[1],index)
        # TODO: is there a better way to convert an Expr into a Function?
        discretization[var] = @eval (Q,u,t) -> $aux
    end
    
    ### Get boundary conditions ################################################
    # TODO: extend to Neumann BCs and Robin BCs
    lhs_deriv_depvars_bcs = get_bcs(pdesys.bcs,tdomain,domain[1])
    t = 0.0
    u_t0 = Array{Float64}(undef,length(interior),length(discretization))
    u_x0 = Array{Any}(undef,length(discretization))
    u_x1 = Array{Any}(undef,length(discretization))
    Q = Array{RobinBC}(undef,length(discretization))
    
    for var in keys(discretization)
        j = index[var]
        bcs = lhs_deriv_depvars_bcs[var]

        g = eval(:((x,t) -> $(bcs[1])))
        u_t0[:,j] = @eval $g.($interior,$t)

        u_x0[j] = @eval (x,t) -> $(bcs[2])
        u_x1[j] = @eval (x,t) -> $(bcs[3])

        a = Base.invokelatest(u_x0[j],X[1],0.0)
        b = Base.invokelatest(u_x1[j],last(X),0.0)
        Q[j] = DirichletBC(a,b)

    end

    ### Define the discretized PDE as an ODE function ##########################

    function f(du,u,p,t)
        
        for j in 1:length(discretization)
            a = Base.invokelatest(u_x0[j],X[1],t)
            b = Base.invokelatest(u_x1[j],last(X),t)
            Q[j] = DirichletBC(a,b)
        end

        for (var,disc) in discretization
            j = index[var]
            res = Base.invokelatest(disc,Q,u,t)
            if haskey(lhs_deriv_depvars,var)
                du[:,j] = res
            else
                u[:,j] .= res
            end
        end

    end

    # Return problem ###########################################################
    return PDEProblem(ODEProblem(f,u_t0,(tdomain.lower,tdomain.upper),nothing),Q,X)
end

