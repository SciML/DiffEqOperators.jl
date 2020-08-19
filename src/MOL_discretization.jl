#using Reduce

# Method of lines discretization scheme
struct MOLFiniteDifference{T} <: DiffEqBase.AbstractDiscretization
  dxs::T
  order::Int
end
MOLFiniteDifference(args...;order=2) = MOLFiniteDifference(args,order)

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
            if isequal(bcs[i].lhs.args[1],tdomain.lower) # u(t=0,x) 
                lhs_deriv_depvars_bcs[var][1] = Expr(bcs[i].rhs)
            elseif isequal(bcs[i].lhs.args[2],domain.lower) # u(t,x=x_init)
                lhs_deriv_depvars_bcs[var][2] = :(var=$(bcs[i].rhs.value))
            elseif isequal(bcs[i].lhs.args[2],domain.upper) # u(t,x=x_final)
                lhs_deriv_depvars_bcs[var][3] = :(var=$(bcs[i].rhs.value))
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
function discretize_2(input,grade,order,dx,lhs_deriv_depvars,lhs_nonderiv_depvars)
    if input isa ModelingToolkit.Constant
        return :($input.value)
    elseif input isa Operation
        if input.op isa Variable
            if haskey(lhs_nonderiv_depvars,input.op)
                x = lhs_nonderiv_depvars[input.op]
                if x isa ModelingToolkit.Constant
                    expr = :($x.value)
                else
                    expr = Expr(x)
                    expr = :(x=i*$dx;eval($expr))
                end
            elseif grade == 1
                # TODO: the discretization order should not be the same for
                #       first derivatives and second derivarives
                j = findfirst(x->x==input.op, lhs_deriv_depvars)
                expr = :((u[i,$j]-u[i-1,$j])/$dx)
            elseif grade == 2
                j = findfirst(x->x==input.op, lhs_deriv_depvars)
                expr = :((u[i+1,$j]-2.0*u[i,$j]+u[i-1,$j])/($dx*$dx))
            else
                j = findfirst(x->x==input.op, lhs_deriv_depvars)
                expr = :(u[i,$j])
            end
            return expr
        elseif input.op isa Differential
            grade += 1
            return discretize_2(input.args[1],grade,order,dx,lhs_deriv_depvars,lhs_nonderiv_depvars)
        elseif input.op isa typeof(*)
            expr1 = discretize_2(input.args[1],grade,order,dx,lhs_deriv_depvars,lhs_nonderiv_depvars)
            expr2 = discretize_2(input.args[2],grade,order,dx,lhs_deriv_depvars,lhs_nonderiv_depvars)
            return Expr(:call,:*,expr1,expr2)
        elseif input.op isa typeof(/)
            expr1 = discretize_2(input.args[1],grade,order,dx,lhs_deriv_depvars,lhs_nonderiv_depvars)
            expr2 = discretize_2(input.args[2],grade,order,dx,lhs_deriv_depvars,lhs_nonderiv_depvars)
            return Expr(:call,:/,expr1,expr2)
        elseif input.op isa typeof(-)
            if size(input.args,1) == 2
                expr1 = discretize_2(input.args[1],grade,order,dx,lhs_deriv_depvars,lhs_nonderiv_depvars)
                expr2 = discretize_2(input.args[2],grade,order,dx,lhs_deriv_depvars,lhs_nonderiv_depvars)
                return Expr(:call,:-,expr1,expr2)
            else #if size(input.args,1) == 1
                expr1 = discretize_2(input.args[1],grade,order,dx,lhs_deriv_depvars,lhs_nonderiv_depvars)
                return Expr(:call,:*,:(-1),expr1)
            end
        elseif input.op isa typeof(+)
            if size(input.args,1) == 2
                expr1 = discretize_2(input.args[1],grade,order,dx,lhs_deriv_depvars,lhs_nonderiv_depvars)
                expr2 = discretize_2(input.args[2],grade,order,dx,lhs_deriv_depvars,lhs_nonderiv_depvars)
                return Expr(:call,:+,expr1,expr2)
            else #if size(input.args,1) == 1
                expr1 = discretize_2(input.args[1],grade,order,dx,lhs_deriv_depvars,lhs_nonderiv_depvars)
                return Expr(expr1)
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

    # TODO: specify order for each derivative
    order = discretization.order

    ### Calculate discretization expression ####################################
    # The discretization is an expression which is then evaluated
    # in the ODE function (f)

    # TODO: improve the code below using index arrays instead of Dicts?
    lhs_nonderiv_depvars = Dict()
    lhs_deriv_depvars = Dict()
    discretization = Dict()
    # if there is only one equation
    if pdesys.eq isa Equation
        var = pdesys.eq.lhs.args[1].op
        discretization[var] = discretize_2( pdesys.eq.rhs,0,order,dx[1],
                                            [var],Dict())

    # if there are many equations (pdesys.eq isa Array)
    else
        # Store 'non-derived' dependent variables (e.g. v(t,x)=t*x)
        # and 'derived' dependent variables (e.g. Dt(u(t,x)))
        n_eqs = size(pdesys.eq,1)
        for i = 1:n_eqs
            input = pdesys.eq[i].lhs
            if input.op isa Variable
                var = input.op
                lhs_nonderiv_depvars[var] = pdesys.eq[i].rhs
            else #var isa Differential
                var = input.args[1].op
                lhs_deriv_depvars[var] = pdesys.eq[i].rhs
            end
        end

        # Calc. coeff. matrix for each differential equation
        lhs_deriv_depvars_arr = collect(keys(lhs_deriv_depvars))
        for (var,rhs) in lhs_deriv_depvars
            discretization[var] = discretize_2( rhs,0,order,dx[1],
                                                lhs_deriv_depvars_arr,
                                                lhs_nonderiv_depvars)
        end
    end

    ### Get boundary conditions ################################################
    # TODO: generalize to N equations
    lhs_deriv_depvars_bcs = get_bcs(pdesys.bcs,tdomain,domain[1])
    t = 0.0
    interior = domain[1].lower+dx[1]:dx[1]:domain[1].upper-dx[1]
    u0 = Array{Float64}(undef,length(interior),length(discretization))
    Q = Array{RobinBC}(undef,length(discretization))
    
    i = 1
    for var in keys(discretization)
        bcs = lhs_deriv_depvars_bcs[var]

        g = eval(:((x,t) -> $(bcs[1])))
        u0[:,i] = @eval $g.($interior,$t)
        
        u_x0 = eval(bcs[2])
        u_x1 = eval(bcs[3])
        Q[i] = DirichletBC(u_x0,u_x1)

        i = i+1

    end


    Qu = Array{Float64}(undef,xx[1]+2,length(discretization))

    ### Define the discretized PDE as an ODE function ##########################
    function f(du,u,p,t)
        for j in 1:length(discretization)
            Qu[:,j] = Q[j]*u[:,j]            
        end
        j = 1
        for (var,disc) in discretization
            g = eval(:((u,t,i) -> $disc))
            for i = 1:xx[1]
                du[i,j] = @eval $g($(Qu),$t,$(i+1))
            end
            j = j+1
        end
    end

    # Return problem ###########################################################
    return PDEProblem(ODEProblem(f,u0,(tdomain.lower,tdomain.upper),nothing),Q,X)
end
