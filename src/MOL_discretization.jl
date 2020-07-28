#using Reduce

# Method of lines discretization scheme
struct MOLFiniteDifference{T} <: DiffEqBase.AbstractDiscretization
  dxs::T
  order::Int
end
MOLFiniteDifference(args...;order=2) = MOLFiniteDifference(args,order)

# Get boundary conditions from an array
function get_bcs(bcs,tdomain,domain)
    u_t0 = 0.0
    u_x0 = 0.0
    u_x1 = 0.0
    n = size(bcs,1)
    for i = 1:n
        if bcs[i].lhs.op isa Variable
            if isequal(bcs[i].lhs.args[1],tdomain.lower) # u(t=0,x)
                u_t0 = Expr(bcs[i].rhs)
            elseif isequal(bcs[i].lhs.args[2],domain.lower) # u(t,x=x_init)
                u_x0 = bcs[i].rhs.value
            elseif isequal(bcs[i].lhs.args[2],domain.upper) # u(t,x=x_final)
                u_x1 = bcs[i].rhs.value
            end
        end
    end
    return (u_t0,u_x0,u_x1)
end

# Calculate coefficient matrix of the finite-difference scheme
# Note: 'non-derived' dependent variables are inserted into the diff. equations
#       E.g. Dx(u(t,x))=v(t,x)*Dx(u(t,x)), v(t,x)=t*x
#            =>  Dx(u(t,x))=t*x*Dx(u(t,x))
function calc_coeff_mat(input,iv,grade,order,dx,m,nonderiv_depvars)
    if input isa ModelingToolkit.Constant
        return :($input.value)
    elseif input isa Operation
        if input.op isa Variable
            if haskey(nonderiv_depvars,input.op)
                x = nonderiv_depvars[input.op]
                if x isa ModelingToolkit.Constant
                    L = :($x.value)
                else
                    # TODO: Here, evaluate expression w.r.t space (x,y,z).
                    #       Then, in the "f" ODE function (in DiffEqBase.discretize),
                    #       evaluate expression w.r.t. time (t).
                    expr = Expr(x)
                    L = :([ (x=i*$dx;eval($expr)) for i=1:$m ])
                end
            elseif grade == 1
                # TODO: the discretization order should not be the same for
                #       first derivatives and second derivarives
                L = :($(UpwindDifference(grade,1,dx,m,1.)))
            else
                L = :($(CenteredDifference(grade,order,dx,m)))
            end
            return L
        elseif input.op isa Differential
            grade += 1
            return calc_coeff_mat(input.args[1],input.op.x,grade,order,dx,m,nonderiv_depvars)
        elseif input.op isa typeof(*)
            expr1 = calc_coeff_mat(input.args[1],iv,grade,order,dx,m,nonderiv_depvars)
            expr2 = calc_coeff_mat(input.args[2],iv,grade,order,dx,m,nonderiv_depvars)
            return Expr(:call,:*,expr1,expr2)
        elseif input.op isa typeof(-)
            if size(input.args,1) == 2
                expr1 = calc_coeff_mat(input.args[1],iv,grade,order,dx,m,nonderiv_depvars)
                expr2 = calc_coeff_mat(input.args[2],iv,grade,order,dx,m,nonderiv_depvars)
                return Expr(:call,:-,expr1,expr2)
            else #if size(input.args,1) == 1
                expr1 = calc_coeff_mat(input.args[1],iv,grade,order,dx,m,nonderiv_depvars)
                return Expr(:call,:*,:(-1),expr1)
            end
        elseif input.op isa typeof(+)
            if size(input.args,1) == 2
                expr1 = calc_coeff_mat(input.args[1],iv,grade,order,dx,m,nonderiv_depvars)
                expr2 = calc_coeff_mat(input.args[2],iv,grade,order,dx,m,nonderiv_depvars)
                return Expr(:call,:+,expr1,expr2)
            else #if size(input.args,1) == 1
                expr1 = calc_coeff_mat(input.args[1],iv,grade,order,dx,m,nonderiv_depvars)
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

    ### Calculate coefficient matrices #########################################
    # Each coeff. matrix (L_expr[x]) is an expression which is then evaluated
    # in the ODE function (f)

    # TODO: improve the code below using index arrays instead of Dicts?
    nonderiv_depvars = Dict()
    deriv_depvars = Dict()
    L_expr = Dict()
    # if there is only one equation
    if pdesys.eq isa Equation
        x = pdesys.eq.lhs.op
        L_expr[x] = calc_coeff_mat(pdesys.eq.rhs,0,0,order,dx[1],xx[1],Dict())
    # if there are many equations (pdesys.eq isa Array)
    else
        # Store 'non-derived' dependent variables (e.g. v(t,x)=t*x)
        # and 'derived' dependent variables (e.g. Dxx(u(t,x)))
        n_eqs = size(pdesys.eq,1)
        for i = 1:n_eqs
            x = pdesys.eq[i].lhs.op
            if x isa Variable
                nonderiv_depvars[x] = pdesys.eq[i].rhs
            else #x isa Differential
                deriv_depvars[x] = pdesys.eq[i].rhs
            end
        end

        # Calc. coeff. matrix for each differential equation
        for (x,rhs) in deriv_depvars
            L_expr[x] = calc_coeff_mat(rhs,0,0,order,dx[1],xx[1],nonderiv_depvars)
        end
    end

    ### Get boundary conditions ################################################
    # TODO: generalize to N equations
    (u_t0,u_x0,u_x1) = get_bcs(pdesys.bcs,tdomain,domain[1])
    # TODO: is there a better way to use eval here?
    t = 0.0
    g = eval(:((x,t) -> $u_t0))
    interior = domain[1].lower+dx[1]:dx[1]:domain[1].upper-dx[1]

    u0 = @eval $g.($interior,$t)
    Q = DirichletBC(u_x0,u_x1)

    # TODO: see TODO below related to "mul!(du,L,Q*u)"
    Id = [ i==j ? 1 : 0 for i = 1:xx[1]+2, j = 1:xx[1]+2 ]

    ### Define the discretized PDE as an ODE function ##########################
    function f(du,u,p,t)
        for L in values(L_expr)
            # TODO: is there a better way to use eval here?
            g = eval(:((t) -> $L))
            L = @eval $g.($t)

            # TODO: mul!(du,L,Q*u) should work when L is a DiffEqOperatorCombination
            #       L*Id should not be needed
            mul!(du,L*Id,Q*u)
        end
    end

    # Return problem ###########################################################
    return PDEProblem(ODEProblem(f,u0,(tdomain.lower,tdomain.upper),nothing),Q,X)
end
