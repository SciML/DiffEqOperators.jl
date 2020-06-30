#using Reduce

# Method of lines discretization scheme
struct MOLFiniteDifference{T} <: DiffEqBase.AbstractDiscretization
  dxs::T
  order::Int
end
MOLFiniteDifference(args...;order=2) = MOLFiniteDifference(args,order)

# Evaluate expression
# TODO: replace this eval with proper substitution
function eval_expr(expr,x,y)
   f = eval(:((x,y) -> $expr))
   return @eval $f.($x,$y)
end

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
                # TODO: Here, evaluate expression w.r.t space (x,y,z).
                #       Then, in the "f" ODE function (in DiffEqBase.discretize),
                #       evaluate expression w.r.t. time (t).
                expr = Expr(nonderiv_depvars[input.op])
                L = :(Diagonal([ (x=i*$dx;eval($expr)) for i=1:$m ]))
            elseif grade == 1
                L = :($(UpwindDifference(grade,order,dx,m,1)))
            else
                L = :($(CenteredDifference(grade,order,dx,m)))
            end
            return L
        elseif input.op isa Differential
            grade += 1
            return calc_coeff_mat(input.args[1],input.op.x,grade,order,dx,m,nonderiv_depvars)
        elseif input.op isa typeof(-)
            L = calc_coeff_mat(input.args[1],iv,grade,order,dx,m,nonderiv_depvars)
            return Expr(:call, :*, :(-1), L)
        elseif input.op isa typeof(*)
            # TODO: probably the code below can be shortened
            expr1 = calc_coeff_mat(input.args[1],iv,grade,order,dx,m,nonderiv_depvars)
            expr2 = calc_coeff_mat(input.args[2],iv,grade,order,dx,m,nonderiv_depvars)
            expr1 = Expr(:call, :*, expr1, expr2)
            for i = 3:size(input.args,1)
                expr2 = calc_coeff_mat(input.args[i],iv,grade,order,dx,m,nonderiv_depvars)
                expr1 = Expr(:call, :*, expr1, expr2)
            end
            return expr1
        end
    end
end

# Convert a PDE problem into an ODE problem
function DiffEqBase.discretize(pdesys::PDESystem,discretization::MOLFiniteDifference)

    ### Get spatial and temporal domains #######################################
    tdomain = pdesys.domain[1].domain
    domain = pdesys.domain[2].domain
    @assert tdomain isa IntervalDomain
    @assert domain isa IntervalDomain
    dx = discretization.dxs
    interior = domain.lower+dx:dx:domain.upper-dx
    X = domain.lower:dx:domain.upper
    order = discretization.order
    m = size(X,1)-2

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
        L_expr[x] = calc_coeff_mat(pdesys.eq.rhs,pdesys.indvars[2],0,
                                   order,dx,m,Dict())
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
            L_expr[x] = calc_coeff_mat(rhs,pdesys.indvars[2],0,order,
                                       dx,m,nonderiv_depvars)
        end
    end

    ### Get boundary conditions ################################################
    # TODO: generalize to N equations
    # TODO: eval issue
    (u_t0,u_x0,u_x1) = get_bcs(pdesys.bcs,tdomain,domain)
    t = 0.0
    u0 = eval_expr(u_t0,interior,t)
    Q = DirichletBC(u_x0,u_x1)

    ### Define the discretized PDE as an ODE function ##########################
    function f(du,u,p,t)
        for L in values(L_expr)
            # TODO: there is probably a fancier way to eval time
            g = eval(:((t) -> $L))
            L = @eval $g.($t)
            mul!(du,L,Q*u)
        end
    end

    # Return problem ###########################################################
    return PDEProblem(ODEProblem(f,u0,(tdomain.lower,tdomain.upper),nothing),Q,X)
end


