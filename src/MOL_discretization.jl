# Method of lines discretization scheme
struct MOLFiniteDifference{T} <: DiffEqBase.AbstractDiscretization
  dxs::T
  order::Int
end
MOLFiniteDifference(args...;order=2) = MOLFiniteDifference(args,order)

# Evaluate expression
function eval_expr(expr,x,y)
   f = eval(:((x,y) -> $expr))
   return @eval $f.($x,$y)
end

# Obtain boundary conditions from an array
function extract_bc(bcs,tdomain,domain)
    u_t0 = 0.0
    u_x0 = 0.0
    u_x1 = 0.0
    n = size(bcs)[1]
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
function calc_coeff_mat(input,iv,grade,order,dx,m)
    if isa(input,ModelingToolkit.Constant)
            return input.value
    elseif isa(input,Operation)
        if isa(input.op,Variable)
            if grade == 1
                L = UpwindDifference(grade,order,dx,m,-1)
            else
                L = CenteredDifference(grade,order,dx,m)
            end
            return L
        elseif isa(input.op,Differential)
            grade += 1
            calc_coeff_mat(input.args[1],input.op.x,grade,order,dx,m)
        elseif isa(input.op,typeof(*))
            n = size(input.args)[1]
            output = calc_coeff_mat(input.args[1],iv,grade,order,dx,m) 
            for i = 2:n
                output *= calc_coeff_mat(input.args[i],iv,grade,order,dx,m) 
            end
            return output
        end
    end
end

# Convert a PDE problem into an ODE problem
function DiffEqBase.discretize(pdesys::PDESystem,discretization::MOLFiniteDifference)
    tdomain = pdesys.domain[1].domain
    domain = pdesys.domain[2].domain
    @assert domain isa IntervalDomain
    len = domain.upper-domain.lower
    dx = discretization.dxs[1]
    interior = domain.lower+dx:dx:domain.upper-dx
    X = domain.lower:dx:domain.upper
    m = size(X)[1]-2
    L = calc_coeff_mat(pdesys.eq.rhs,pdesys.indvars[2],0,discretization.order,dx,m)
    (u_t0,u_x0,u_x1) = extract_bc(pdesys.bcs,tdomain,domain)
    Q = DirichletBC(u_x0,u_x1)
    function f(du,u,p,t)
        mul!(du,L,Q*u)
    end
    t = 0.0
    u0 = eval_expr(u_t0,interior,t)
    PDEProblem(ODEProblem(f,u0,(tdomain.lower,tdomain.upper),nothing),Q,X)
end
