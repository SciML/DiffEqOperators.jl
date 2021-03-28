using ModelingToolkit: operation, istree, arguments
# Method of lines discretization scheme
struct MOLFiniteDifference{T,T2} <: DiffEqBase.AbstractDiscretization
    dxs::T
    time::T2
    upwind_order::Int
    centered_order::Int
end

# Constructors. If no order is specified, both upwind and centered differences will be 2nd order
MOLFiniteDifference(dxs, time; upwind_order = 1, centered_order = 2) =
    MOLFiniteDifference(dxs, time, upwind_order, centered_order)

function SciMLBase.symbolic_discretize(pdesys::ModelingToolkit.PDESystem,discretization::DiffEqOperators.MOLFiniteDifference)
    pdeeqs = pdesys.eqs isa Vector ? pdesys.eqs : [pdesys.eqs]
    t = discretization.time
    nottime = filter(x->x.val != t.val,pdesys.indvars)

    # Discretize space

    space = map(nottime) do x
        xdomain = pdesys.domain[findfirst(d->x.val == d.variables,pdesys.domain)]
        @assert xdomain.domain isa IntervalDomain
        dx = discretization.dxs[findfirst(dxs->x.val == dxs[1].val,discretization.dxs)][2]
        dx isa Number ? (xdomain.domain.lower:dx:xdomain.domain.upper) : dx
    end
    tdomain = pdesys.domain[findfirst(d->t.val == d.variables,pdesys.domain)]
    @assert tdomain.domain isa IntervalDomain
    tspan = (tdomain.domain.lower,tdomain.domain.upper)

    # Build symbolic variables
    indices = CartesianIndices(((axes(s)[1] for s in space)...,))
    depvars = map(pdesys.depvars) do u
        [Num(Variable{Symbolics.FnType{Tuple{Any}, Real}}(Base.nameof(ModelingToolkit.operation(u.val)),II.I...))(t) for II in indices]
    end
    spacevals = map(y->[Pair(nottime[i],space[i][y.I[i]]) for i in 1:length(nottime)],indices)

    # Build symbolic maps
    edges = reduce(vcat,[[vcat([Colon() for j in 1:i-1],1,[Colon() for j in i+1:length(nottime)]),
      vcat([Colon() for j in 1:i-1],size(depvars[1],i),[Colon() for j in i+1:length(nottime)])] for i in 1:length(nottime)])

    edgevals = reduce(vcat,[[nottime[i]=>first(space[i]),nottime[i]=>last(space[i])] for i in 1:length(space)])
    edgevars = [[d[e...] for e in edges] for d in depvars]
    depvarmaps = reduce(vcat,[substitute.((pdesys.depvars[i],),edgevals) .=> edgevars[i] for i in 1:length(pdesys.depvars)])
    edgemaps = [spacevals[e...] for e in edges]
    initmaps = substitute.(pdesys.depvars,[t=>tspan[1]])

    # Generate initial conditions and bc equations
    # Assume in the form `u(...) ~ ...` for now
    u0 = []
    bceqs = []
    for bc in pdesys.bcs
        if t.val ∉ ModelingToolkit.arguments(bc.lhs)
            # initial condition
            push!(u0,vec(depvars[findfirst(isequal(bc.lhs),initmaps)] .=> substitute.((bc.rhs,),spacevals)))
        else
            # Algebraic equations for BCs
            i = findfirst(x->isequal(x,bc.lhs),first.(depvarmaps))
            lhs = substitute(bc.lhs,depvarmaps[i])
            rhs = substitute.((bc.rhs,),edgemaps[i])
            lhs = lhs isa Vector ? lhs : [lhs] # handle 1D
            push!(bceqs,lhs .~ rhs)
        end
    end
    u0 = reduce(vcat,u0)
    bceqs = reduce(vcat,bceqs)

    # Generate PDE Equations
    interior = indices[[2:length(s)-1 for s in space]...]
    eqs = vec(map(Base.product(interior,pdeeqs)) do p
        i,eq = p

        # TODO: Number of points in the central_neighbor_idxs should be dependent
        # on discretization.centered_order
        # TODO: Generalize central difference handling to allow higher even order derivatives
        central_neighbor_idxs(i,j) = [i+CartesianIndex([ifelse(l==j,-1,0) for l in 1:length(nottime)]...),i,i+CartesianIndex([ifelse(l==j,1,0) for l in 1:length(nottime)]...)]
        central_weights(i,j) = DiffEqOperators.calculate_weights(2, 0.0, [space[j][i[j]-1],space[j][i[j]],space[j][i[j]+1]])
        central_deriv_rules = [(Differential(nottime[j])^2)(pdesys.depvars[k]) => dot(central_weights(i,j),depvars[k][central_neighbor_idxs(i,j)]) for j in 1:length(nottime), k in 1:length(pdesys.depvars)]
        valrules = [pdesys.depvars[k] => depvars[k][i] for k in 1:length(pdesys.depvars)]

        # TODO: Use rule matching for nonlinear Laplacian

        # TODO: upwind rules needs interpolation into `@rule`
        #forward_weights(i,j) = DiffEqOperators.calculate_weights(discretization.upwind_order, 0.0, [space[j][i[j]],space[j][i[j]+1]])
        #reverse_weights(i,j) = DiffEqOperators.calculate_weights(discretization.upwind_order, 0.0, [space[j][i[j]-1],space[j][i[j]]])
        #upwinding_rules = [@rule(*(~~a,(Differential(nottime[j]))(u),~~b) => IfElse.ifelse(*(~~a..., ~~b...,)>0,
        #                        *(~~a..., ~~b..., dot(reverse_weights(i,j),depvars[k][central_neighbor_idxs(i,j)[1:2]])),
        #                        *(~~a..., ~~b..., dot(forward_weights(i,j),depvars[k][central_neighbor_idxs(i,j)[2:3]]))))
        #                        for j in 1:length(nottime), k in 1:length(pdesys.depvars)]

        substitute(eq.lhs,vcat(vec(central_deriv_rules),valrules)) ~ substitute(eq.rhs,vcat(vec(central_deriv_rules),valrules))
    end)

    # Finalize
    defaults = pdesys.ps === nothing ? u0 : vcat(u0,pdesys.ps)
    ps = pdesys.ps === nothing ? Num[] : first.(pdesys.ps)
    sys = ODESystem(vcat(eqs,unique(bceqs)),t,vec(reduce(vcat,vec(depvars))),ps,defaults=defaults)
    sys, tspan
end

function SciMLBase.discretize(pdesys::ModelingToolkit.PDESystem,discretization::DiffEqOperators.MOLFiniteDifference)
    sys, tspan = SciMLBase.symbolic_discretize(pdesys,discretization)
    simpsys = structural_simplify(sys)
    prob = ODEProblem(simpsys,Pair[],tspan)
end
