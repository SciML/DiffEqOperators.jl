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
    nottime = filter(x->~isequal(x.val, t.val),pdesys.indvars)
    nspace = length(nottime)
    depvars = pdesys.depvars

    # Discretize space
    space = map(nottime) do x
        xdomain = pdesys.domain[findfirst(d->isequal(x.val, d.variables),pdesys.domain)]
        @assert xdomain.domain isa IntervalDomain
        dx = discretization.dxs[findfirst(dxs->isequal(x.val, dxs[1].val),discretization.dxs)][2]
        dx isa Number ? (xdomain.domain.lower:dx:xdomain.domain.upper) : dx
    end
    dxs = map(nottime) do x        
        dx = discretization.dxs[findfirst(dxs->isequal(x.val, dxs[1].val),discretization.dxs)][2]
    end
    # Get tspan
    tdomain = pdesys.domain[findfirst(d->isequal(t.val, d.variables),pdesys.domain)]
    @assert tdomain.domain isa IntervalDomain
    tspan = (tdomain.domain.lower,tdomain.domain.upper)

    # Build symbolic variables
    indices = CartesianIndices(((axes(s)[1] for s in space)...,))
    depvarsdisc = map(depvars) do u
        [Num(Variable{Symbolics.FnType{Tuple{Any}, Real}}(Base.nameof(ModelingToolkit.operation(u.val)),II.I...))(t) for II in indices]
    end
    spacevals = map(y->[Pair(nottime[i],space[i][y.I[i]]) for i in 1:nspace],indices)


    ### INITIAL AND BOUNDARY CONDITIONS ###
    # Build symbolic maps for boundaries
    edges = reduce(vcat,[[vcat([Colon() for j in 1:i-1],1,[Colon() for j in i+1:nspace]),
      vcat([Colon() for j in 1:i-1],size(depvarsdisc[1],i),[Colon() for j in i+1:nspace])] for i in 1:nspace])

    #edgeindices = [indices[e...] for e in edges]
    edgevals = reduce(vcat,[[nottime[i]=>first(space[i]),nottime[i]=>last(space[i])] for i in 1:length(space)])
    edgevars = [[d[e...] for e in edges] for d in depvarsdisc]
    
    bclocs = map(e->substitute.(pdesys.indvars,e),edgevals) # location of the boundary conditions e.g. (t,0.0,y)
    edgemaps = Dict(bclocs .=> [spacevals[e...] for e in edges])
    initmaps = substitute.(depvars,[t=>tspan[1]])

    # Generate map from variable (e.g. u(t,0)) to discretized variable (e.g. u₁(t))
    subvar(depvar) = substitute.((depvar,),edgevals)
    # depvarbcmaps will dictate what to replace the variable terms with in the bcs
    depvarbcmaps = reduce(vcat,[subvar(depvar) .=> edgevar for (depvar, edgevar) in zip(depvars, edgevars)])
    # depvarderivbcmaps will dictate what to replace the Differential terms with in the bcs
    if nspace == 1
        # 1D system
        left_weights(j) = DiffEqOperators.calculate_weights(discretization.upwind_order, space[j][1], space[j][1:2])
        right_weights(j) = DiffEqOperators.calculate_weights(discretization.upwind_order, space[j][end], space[j][end-1:end])
        central_neighbor_idxs(i,j) = [i+CartesianIndex([ifelse(l==j,-1,0) for l in 1:nspace]...),i,i+CartesianIndex([ifelse(l==j,1,0) for l in 1:nspace]...)]
        left_idxs = central_neighbor_idxs(CartesianIndex(2),1)[1:2]
        right_idxs(j) = central_neighbor_idxs(CartesianIndex(length(space[j])-1),1)[end-1:end]
        # Constructs symbolic spatially discretized terms of the form e.g. au₂ - bu₁
        derivars = [[dot(left_weights(1),depvar[left_idxs]), dot(right_weights(1),depvar[right_idxs(1)])]
                    for depvar in depvarsdisc]
        # Create list of all the symbolic Differential terms evaluated at boundary e.g. Differential(x)(u(t,0))
        subderivar(depvar,s) = substitute.((Differential(s)(depvar),),edgevals)
        # Create map of symbolic Differential terms with symbolic spatially discretized terms
        depvarderivbcmaps = reduce(vcat,[subderivar(depvar, s) .=> derivars[i]
                                       for (i, depvar) in enumerate(depvars) for s in nottime])
   else
        # Higher dimension
        # TODO: Fix Neumann and Robin on higher dimension
        depvarderivbcmaps = []
   end

    # Generate initial conditions and bc equations
    u0 = []
    bceqs = []
    for bc in pdesys.bcs
        if ModelingToolkit.operation(bc.lhs) isa Sym && ~any(x -> isequal(x, t.val), ModelingToolkit.arguments(bc.lhs))
            # initial condition
            # Assume in the form `u(...) ~ ...` for now
            push!(u0,vec(depvarsdisc[findfirst(isequal(bc.lhs),initmaps)] .=> substitute.((bc.rhs,),spacevals)))
        else
            # Algebraic equations for BCs
            i = findfirst(x->occursin(x.val,bc.lhs),first.(depvarbcmaps))
            bcargs = first(depvarbcmaps[i]).val.arguments
            # Replace Differential terms in the bc lhs with the symbolic spatially discretized terms
            # TODO: Fix Neumann and Robin on higher dimension
            lhs = nspace == 1 ? substitute(bc.lhs,depvarderivbcmaps[i]) : bc.lhs

            # Replace symbol in the bc lhs with the spatial discretized term
            lhs = substitute(lhs,depvarbcmaps[i])
            rhs = substitute.((bc.rhs,),edgemaps[bcargs])
            lhs = lhs isa Vector ? lhs : [lhs] # handle 1D
            push!(bceqs,lhs .~ rhs)
        end
    end

    u0 = reduce(vcat,u0)
    bceqs = reduce(vcat,bceqs)

    ### PDE EQUATIONS ###
    interior = indices[[2:length(s)-1 for s in space]...]
    eqs = vec(map(Base.product(interior,pdeeqs)) do p
        II,eq = p
    
        # Create a stencil in the required dimension centered around 0
        # e.g. (-1,0,1) for 2nd order, (-2,-1,0,1,2) for 4th order, etc
        order = discretization.centered_order
        stencil(j) = CartesianIndices(Tuple(map(x -> -x:x, (1:nspace.==j) * (order÷2))))
    
        # TODO: Generalize central difference handling to allow higher even order derivatives
        # The central neighbour indices should add the stencil to II, unless II is too close
        # to an edge in which case we need to shift away from the edge

        # Calculate buffers
        # TODO: check central_neighbor_idxs. It does not work like the old implementation.
        I1 = oneunit(first(indices))
        Imin = first(indices) + I1 * (order÷2)
        Imax = last(indices) - I1 * (order÷2)
        # Use max and min to apply buffers
        central_neighbor_idxs(II,j) = stencil(j) .+ max(Imin,min(II,Imax))  
        central_neighbor_space(II,j) = vec(space[j][map(i->i[j],central_neighbor_idxs(II,j))])
        central_weights(II,j) = DiffEqOperators.calculate_weights(2, space[j][II[j]], central_neighbor_space(II,j))
        central_deriv(II,j,k) = dot(central_weights(II,j),depvarsdisc[k][central_neighbor_idxs(II,j)])

        central_deriv_rules = [(Differential(s)^2)(u) => central_deriv(II,j,k) for (j,s) in enumerate(nottime), (k,u) in enumerate(depvars)]
        valrules = vcat([depvars[k] => depvarsdisc[k][II] for k in 1:length(depvars)],
                        [nottime[j] => space[j][II[j]] for j in 1:nspace])
    
        # TODO: upwind rules needs interpolation into `@rule`
        forward_weights(i,j) = DiffEqOperators.calculate_weights(discretization.upwind_order, 0.0, [space[j][i[j]],space[j][i[j]+1]])
        reverse_weights(i,j) = DiffEqOperators.calculate_weights(discretization.upwind_order, 0.0, [space[j][i[j]-1],space[j][i[j]]])
        # upwinding_rules = [@rule(*(~~a,$(Differential(nottime[j]))(u),~~b) => IfElse.ifelse(*(~~a..., ~~b...,)>0,
        #                         *(~~a..., ~~b..., dot(reverse_weights(i,j),depvars[k][central_neighbor_idxs(i,j)[1:2]])),
        #                         *(~~a..., ~~b..., dot(forward_weights(i,j),depvars[k][central_neighbor_idxs(i,j)[2:3]]))))
        #                         for j in 1:nspace, k in 1:length(pdesys.depvars)]

        ## Discretization of non-linear laplacian. 
        # d/dx( a du/dx ) ~ (a(x+1/2) * (u[i+1] - u[i]) - a(x-1/2) * (u[i] - u[i-1]) / dx^2
        b1(i, j, k) = dot(reverse_weights(i, j), depvarsdisc[k][central_neighbor_idxs(i, j)[1:2]]) / dxs[j]
        b2(i, j, k) = dot(forward_weights(i, j), depvarsdisc[k][central_neighbor_idxs(i, j)[2:3]]) / dxs[j]
        # TODO: improve interpolation of g(x) = u(x) for calculating u(x+-dx/2)
        g(i, j, k, l) = sum([depvarsdisc[k][central_neighbor_idxs(i, j)][s] for s in (l == 1 ? [2,3] : [1,2])]) / 2.
        # iv_mid returns middle space values. E.g. x(i-1/2) or y(i+1/2).
        iv_mid(i, j, l) = (space[j][i[j]] + space[j][i[j]+l]) / 2.0 
        # Dependent variable rules
        r_mid_dep(i, j, k, l) = [depvars[k] => g(i, j, k, l) for k in 1:length(depvars)]
        # Independent variable rules
        r_mid_indep(i, j, l) = [nottime[j] => iv_mid(i, j, l) for j in 1:length(nottime)]
        # Replacement rules: new approach
        rules = [@rule ($(Differential(iv))(*(~~a, $(Differential(iv))(dv), ~~b))) =>
                 dot([Num(substitute(substitute(*(~~a..., ~~b...), r_mid_dep(II, j, k, -1)), r_mid_indep(II, j, -1))),
                      Num(substitute(substitute(*(~~a..., ~~b...), r_mid_dep(II, j, k, 1)), r_mid_indep(II, j, 1)))],
                     [-b1(II, j, k), b2(II, j, k)])
                 for (j, iv) in enumerate(nottime) for (k, dv) in enumerate(depvars)]
        rhs_arg = (SymbolicUtils.operation(eq.rhs) == +) ? SymbolicUtils.arguments(eq.rhs) : [eq.rhs]
        lhs_arg = (SymbolicUtils.operation(eq.lhs) == +) ? SymbolicUtils.arguments(eq.lhs) : [eq.lhs]
        nonlinlap_rules = []
        for t in vcat(lhs_arg,rhs_arg)
            for r in rules
                if r(t) != nothing
                    push!(nonlinlap_rules, t => r(t))
                end
            end
        end

        rules = vcat(vec(nonlinlap_rules),
                     vec(central_deriv_rules),valrules)

        substitute(eq.lhs,rules) ~ substitute(eq.rhs,rules)

    end)

    # Finalize
    defaults = pdesys.ps === nothing || pdesys.ps === SciMLBase.NullParameters() ? u0 : vcat(u0,pdesys.ps)
    ps = pdesys.ps === nothing || pdesys.ps === SciMLBase.NullParameters() ? Num[] : first.(pdesys.ps)
    # Combine PDE equations and BC equations
    sys = ODESystem(vcat(eqs,unique(bceqs)),t,vec(reduce(vcat,vec(depvarsdisc))),ps,defaults=Dict(defaults))
    sys, tspan
end

function SciMLBase.discretize(pdesys::ModelingToolkit.PDESystem,discretization::DiffEqOperators.MOLFiniteDifference)
    sys, tspan = SciMLBase.symbolic_discretize(pdesys,discretization)
    simpsys = structural_simplify(sys)
    prob = ODEProblem(simpsys,Pair[],tspan)
end
