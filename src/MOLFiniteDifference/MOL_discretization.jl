using ModelingToolkit: operation, istree, arguments
# Method of lines discretization scheme
struct MOLFiniteDifference{T,T2} <: DiffEqBase.AbstractDiscretization
    dxs::T
    time::T2
    upwind_order::Int
    centered_order::Int
end

# for terms that of the form:
# u(x,t) or Differential(x)(u(x,t)), returns, u(x,t) in both cases.
function get_sym(term)
    if !Symbolics.istree(term)
        return nothing
    end
    if SymbolicUtils.operation(term) isa Sym
        return term
    else
        # FIXME: for multiple terms, simply return the first...possible bug here
        syms = filter(!isnothing, [get_sym(t) for t in SymbolicUtils.arguments(term)])
        return isempty(syms) ? nothing : first(syms)
    end
end

# get all terms on the lhs of given equatios that involve the given dependent
# variable
function get_depvar_terms(eqs, depvar)
    syms = filter(!isnothing, [get_sym(eq.lhs) for eq in eqs])
    return filter(s -> depvar.val.f === s.f, syms)
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

    #---- Count Boundary Equations --------------------
    # Count the number of boundary equations that lie at the spatial boundary on
    # both the left and right side. This will be used to determine number of
    # interior equations s.t. we have a balanced system of equations.
    # TODO: Check/Generalize to work with multi-dimensional equations
    # TODO: Check Generalization to higher order boundary conditions (e.g., beam equation with 2nd order/3rd order boundary)
    # TODO: Check against equations with multiple dependent vars

    # bc_values will contain the pair mappings from vars to their boundary location.
    # FIXME: Currently only considering the first depvar, we need to consider all available depvars in the system.
    bc_values = reduce(vcat, [pdesys.indvars .=> t.arguments for t in get_depvar_terms(pdesys.bcs, depvars[1])])
    # remove non-value terms and time, we are only concerned with boudary edge values.
    bc_values = filter(v -> !(last(v) isa Sym) && ~isequal(first(v), t.val), bc_values)
    bc_count(indvar,edgefunc) = count(v -> last(v) == edgefunc(last.(edgevals)), filter(kv->first(kv) === indvar, bc_values))
    # FIXME: Assuming that the bc count for all indendent vars is the same, so we just take the first one here.
    left_bc_count = first([bc_count(indvar, minimum) for indvar in nottime])
    right_bc_count = first([bc_count(indvar, maximum) for indvar in nottime])
    #--------------------------------------------------

    ### PDE EQUATIONS ###
    interior = indices[[(1 + left_bc_count):length(s)-(right_bc_count) for s in space]...]
    eqs = vec(map(Base.product(interior,pdeeqs)) do p
        II,eq = p
    
        # Create a stencil in the required dimension centered around 0
        # e.g. (-1,0,1) for 2nd order, (-2,-1,0,1,2) for 4th order, etc
        if discretization.centered_order % 2 != 0
            throw(ArgumentError("Discretization centered_order must be even, given $(discretization.centered_order)"))
        end
        approx_order = discretization.centered_order
        stencil(j) = CartesianIndices(Tuple(map(x -> -x:x, (1:nspace.==j) * (approx_order÷2))))
    
        # TODO: Generalize central difference handling to allow higher even order derivatives
        # The central neighbour indices should add the stencil to II, unless II is too close
        # to an edge in which case we need to shift away from the edge
        # Calculate buffers
        I1 = oneunit(first(indices))
        Imin = first(indices) + I1 * (approx_order÷2)
        Imax = last(indices) - I1 * (approx_order÷2)
        # Use max and min to apply buffers
        central_neighbor_idxs(II,j) = stencil(j) .+ max(Imin,min(II,Imax))
        central_neighbor_space(II,j) = vec(space[j][map(i->i[j],central_neighbor_idxs(II,j))])
        central_weights(d_order,II,j) = DiffEqOperators.calculate_weights(d_order, space[j][II[j]], central_neighbor_space(II,j))
        central_deriv(d_order,II,j,k) = dot(central_weights(d_order,II,j),depvarsdisc[k][central_neighbor_idxs(II,j)])

        # get a sorted list derivative order such that highest order is first. This is useful when substituting rules
        # starting from highest to lowest order.
        d_orders(s) = reverse(sort(collect(union(differential_order(eq.rhs, s.val), differential_order(eq.lhs, s.val)))))

        # central_deriv_rules = [(Differential(s)^2)(u) => central_deriv(2,II,j,k) for (j,s) in enumerate(nottime), (k,u) in enumerate(depvars)]
        central_deriv_rules = Array{Pair{Num,Num},1}()
        for (j,s) in enumerate(nottime)
            rs = [(Differential(s)^d)(u) => central_deriv(d,II,j,k) for d in d_orders(s), (k,u) in enumerate(depvars)]
            for r in rs
              push!(central_deriv_rules, r)
            end
        end
        valrules = vcat([depvars[k] => depvarsdisc[k][II] for k in 1:length(depvars)],
                        [nottime[j] => space[j][II[j]] for j in 1:nspace])
    
        # TODO: Use rule matching for nonlinear Laplacian
    
        # TODO: upwind rules needs interpolation into `@rule`
        #forward_weights(i,j) = DiffEqOperators.calculate_weights(discretization.upwind_order, 0.0, [space[j][i[j]],space[j][i[j]+1]])
        #reverse_weights(i,j) = DiffEqOperators.calculate_weights(discretization.upwind_order, 0.0, [space[j][i[j]-1],space[j][i[j]]])
        #upwinding_rules = [@rule(*(~~a,(Differential(nottime[j]))(u),~~b) => IfElse.ifelse(*(~~a..., ~~b...,)>0,
        #                        *(~~a..., ~~b..., dot(reverse_weights(i,j),depvarsdisc[k][central_neighbor_idxs(i,j)[1:2]])),
        #                        *(~~a..., ~~b..., dot(forward_weights(i,j),depvarsdisc[k][central_neighbor_idxs(i,j)[2:3]]))))
        #                        for j in 1:nspace, k in 1:length(depvars)]
    
        rules = vcat(vec(central_deriv_rules),valrules)
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
