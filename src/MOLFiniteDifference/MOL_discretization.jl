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

function calculate_weights_general(order::Int, x0::T, xs::AbstractVector, idxs::AbstractVector, domain::IntervalDomain) where T<:Real
        # Cartesian domain: use Fornberg
        DiffEqOperators.calculate_weights(order, x0, vec(xs[idxs]))
 end
 function calculate_weights_general(order::Int, x0::T, x::AbstractVector, idxs::AbstractVector, domain::AxisymmetricSphereDomain) where T<:Real
        # Spherical domain: see #367
        # https://web.mit.edu/braatzgroup/analysis_of_finite_difference_discretization_schemes_for_diffusion_in_spheres_with_variable_diffusivity.pdf
        # Only order 2 is implemented
        @assert order == 2
        # TODO: nonlinear diffusion in a spherical domain
        i = idxs[2] 
        dx1 = x[i] - x[i-1]
        dx2 = x[i+1] - x[i]
        i0 = i - 1 # indexing starts at 0 in the paper and starts at 1 in julia
        1 / (i0 * dx1 * dx2) * [i0-1, -2i0, i0+1]
 end
 

function SciMLBase.symbolic_discretize(pdesys::ModelingToolkit.PDESystem,discretization::DiffEqOperators.MOLFiniteDifference)
    pdeeqs = pdesys.eqs isa Vector ? pdesys.eqs : [pdesys.eqs]
    t = discretization.time
    nottime = filter(x->~isequal(x.val, t.val),pdesys.indvars)
    nspace = length(nottime)

    # Discretize space
    spacedomains = map(nottime) do x
        xdomain = pdesys.domain[findfirst(d->isequal(x.val, d.variables),pdesys.domain)]
        @assert xdomain.domain isa AbstractDomain
        xdomain.domain
    end
    space = map(nottime) do x
        xdomain = pdesys.domain[findfirst(d->isequal(x.val, d.variables),pdesys.domain)]
        dx = discretization.dxs[findfirst(dxs->isequal(x.val, dxs[1].val),discretization.dxs)][2]
        dx isa Number ? (xdomain.domain.lower:dx:xdomain.domain.upper) : dx
    end
    # Get tspan
    tdomain = pdesys.domain[findfirst(d->isequal(t.val, d.variables),pdesys.domain)]
    @assert tdomain.domain isa IntervalDomain
    tspan = (tdomain.domain.lower,tdomain.domain.upper)

    # Build symbolic variables
    indices = CartesianIndices(((axes(s)[1] for s in space)...,))
    depvars = map(pdesys.depvars) do u
        [Num(Variable{Symbolics.FnType{Tuple{Any}, Real}}(Base.nameof(ModelingToolkit.operation(u.val)),II.I...))(t) for II in indices]
    end
    spacevals = map(y->[Pair(nottime[i],space[i][y.I[i]]) for i in 1:nspace],indices)


    ### INITIAL AND BOUNDARY CONDITIONS ###
    # Build symbolic maps for boundaries
    edges = reduce(vcat,[[vcat([Colon() for j in 1:i-1],1,[Colon() for j in i+1:nspace]),
    vcat([Colon() for j in 1:i-1],size(depvars[1],i),[Colon() for j in i+1:nspace])] for i in 1:nspace])

    #edgeindices = [indices[e...] for e in edges]
    edgevals = reduce(vcat,[[nottime[i]=>first(space[i]),nottime[i]=>last(space[i])] for i in 1:length(space)])
    edgevars = [[d[e...] for e in edges] for d in depvars]
    edgemaps = [spacevals[e...] for e in edges]
    initmaps = substitute.(pdesys.depvars,[t=>tspan[1]])

    # Generate map from variable (e.g. u(t,0)) to discretized variable (e.g. u₁(t))
    subvar(depvar) = substitute.((depvar,),edgevals)
    depvarmaps = reduce(vcat,[subvar(depvar) .=> edgevars[i] for (i, depvar) in enumerate(pdesys.depvars)])
    # depvarderivmaps will dictate what to replace the Differential terms with
    # use Fornberg (DiffEqOperators.calculate_weights) for all geometries since gradient operator is the same
    if nspace == 1
        # 1D system
        left_weights(j) = DiffEqOperators.calculate_weights(discretization.upwind_order, space[j][1], space[j][1:2])
        right_weights(j) = DiffEqOperators.calculate_weights(discretization.upwind_order, space[j][end], space[j][end-1:end])
        central_neighbor_idxs(II,j) = [II-CartesianIndex((1:nspace.==j)...),II,II+CartesianIndex((1:nspace.==j)...)]
        left_idxs = central_neighbor_idxs(CartesianIndex(2),1)[1:2]
        right_idxs(j) = central_neighbor_idxs(CartesianIndex(length(space[j])-1),1)[end-1:end]
        # Constructs symbolic spatially discretized terms of the form e.g. au₂ - bu₁
        derivars = [[dot(left_weights(j),depvar[left_idxs]), dot(right_weights(j),depvar[right_idxs(j)])]
                    for (j, depvar) in enumerate(depvars)]
        # Create list of all the symbolic Differential terms evaluated at boundary e.g. Differential(x)(u(t,0))
        subderivar(depvar,s) = substitute.((Differential(s)(depvar),),edgevals)
        # Create map of symbolic Differential terms with symbolic spatially discretized terms
        depvarderivmaps = reduce(vcat,[subderivar(depvar, s) .=> derivars[i]
                                    for (i, depvar) in enumerate(pdesys.depvars) for s in nottime])
    else
        # Higher dimension
        # TODO: Fix Neumann and Robin on higher dimension
        depvarderivmaps = []
    end

    # Generate initial conditions and bc equations
    u0 = []
    bceqs = []
    for bc in pdesys.bcs
        if ModelingToolkit.operation(bc.lhs) isa Sym && ~any(x -> isequal(x, t.val), ModelingToolkit.arguments(bc.lhs))
            # initial condition
            # Assume in the form `u(...) ~ ...` for now
            push!(u0,vec(depvars[findfirst(isequal(bc.lhs),initmaps)] .=> substitute.((bc.rhs,),spacevals)))
        else
            # Algebraic equations for BCs
            i = findfirst(x->occursin(x.val,bc.lhs),first.(depvarmaps))

            # Replace Differential terms in the bc lhs with the symbolic spatially discretized terms
            # TODO: Fix Neumann and Robin on higher dimension
            lhs = nspace == 1 ? substitute(bc.lhs,depvarderivmaps[i]) : bc.lhs

            # Replace symbol in the bc lhs with the spatial discretized term
            lhs = substitute(lhs,depvarmaps[i])
            rhs = substitute.((bc.rhs,),edgemaps[i])
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
        I1 = oneunit(first(indices))
        Imin = first(indices) + I1 * (order÷2)
        Imax = last(indices) - I1 * (order÷2)
        # Use max and min to apply buffers
        central_neighbor_idxs(II,j) = stencil(j) .+ max(Imin,min(II,Imax))
        central_weights(II,j) = calculate_weights_general(2, space[j][II[j]], space[j], vec(map(i->i[j], central_neighbor_idxs(II,j))), spacedomains[j])
        central_deriv(II,j,k) = dot(central_weights(II,j),depvars[k][central_neighbor_idxs(II,j)])

        # TODO: detect Laplacian in general form
        central_deriv_rules = [(Differential(s)^2)(u) => central_deriv(II,j,k) for (j,s) in enumerate(nottime), (k,u) in enumerate(pdesys.depvars)]
        central_deriv_rules_sphere = [(Differential(s))(s^2*Differential(s)(u))/s^2 => central_deriv(II,j,k) 
                                    for (j,s) in enumerate(nottime), (k,u) in enumerate(pdesys.depvars)]
        valrules = vcat([pdesys.depvars[k] => depvars[k][II] for k in 1:length(pdesys.depvars)],
                        [nottime[j] => space[j][II[j]] for j in 1:nspace])

        # TODO: Use rule matching for nonlinear Laplacian

        # TODO: upwind rules needs interpolation into `@rule`
        #forward_weights(i,j) = DiffEqOperators.calculate_weights(discretization.upwind_order, 0.0, [space[j][i[j]],space[j][i[j]+1]])
        #reverse_weights(i,j) = DiffEqOperators.calculate_weights(discretization.upwind_order, 0.0, [space[j][i[j]-1],space[j][i[j]]])
        #upwinding_rules = [@rule(*(~~a,(Differential(nottime[j]))(u),~~b) => IfElse.ifelse(*(~~a..., ~~b...,)>0,
        #                        *(~~a..., ~~b..., dot(reverse_weights(i,j),depvars[k][central_neighbor_idxs(i,j)[1:2]])),
        #                        *(~~a..., ~~b..., dot(forward_weights(i,j),depvars[k][central_neighbor_idxs(i,j)[2:3]]))))
        #                        for j in 1:nspace, k in 1:length(pdesys.depvars)]

        rules = vcat(vec(central_deriv_rules),vec(central_deriv_rules_sphere),valrules)
        substitute(eq.lhs,rules) ~ substitute(eq.rhs,rules)
    end)

    # Finalize
    defaults = pdesys.ps === nothing || pdesys.ps === SciMLBase.NullParameters() ? u0 : vcat(u0,pdesys.ps)
    ps = pdesys.ps === nothing || pdesys.ps === SciMLBase.NullParameters() ? Num[] : first.(pdesys.ps)
    # Combine PDE equations and BC equations
    sys = ODESystem(vcat(eqs,unique(bceqs)),t,vec(reduce(vcat,vec(depvars))),ps,defaults=Dict(defaults))
    sys, tspan
end

function SciMLBase.discretize(pdesys::ModelingToolkit.PDESystem,discretization::DiffEqOperators.MOLFiniteDifference)
    sys, tspan = SciMLBase.symbolic_discretize(pdesys,discretization)
    simpsys = structural_simplify(sys)
    prob = ODEProblem(simpsys,Pair[],tspan)
end
