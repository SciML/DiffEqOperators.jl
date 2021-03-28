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

    #edgeindices = [indices[e...] for e in edges]
    edgevals = reduce(vcat,[[nottime[i]=>first(space[i]),nottime[i]=>last(space[i])] for i in 1:length(space)])
    edgevars = [[d[e...] for e in edges] for d in depvars]
    edgemaps = [spacevals[e...] for e in edges]
    initmaps = substitute.(pdesys.depvars,[t=>tspan[1]])

    depvarmaps = reduce(vcat,[substitute.((pdesys.depvars[i],),edgevals) .=> edgevars[i] for i in 1:length(pdesys.depvars)])
    if length(nottime) == 1
        left_weights(j) = DiffEqOperators.calculate_weights(discretization.upwind_order, 0.0, [space[j][1],space[j][2]])
        right_weights(j) = DiffEqOperators.calculate_weights(discretization.upwind_order, 0.0, [space[j][end-1],space[j][end]])
        central_neighbor_idxs(i,j) = [i+CartesianIndex([ifelse(l==j,-1,0) for l in 1:length(nottime)]...),i,i+CartesianIndex([ifelse(l==j,1,0) for l in 1:length(nottime)]...)]
        derivars = [[dot(left_weights(j),[depvars[j][central_neighbor_idxs(CartesianIndex(2),1)[1:2][2]],depvars[j][central_neighbor_idxs(CartesianIndex(2),1)[1:2][1]]]),
                    dot(right_weights(j),[depvars[j][central_neighbor_idxs(CartesianIndex(length(space[1])-1),1)[end-1:end][1]],depvars[j][central_neighbor_idxs(CartesianIndex(length(space[1])-1),1)[end-1:end][2]]])]
                    for j in 1:length(pdesys.depvars)]
        depvarderivmaps = reduce(vcat,[substitute.((Differential(nottime[j])(pdesys.depvars[i]),),edgevals) .=> derivars[i]
                                       for i in 1:length(pdesys.depvars) for j in 1:length(nottime)])
   else
       # TODO: Fix Neumann and Robin on higher dimension
       depvarderivmaps = []
   end

    # Generate initial conditions and bc equations
    u0 = []
    bceqs = []
    for bc in pdesys.bcs
        if ModelingToolkit.operation(bc.lhs) isa Sym && t.val ∉ ModelingToolkit.arguments(bc.lhs)
            # initial condition
            # Assume in the form `u(...) ~ ...` for now
            push!(u0,vec(depvars[findfirst(isequal(bc.lhs),initmaps)] .=> substitute.((bc.rhs,),spacevals)))
        else
            # Algebraic equations for BCs
            i = findfirst(x->occursin(x.val,bc.lhs),first.(depvarmaps))

            # TODO: Fix Neumann and Robin on higher dimension
            lhs = length(nottime) == 1 ? substitute(bc.lhs,depvarderivmaps[i]) : bc.lhs

            lhs = substitute(lhs,depvarmaps[i])
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
        valrules = vcat([pdesys.depvars[k] => depvars[k][i] for k in 1:length(pdesys.depvars)],
                        [nottime[k] => space[k][i[k]] for k in 1:length(nottime)])

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
    defaults = pdesys.ps === nothing || pdesys.ps === SciMLBase.NullParameters() ? u0 : vcat(u0,pdesys.ps)
    ps = pdesys.ps === nothing || pdesys.ps === SciMLBase.NullParameters() ? Num[] : first.(pdesys.ps)
    sys = ODESystem(vcat(eqs,unique(bceqs)),t,vec(reduce(vcat,vec(depvars))),ps,defaults=Dict(defaults))
    sys, tspan
end

function SciMLBase.discretize(pdesys::ModelingToolkit.PDESystem,discretization::DiffEqOperators.MOLFiniteDifference)
    sys, tspan = SciMLBase.symbolic_discretize(pdesys,discretization)
    simpsys = structural_simplify(sys)
    prob = ODEProblem(simpsys,Pair[],tspan)
end

# Piracy, to be deleted when https://github.com/JuliaSymbolics/SymbolicUtils.jl/pull/251
# merges
Base.occursin(needle::ModelingToolkit.SymbolicUtils.Symbolic, haystack::ModelingToolkit.SymbolicUtils.Symbolic) = _occursin(needle, haystack)
Base.occursin(needle, haystack::ModelingToolkit.SymbolicUtils.Symbolic) = _occursin(needle, haystack)
Base.occursin(needle::ModelingToolkit.SymbolicUtils.Symbolic, haystack) = _occursin(needle, haystack)
function _occursin(needle, haystack)
    isequal(needle, haystack) && return true

    if istree(haystack)
        args = arguments(haystack)
        for arg in args
            occursin(needle, arg) && return true
        end
    end
    return false
end
