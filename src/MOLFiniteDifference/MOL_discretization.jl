using ModelingToolkit: operation, istree, arguments, variable
import DomainSets

# Method of lines discretization scheme
struct MOLFiniteDifference{T,T2} <: DiffEqBase.AbstractDiscretization

end

MOLFiniteDifference(args...) = throw(ErrorException("MOLFiniteDifference has been moved to the MethodOfLines packgage, please use MethodOfLines.MOLFiniteDifference instead"))

# Constructors. If no order is specified, both upwind and centered differences will be 2nd order
MOLFiniteDifference(dxs, time; upwind_order = 1, centered_order = 2, grid_align=center_align) = throw(ErrorException("MOLFiniteDifference has been moved to the MethodOfLines packgage, please install with `Pkg.add(\"MethodOfLines\")` and use MethodOfLines.MOLFiniteDifference instead"))


function SciMLBase.symbolic_discretize(pdesys::ModelingToolkit.PDESystem,discretization::DiffEqOperators.MOLFiniteDifference)
    throw(ErrorException("Symbolic Method of Lines discretization has been moved to the MethodOfLines packgage, please install this instead with `Pkg.add(\"MethodOfLines\")`"))
end

function SciMLBase.discretize(pdesys::ModelingToolkit.PDESystem,discretization::DiffEqOperators.MOLFiniteDifference)
    throw(ErrorException("Symbolic Method of Lines discretization has been moved to the MethodOfLines packgage, please install this instead with `Pkg.add(\"MethodOfLines\")`"))
end
