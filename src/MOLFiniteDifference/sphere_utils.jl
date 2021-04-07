#
# Temporary file with utilities for symbolic discretization of the sphere domain
#
using ModelingToolkit: AbstractDomain, @variables, Differential

# This type should go into ModelingToolkit.jl
struct AxisymmetricSphereDomain{T} <: AbstractDomain{T,1}
    lower::T
    upper::T
end
