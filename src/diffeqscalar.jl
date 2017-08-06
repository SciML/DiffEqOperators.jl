"""
DiffEqScalar Interface

DiffEqScalar(func,coeff=1.0)

This is a function with a coefficient.

α(t) returns a new DiffEqScalar with an updated coefficient.
"""
struct DiffEqScalar{F,T}
    func::F
    coeff::T
    DiffEqScalar{T}(func) where T = new{typeof(func),T}(func,one(T))
    DiffEqScalar{F,T}(func,coeff) where {F,T} = new{F,T}(func,coeff)
end
DiffEqScalar(func,coeff=1.0) = DiffEqScalar{typeof(func),typeof(coeff)}(func,coeff)
function (α::DiffEqScalar)(t)
    if α.func == nothing
        return DiffEqScalar(α.func,α.coeff)
    else
        return DiffEqScalar(α.func,α.func(t))
    end
end
Base.:*(α::Number,B::DiffEqScalar) = DiffEqScalar(B.func,B.coeff*α)
Base.:*(B::DiffEqScalar,α::Number) = DiffEqScalar(B.func,B.coeff*α)
