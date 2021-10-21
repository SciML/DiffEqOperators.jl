# Vector-Jacobian Product Operators

```julia
VecJacOperator{T}(f,u::AbstractArray,p=nothing,t::Union{Nothing,Number}=nothing;autodiff=true,ishermitian=false,opnorm=true)
```

The `VecJacOperator` is a linear operator `J'*v` where `J` acts like `df/du`
for some function `f(u,p,t)`. For in-place operations `mul!(w,J,v)`, `f`
is an in-place function `f(du,u,p,t)`.

!!! note
    This operator is available when `Zygote` is imported.