# Matrix-Free Operators

```julia
MatrixFreeOperator(f::F, args::N;
                   size=nothing, opnorm=true, ishermitian=false) where {F,N}
```

A `MatrixFreeOperator` is a linear operator `A*u` where the action of `A` is
explicitly defined by an in-place function `f(du, u, p, t)`.
