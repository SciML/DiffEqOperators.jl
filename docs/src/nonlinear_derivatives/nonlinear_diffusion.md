# Nonlinear Diffusion

This function handles expressions of the form `ðₙ(D(ðₘu))` where `n,m > 0` and 
`D` is a function of `u` i.e. they vary as `u(x,t)` and `D(u)`. The expansion can be carried out via 
[general Leibniz rule](https://en.wikipedia.org/wiki/General_Leibniz_rule).

A boundary condition operator `bc` is first operated on `u` resulting in a 
boundary padded vector `bc*u`. Since `D` is a function of `u`, its discrete values 
can be obtained at grid points once `u` has been padded.  After producing these 
two functions in the grid range, we can expand the given expression via
binomial expansion through the `nonlinear_diffusion` and 
`nonlinear_diffusion!` functions and produce the final discretized derivatives.

![Expressions for general Leibnuz rule with varying m](https://github.com/SciML/DiffEqOperators.jl/raw/master/binomial_expansion.svg)

The functions implicitly put the `CenteredDifference` operator 
to use for computing derivates of various orders, e.g. 
`uᵏ = CenteredDifference(k,approx_order,dx,nknots)*u`, helping us
generate a symmetric discretization. The two functions differ in terms of memory
allocation, since the non-`!` one will allocate memory to the output whereas
the `!` one can be used for non-allocating applications.


## Functions

The two functions are as follows :

```julia
nonlinear_diffusion(second_differential_order::Int, first_differential_order::Int, approx_order::Int,
                    p::AbstractVector{T}, q::AbstractVector{T}, dx::Union{T , AbstractVector{T} , Real},
                    nknots::Int) where {T<:Real, N}

nonlinear_diffusion!(du::AbstractVector{T}, second_differential_order::Int, first_differential_order::Int,
                     approx_order::Int,p::AbstractVector{T}, q::AbstractVector{T},
                     dx::Union{T , AbstractVector{T} , Real}, nknots::Int) where {T<:Real, N}
```

Arguments :

- `du` : an input `AbstractVector` similar to `u`, to store the final discretized expression.
- `second_differential_order` : the overall order of derivative on the expression.`(n)`
- `first_differential_order` : the inner order of derivative to discretize for `u`.`(m)`
- `approx_order` : the order of the discretization in terms of O(dx^order).
- `p` : boundary padded `D`.
- `q` : boundary padded `u` obtained by `bc*u`.
- `dx`: spacing of the discretization. If `dx` is a `Number`, the discretization
        is uniform. If `dx` is an array, then the discretization is non-uniform.
- `nknots` : the length of discretization in the direction of operator.
