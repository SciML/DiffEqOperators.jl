# Vector Calculus Operators

A good way to represent physical vectors is by storing them as a `N+1` dimensional matrix for a `N`-dimensional physical vector, with each last index `i` storing the iᵗʰ component of it at grid point specified by the indices prior to it. For e.g., `u[p,q,r,2]` stores the 2ⁿᵈ component at `x[p], y[q], z[r]`. 

Various operators and functions have been introduced here to carry out common calculus operations like 
`Gradient`, `Curl` , `square_norm` etc. for them.

## Operators

All operators store `CenteredDifference` operators along various axes for computing the underlying 
derivatives lazily. They differ in the way convolutions are performed.

Following are the constructors :

```julia
Gradient(approximation_order :: Int,
        dx::Union{NTuple{N,AbstractVector},NTuple{N,T}},
        len::NTuple{N,Int}, coeff_func=nothing)

Curl(approximation_order :: Int,
     dx::Union{NTuple{3,AbstractVector},NTuple{3,T}},
     len::NTuple{3,Int}, coeff_func = nothing )

Divergence(approximation_order :: Int,
           dx::Union{NTuple{N,AbstractVector},NTuple{N,T}},
           len::NTuple{N,Int}, coeff_func=nothing)
```
These can then be used as `A*u`, `A` holding our constructor and `u` being the input `Array`,
either representing a multi-variable function in a `N`-dim Tensor, which would be compatible with `Gradient` or
the `N+1`-dim Tensor representation desribed earlier, holding our physical vector compatible with `Divergence` and `Curl`.  

The arguments are :

- `approximation_order` : the order of the discretization in terms of O(dx^order).
- `dx` : tuple containing the spacing of the discretization in order of dimensions.
   When `dx` has `eltype <: Number`, that would imply uniform discretization, while it also
   supports `eltype <: Array{Number}` for non-uniform grids.
- `len`: tuple storing length of the discretization in the respective directions.
- `coeff_func`: An operational argument which sets the coefficients of the operator.
  If `coeff_func` is a `Number`, then the coefficients are set to be constant with that number.
  If `coeff_func` is an `AbstractArray` with length matching `len`, then the coefficients
  are constant but spatially dependent.

## Functions

Some common functions used in Vector calculus that have been made available are :

```julia
dot_product(A::AbstractArray{T1,N},B::AbstractArray{T2,N})
dot_product!(u::AbstractArray{T1,N}, A::AbstractArray{T2,N2},B::AbstractArray{T3,N2})

cross_product(A::AbstractArray{T1,4},B::AbstractArray{T2,4})
cross_product!(u::AbstractArray{T1,4},A::AbstractArray{T2,4},B::AbstractArray{T3,4})

square_norm(A::AbstractArray{T,N})
square_norm!(u::AbstractArray{T1,N1},A::AbstractArray{T2,N2})
```

`A` and  `B` are `N+1`-dim Tensors of same sizes. The output would be a `N`-dim Tensor storing the corresponding
value of operation at each grid point. All of these support inplace operations with `!` notation as described above.

`dot_product` translates to `A ⋅ B`, `cross_product` to `A × B` and `square_norm` to `L2-norm` in real sense.