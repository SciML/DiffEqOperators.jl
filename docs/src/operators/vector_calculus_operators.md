# Vector Calculus Operators

A good way to represent physical vectors is by storing them as a space tensor with each entry
taking the form `[u₁ u₂ u₃ .... uₙ]` for a `n`-dimensional physical vector, each index 
holding its component along that direction. Defining such entries at all grid points will lead to
creation of a n-dim matrix.

Various operators and functions have been introduced to carry out common calculus operations like 
`Gradient`, `Curl` , `norm` etc. for those.

#### Operators

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
These can then be used as `A*u`, `A` holding our constructor and `u` being the input `N`-dim `Array`,
either representing a multi-variable function which would be compatible with `Gradient` or
the Tensor representation desribe earlier, holding our physical vector compatible with `Divergence` and `Curl`.  

The arguements are :

- `approximation_order` : the order of the discretization in terms of O(dx^order).
- `dx` : tuple containing the spacing of the discretization in order of dimensions.
   When `dx` has `eltype <: Number`, that would imply uniform discretization, while it also
   supports `eltype <: Array{Number}` for non-uniform grids.
- `len`: tuple storing length of the discretization in the respective directions.
- `coeff_func`: An operational argument which sets the coefficients of the operator.
  If `coeff_func` is a `Number`, then the coefficients are set to be constant with that number.
  If `coeff_func` is an `AbstractArray` with length matching `len`, then the coefficients
  are constant but spatially dependent.

#### Functions

Some common functions used in Vector calculus have been provided :

```julia
dot(A::AbstractArray{Array{T,1},N},B::AbstractArray{Array{T,1},N})
dot!(u::AbstractArray{T,N}, A::AbstractArray{Array{T,1},N},B::AbstractArray{Array{T,1},N})

cross(A::AbstractArray{Array{T,1},3},B::AbstractArray{Array{T,1},3})
cross!(u::AbstractArray{Array{T,1},3},A::AbstractArray{Array{T,1},3},B::AbstractArray{Array{T,1},3})

norm(A::AbstractArray{Array{T,1},N})
norm!(u::AbstractArray{T,N},A::AbstractArray{Array{T,1},N})
```

`A` and  `B` are Tensors of same sizes. The output would be a `N`-dim Matrix storing the corresponding
value of operation at each grid point. All of these support inplace operations with `!` notation as described above.
