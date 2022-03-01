# DiffEqOperators.jl

DiffEqOperators.jl is a package for finite difference discretization of partial
differential equations. It is for building fast lazy operators for high order
non-uniform finite differences.

!!! note

    For automated finite difference discretization of symbolically-defined PDEs,
    see [MethodOfLines.jl](https://github.com/SciML/MethodOfLines.jl).


!!! warn

    This library is not complete, especially for higher dimensional operators.
    Use with caution.
    
For the operators, both centered and
[upwind](https://en.wikipedia.org/wiki/Upwind_scheme) operators are provided,
for domains of any dimension, arbitrarily spaced grids, and for any order of accuracy.
The cases of 1, 2, and 3 dimensions with an evenly spaced grid are optimized with a
convolution routine from `NNlib.jl`. Care is taken to give efficiency by avoiding
unnecessary allocations, using purpose-built stencil compilers, allowing GPUs
and parallelism, etc. Any operator can be concretized as an `Array`, a
`BandedMatrix` or a sparse matrix.
