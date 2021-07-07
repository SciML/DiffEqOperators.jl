# Operator Overview

The operators in DiffEqOperators.jl are instantiations of the `AbstractSciMLOperator`
interface. This is documented in [SciMLBase](https://juliahub.com/docs/SciMLBase/jigfq/1.8.1/autodocs/#SciMLBase.AbstractSciMLOperator). Thus each of the operators
have the functions and traits as defined for the operator interface. In addition,
the DiffEqOperators.jl operators satisfy the following properties:

1. Derivative * Boundary gives a GhostDerivative operator, representing a
   derivative operator which respects boundary conditions
2. Boundary conditions generate extended vectors in a non-allocating form
3. Operators can be concretized into matrices

## Operator Compositions

Multiplying two DiffEqOperators will build a `DiffEqOperatorComposition`, while
adding two DiffEqOperators will build a `DiffEqOperatorCombination`. Multiplying
a DiffEqOperator by a scalar will produce a `DiffEqScaledOperator`. All
will inherit the appropriate action.

### Efficiency of Composed Operator Actions

Composed operator actions utilize NNLib.jl in order to do cache-efficient
convolution operations in higher-dimensional combinations.
