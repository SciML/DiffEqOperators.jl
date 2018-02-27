## Description of General Boundary Values for Composed Operators
- **TODO:** document the operators for boundary values and restriction operators https://github.com/JuliaDiffEq/DifferentialEquations.jl/issues/260#issuecomment-368544485

## General Description
- TODO: Explain the R, Q, B, etc. for the general notaiton
- TODO: We will want to make a set of simple examples

## Examples

### Zero Neumann Heat Operator ###
https://github.com/JuliaDiffEq/DifferentialEquations.jl/issues/260#issuecomment-368533633

See [Zero-Neumman Heat](operator_examples/zero_neumann_heat.jl)
Some example code from that for generating the operator.
```
B = [-1 1 zeros(1,n-4)  0 0;
      0 0 zeros(1,n-4) -1 1]/Δx

Δ = Toeplitz([1; zeros(n-3)], [1; -2; 1; zeros(n-3)])/Δx^2
Ã = Δ - Δ[:,[1,n]]*(B[:,[1,n]]\B)
A = Ã[:,2:n-1]
```

Also, see [Simple HJBE with Reflecting Barriers](operator_examples\simple_stationary_HJBE_reflecting.jl)
