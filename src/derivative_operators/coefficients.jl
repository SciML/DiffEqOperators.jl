"""
```
init_coefficients(coeff_func, len::Int)
```

Return the initial value of an operator's `coefficients` field based on the type
of `coeff_func`.
"""
init_coefficients(coeff_func::Nothing, len::Int) = nothing

init_coefficients(coeff_func::Number, len::Int) = coeff_func * ones(typeof(coeff_func), len)

function init_coefficients(coeff_func::AbstractVector{T}, len::Int) where T <: Number
    coeff_func
end

init_coefficients(coeff_func::Function, len::Int) = ones(Float64, len)



"""
```
get_coefficient(coefficients, index)
```
"""
get_coefficient(coefficients::AbstractVector, index::Int) = coefficients[index]

# FIXME: I don't think this case is used anymore
get_coefficient(coefficients::Number, index::Int) = coefficients

# FIXME: Why use "true" here for the value 1?
get_coefficient(coefficients::Nothing, index::Int) = true
