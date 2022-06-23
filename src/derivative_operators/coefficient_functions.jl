"""
```
compute_coeffs(coeff_func, current_coeffs)
```
Calculates the coefficients for the stencil of UpwindDifference operators.
"""
function compute_coeffs!(
    coeff_func::Number,
    current_coeffs::AbstractVector{T},
) where {T<:Number}
    return current_coeffs .+= coeff_func
end

function compute_coeffs!(
    coeff_func::AbstractVector{T},
    current_coeffs::AbstractVector{T},
) where {T<:Number}
    return current_coeffs[:] += coeff_func
end

# Coefficient functions when coeff_func is a Function and current_coeffs exists
function compute_coeffs!(
    coeff_func::Function,
    current_coeffs::AbstractVector{T},
) where {T<:Number}
    if hasmethod(coeff_func, (Vector{T},))
        current_coeffs[:] = coeff_func(current_coeffs)
    else
        map!(coeff_func, current_coeffs, current_coeffs)
    end
    return current_coeffs
    # if hasmethod(coeff_func, (Int,)) # assume we want to provide the index of coefficients to coeff_func
    #     for i = 1:length(current_coeffs)
    #         current_coeffs[i] += coeff_func(i)
    #     end
    #     return current_coeffs
    # elseif hasmethod(coeff_func, (Vector{Int},))  # assume we want to provide the index of coefficients to coeff_func
    #     current_coeffs[:] += coeff_func(collect(1:length(current_coeffs)))
    #     return current_coeffs
    # elseif hasmethod(coeff_func, (UnitRange{Int},)) # assume we want to provide the index of coefficients to coeff_func
    #     current_coeffs[:] += coeff_func(1:length(current_coeffs))
    #     return current_coeffs
    # elseif hasmethod(coeff_func, (Float64,)) # assume we want coeff_func to operate on existing coefficients
    #     map!(coeff_func, current_coeffs, current_coeffs)
    #     return current_coeffs
    # elseif hasmethod(coeff_func, (Vector{Float64},)) # assume we want to coeff_func to operate on existing coefficients
    #     current_coeffs[:] = coeff_func(current_coeffs)
    #     return current_coeffs
    # else
    #     error("Coefficient functions with the arguments of $coeff_func have not been implemented.")
    # end
end

compute_coeffs(coeff_func::Number, current_coeffs::AbstractVector{T}) where {T<:Number} =
    current_coeffs .+ coeff_func
compute_coeffs(
    coeff_func::AbstractVector{T},
    current_coeffs::AbstractVector{T},
) where {T<:Number} = coeff_func + current_coeffs

function compute_coeffs(
    coeff_func::Function,
    current_coeffs::AbstractVector{T},
) where {T<:Number}
    if hasmethod(coeff_func, (Vector{T},))
        return coeff_func(current_coeffs)
    else
        return map(coeff_func, current_coeffs)
    end
    # if hasmethod(coeff_func, (Int,))
    #     return current_coeffs + map(coeff_func, collect(1:length(current_coeffs)))
    # elseif hasmethod(coeff_func, (Vector{Int},))
    #     return current_coeffs + coeff_func(collect(1:length(current_coeffs)))
    # elseif hasmethod(coeff_func, (UnitRange{Int},))
    #     return current_coeffs + coeff_func(1:length(current_coeffs))
    # elseif hasmethod(coeff_func, (Float64,)) # assume we want coeff_func to operate on existing coefficients
    #     return map(coeff_func, current_coeffs)
    # elseif hasmethod(coeff_func, (Vector{Float64},)) # assume we want to coeff_func to operate on existing coefficients
    #     return coeff_func(current_coeffs)
    # else
    #     error("Coefficient functions with the arguments of coeff_func $coeff_func have not been implemented.")
    # end
end
