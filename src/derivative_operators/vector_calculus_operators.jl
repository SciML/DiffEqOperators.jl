# Structures of Vector calculus operators store all the operators required during computations.

# Operator for calculating gradients of a N-dimensional function

struct GradientOperator{T,N,O}
    ops :: O
end

function Gradient(approximation_order :: Int, dx::Union{NTuple{N,AbstractVector{T}},NTuple{N,T}},
        len::NTuple{N,Int}, coeff_func=nothing) where {T<:Real, N}
    
    ops = permutedims([CenteredDifference{n}(1, approximation_order, dx[n], len[n], coeff_func) for n in 1:N])

    GradientOperator{T,N,typeof(ops)}(
        ops
        )
end

# Operator for calculating curls of a given 3-dimensional vector stored as space-tensor

struct CurlOperator{T,O}
    ops :: O
end 

function Curl(approximation_order :: Int, dx::Union{NTuple{3,AbstractVector{T}},NTuple{3,T}},
    len::NTuple{3,Int}, coeff_func = nothing ) where {T<:Real}

    @assert len[1] == len[2] && len[2] == len[3] "All dimensions must have equal no. of grid points"
    ops = permutedims([CenteredDifference{n}(1, approximation_order, dx[n], len[n], coeff_func) for n in 1:3])

    CurlOperator{T,typeof(ops)}(
        ops
        )

end

# Operator for calculating divergence of a given N-dimensional vector stored as space-tensor

struct DivergenceOperator{T,N,O}
    ops :: O
end

function Divergence(approximation_order :: Int, dx::Union{NTuple{N,AbstractVector{T}},NTuple{N,T}},
    len::NTuple{N,Int}, coeff_func=nothing) where {T<:Real, N}

    ops = permutedims([CenteredDifference{n}(1, approximation_order, dx[n], len[n], coeff_func) for n in 1:N])

    DivergenceOperator{T,N,typeof(ops)}(
        ops
    )

end

function *(A::GradientOperator{T},M::AbstractArray{T,N}) where {T<:Real,N}
    
    size_x_temp = [size(M)...].-2

    x_temp = Array{Array{T,1},length(A.ops)}(undef,size_x_temp...)
    
    for I in CartesianIndices(x_temp)
        x_temp[I] = zeros(T,length(A.ops))
    end

    for L in A.ops
        mul!(x_temp, true, L, M)
    end

    return x_temp
end

function *(A::CurlOperator{T},M::AbstractArray{Array{T,1},3}) where {T<:Real}
    
    size_x_temp = [size(M)...].-2

    x_temp = similar(M,size_x_temp...)
    
    for I in CartesianIndices(x_temp)
        x_temp[I] = zeros(T,3)
    end

    mul!(x_temp, A, M,overwrite = false)

    return x_temp
end

function *(A::DivergenceOperator{T},M::AbstractArray{Array{T,1},N}) where {T<:Real, N}
    
    size_x_temp = [size(M)...].-2

    x_temp = Array{T,length(A.ops)}(undef,size_x_temp...)
    
    for I in CartesianIndices(x_temp)
        x_temp[I] = zero(T)
    end

    for L in A.ops
        mul!(x_temp, true, L, M)
    end

    return x_temp
end

function *(c::Number, A::GradientOperator{T,N}) where {T,N} 
    ops = permutedims([c*A.ops[i] for i in 1:N])
    GradientOperator{T,N,typeof(ops)}(
        ops
        )
end

function *(c::Number, A::CurlOperator{T}) where {T} 
    ops = permutedims([c*A.ops[i] for i in 1:3])
    CurlOperator{T,typeof(ops)}(
        ops
        )
end

function *(c::Number, A::DivergenceOperator{T,N}) where {T,N} 
    ops = permutedims([c*A.ops[i] for i in 1:N])
    DivergenceOperator{T,N,typeof(ops)}(
        ops
        )
end