
function dot_product(A::AbstractArray{Array{T,1},N},B::AbstractArray{Array{T,1},N}) where {T<:Real,N}
    size(A) === size(B) || throw(ArgumentError("Vectors must have the same shape"))
    u = Array{T,N}(undef,size(A)...)
    
    for I in CartesianIndices(u)
        u[I] = zero(T)
        for i in 1:N
            u[I] += A[I][i]*B[I][i]
        end
    end

    return u
end

function dot_product!(u::AbstractArray{T,N}, A::AbstractArray{Array{T,1},N},B::AbstractArray{Array{T,1},N}) where {T<:Real,N}
    (size(A) === size(B) && size(A) === size(u)) || throw(ArgumentError("Vectors must have the same shape"))
    
    for I in CartesianIndices(u)
        u[I] =  A[I][1]*B[I][1]
        for i in 2:N
            u[I] += A[I][i]*B[I][i]
        end
    end

    return u
end

function cross_product(A::AbstractArray{Array{T,1},3},B::AbstractArray{Array{T,1},3}) where {T<:Real}
    size(A) === size(B) || throw(ArgumentError("Vectors must have the same shape"))
    u = similar(A)
    
    for I in CartesianIndices(u)
        u[I] = zeros(T,3)
        for i in 1:3
            u[I][i] = A[I][i%3 + 1]*B[I][(i+1)%3 + 1] - A[I][(i+1)%3 + 1]*B[I][i%3 + 1]
        end
    end

    return u
end

function cross_product!(u::AbstractArray{Array{T,1},3},A::AbstractArray{Array{T,1},3},B::AbstractArray{Array{T,1},3}) where {T<:Real}
    (size(A) === size(B) && size(A) === size(u))|| throw(ArgumentError("Vectors must have the same shape"))
    
    for I in CartesianIndices(u)
        u[I] = zeros(T,3)
        for i in 1:3
            u[I][i] = A[I][i%3 + 1]*B[I][(i+1)%3 + 1] - A[I][(i+1)%3 + 1]*B[I][i%3 + 1]
        end
    end

    return u
end

function square_norm(A::AbstractArray{Array{T,1},N}) where {T<:Real,N}
    return dot_product(A,A).^(0.5)
end

function square_norm!(u::AbstractArray{T,N},A::AbstractArray{Array{T,1},N}) where {T<:Real,N}
    dot_product!(u,A,A)
    u .= u.^0.5
end