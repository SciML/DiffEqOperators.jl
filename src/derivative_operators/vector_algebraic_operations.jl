# Common functions compatible for use with physical vectors

function dot_product(A::AbstractArray{T1,N},B::AbstractArray{T2,N}) where {T1<:Number,T2<:Number,N}
    
    size(A) === size(B) || throw(ArgumentError("Vectors must have the same shape"))
    u = zeros(promote_type(T1,T2),size(A)[1:end-1]...)
    dot_product!(u,A,B)
    return u
end

function dot_product!(u::AbstractArray{T1,N}, A::AbstractArray{T2,N2},B::AbstractArray{T3,N2}) where {T1<:Number,T2<:Number,T3<:Number,N,N2}
    
    (size(A) === size(B)) || throw(ArgumentError("Vectors must have the same shape"))
    (N === N2-1) || throw(ArgumentError("Output should be scalar matrix, one dimension less than input"))
    
    T = promote_type(T1,T2,T3)
    @turbo for I in CartesianIndices(u)
        x_temp = zero(T)
        for i in 1:N
            x_temp += A[I,i]*B[I,i]
        end
        u[I] = x_temp
    end
    return u
end

function cross_product(A::AbstractArray{T1,4},B::AbstractArray{T2,4}) where {T1<:Number,T2<:Number}
    
    size(A) === size(B) || throw(ArgumentError("Vectors must have the same shape"))
    s = size(A)
    u = zeros(promote_type(T1,T2),s...)
    cross_product!(u,A,B)
    return u
end

function cross_product!(u::AbstractArray{T1,4},A::AbstractArray{T2,4},B::AbstractArray{T3,4}) where {T1<:Number,T2<:Number,T3<:Number}
    
    (size(A) === size(B) && size(A) === size(u))|| throw(ArgumentError("Vectors must have the same shape"))
    s = size(u)

    @turbo for r in 1:s[3], q in 1:s[2], p in 1:s[1]
        u[p,q,r,1] = A[p,q,r,2]*B[p,q,r,3] - A[p,q,r,3]*B[p,q,r,2]
        u[p,q,r,2] = A[p,q,r,3]*B[p,q,r,1] - A[p,q,r,1]*B[p,q,r,3]
        u[p,q,r,3] = A[p,q,r,1]*B[p,q,r,2] - A[p,q,r,2]*B[p,q,r,1]
    end
end

function square_norm(A::AbstractArray{T,N}) where {T<:Number,N}
    u = zeros(T,size(A)[1:end-1]...)
    square_norm!(u,A)
    return u
end

function square_norm!(u::AbstractArray{T1,N1},A::AbstractArray{T2,N2}) where {T1<:Number,T2<:Number,N1,N2}
    dot_product!(u,A,A)
    @. u = u^0.5
end