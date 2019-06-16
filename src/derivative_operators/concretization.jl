function LinearAlgebra.Array(A::DerivativeOperator{T}, N::Int=A.len) where T
    L = zeros(T, N, N+2)
    bl = A.boundary_point_count
    stl = A.stencil_length
    bstl = A.boundary_stencil_length
    coeff   = A.coefficients
    stl_2 = div(stl,2)
    for i in 1:A.boundary_point_count
        cur_coeff   = typeof(coeff)   <: AbstractVector ? coeff[i] : true
        cur_stencil = cur_coeff < 0 && A.use_winding ? reverse(A.low_boundary_coefs[i]) : A.low_boundary_coefs[i]
        L[i,1:bstl] = cur_coeff * cur_stencil
    end
    for i in bl+1:N-bl
        cur_coeff   = typeof(coeff)   <: AbstractVector ? coeff[i] : true
        stencil     = eltype(A.stencil_coefs) <: AbstractVector ? A.stencil_coefs[i] : A.stencil_coefs
        cur_stencil = cur_coeff < 0 && A.use_winding ? reverse(stencil) : stencil
        L[i,i+1-stl_2:i+1+stl_2] = cur_coeff * cur_stencil
    end
    for i in N-bl+1:N
        cur_coeff   = typeof(coeff)   <: AbstractVector ? coeff[i] : true
        cur_stencil = cur_coeff < 0 && A.use_winding ? reverse(A.high_boundary_coefs[i-N+bl]) : A.high_boundary_coefs[i-N+bl]
        L[i,N-bstl+3:N+2] = cur_coeff * cur_stencil
    end
    return L / A.dx^A.derivative_order
end

function SparseArrays.SparseMatrixCSC(A::DerivativeOperator{T}, N::Int=A.len) where T
    L = spzeros(T, N, N+2)
    bl = A.boundary_point_count
    stl = A.stencil_length
    stl_2 = div(stl,2)
    bstl = A.boundary_stencil_length
    coeff   = A.coefficients
    for i in 1:A.boundary_point_count
        cur_coeff   = typeof(coeff)   <: AbstractVector ? coeff[i] : true
        cur_stencil = cur_coeff < 0 && A.use_winding ? reverse(A.low_boundary_coefs[i]) : A.low_boundary_coefs[i]
        L[i,1:bstl] = cur_coeff * cur_stencil
    end
    for i in bl+1:N-bl
        cur_coeff   = typeof(coeff)   <: AbstractVector ? coeff[i] : true
        stencil     = eltype(A.stencil_coefs) <: AbstractVector ? A.stencil_coefs[i] : A.stencil_coefs
        cur_stencil = cur_coeff < 0 && A.use_winding ? reverse(stencil) : stencil
        L[i,i+1-stl_2:i+1+stl_2] = cur_coeff * cur_stencil
    end
    for i in N-bl+1:N
        cur_coeff   = typeof(coeff)   <: AbstractVector ? coeff[i] : true
        cur_stencil = cur_coeff < 0 && A.use_winding ? reverse(A.high_boundary_coefs[i-N+bl]) : A.high_boundary_coefs[i-N+bl]
        L[i,N-bstl+3:N+2] = cur_coeff * cur_stencil
    end
    return L / A.dx^A.derivative_order
end

function SparseArrays.sparse(A::AbstractDerivativeOperator{T}, N::Int=A.len) where T
    SparseMatrixCSC(A,N)
end

function BandedMatrices.BandedMatrix(A::DerivativeOperator{T}, N::Int=A.len) where T
    bl = A.boundary_point_count
    stl = A.stencil_length
    bstl = A.boundary_stencil_length
    coeff   = A.coefficients
    stl_2 = div(stl,2)
    L = BandedMatrix{T}(Zeros(N, N+2), (max(stl-3,0,bstl),max(stl-1,0,bstl)))
    for i in 1:A.boundary_point_count
        cur_coeff   = typeof(coeff)   <: AbstractVector ? coeff[i] : true
        cur_stencil = cur_coeff < 0 && A.use_winding ? reverse(A.low_boundary_coefs[i]) : A.low_boundary_coefs[i]
        L[i,1:bstl] = cur_coeff * cur_stencil
    end
    for i in bl+1:N-bl
        cur_coeff   = typeof(coeff)   <: AbstractVector ? coeff[i] : true
        stencil     = eltype(A.stencil_coefs) <: AbstractVector ? A.stencil_coefs[i] : A.stencil_coefs
        cur_stencil = cur_coeff < 0 && A.use_winding ? reverse(stencil) : stencil
        L[i,i+1-stl_2:i+1+stl_2] = cur_coeff * cur_stencil
    end
    for i in N-bl+1:N
        cur_coeff   = typeof(coeff)   <: AbstractVector ? coeff[i] : true
        cur_stencil = cur_coeff < 0 && A.use_winding ? reverse(A.high_boundary_coefs[i-N+bl]) : A.high_boundary_coefs[i-N+bl]
        L[i,N-bstl+3:N+2] = cur_coeff * cur_stencil
    end
    return L / A.dx^A.derivative_order
end

function Base.convert(::Type{Array},A::DerivativeOperator{T}) where T
    Array(A)
end

function Base.convert(::Type{SparseMatrixCSC},A::DerivativeOperator{T}) where T
    SparseMatrixCSC(A)
end

function Base.convert(::Type{BandedMatrix},A::DerivativeOperator{T}) where T
    BandedMatrix(A)
end

function Base.convert(::Type{AbstractMatrix},A::DerivativeOperator{T}) where T
    BandedMatrix(A)
end
