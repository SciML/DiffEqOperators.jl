function LinearAlgebra.Array(A::DerivativeOperator{T}, N::Int=A.dimension) where T
    L = zeros(T, N, N+2)
    bl = A.boundary_point_count
    stl = A.stencil_length
    bstl = A.boundary_stencil_length
    stl_2 = div(stl,2)
    for i in 1:A.boundary_point_count
        L[i,1:bstl] = A.low_boundary_coefs[i]
    end
    for i in bl+1:N-bl
        L[i,i+1-stl_2:i+1+stl_2] = A.stencil_coefs
    end
    for i in N-bl+1:N
        L[i,N-bstl+3:N+2] = A.high_boundary_coefs[i-N+bl]
    end
    return L / A.dx^A.derivative_order
end

function SparseArrays.SparseMatrixCSC(A::DerivativeOperator{T}, N::Int=A.dimension) where T
    L = spzeros(T, N, N+2)
    bl = A.boundary_point_count
    stl = A.stencil_length
    stl_2 = div(stl,2)
    bstl = A.boundary_stencil_length
    for i in 1:A.boundary_point_count
        L[i,1:bstl] = A.low_boundary_coefs[i]
    end
    for i in bl+1:N-bl
        L[i,i+1-stl_2:i+1+stl_2] = A.stencil_coefs
    end
    for i in N-bl+1:N
        L[i,N-bstl+3:N+2] = A.high_boundary_coefs[i-N+bl]
    end
    return L / A.dx^A.derivative_order
end

function SparseArrays.sparse(A::AbstractDerivativeOperator{T}, N::Int=A.dimension) where T
    SparseMatrixCSC(A,N)
end

function BandedMatrices.BandedMatrix(A::DerivativeOperator{T}, N::Int=A.dimension) where T
    N = A.dimension
    bl = A.boundary_point_count
    stl = A.stencil_length
    bstl = A.boundary_stencil_length
    stl_2 = div(stl,2)
    L = BandedMatrix{T}(Zeros(N, N+2), (max(stl-3,0,bstl),max(stl-1,0,bstl)))
    for i in 1:A.boundary_point_count
        L[i,1:bstl] = A.low_boundary_coefs[i]
    end
    for i in bl+1:N-bl
        L[i,i+1-stl_2:i+1+stl_2] = A.stencil_coefs
    end
    for i in N-bl+1:N
        L[i,N-bstl+3:N+2] = A.high_boundary_coefs[i-N+bl]
    end
    return L / A.dx^A.derivative_order
end
