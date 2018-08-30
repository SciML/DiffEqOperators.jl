struct GenericDerivativeOperator{T,LT<:AbstractDiffEqLinearOperator{T}} <: AbstractDiffEqLinearOperator{T}
    L::LT
    dimension::Int
    boundary_pts::Int
    leftBC::Symbol
    rightBC::Symbol
    left_cache::Vector{T}
    right_cache::Vector{T}
    function GenericDerivativeOperator(L, leftBC, rightBC)
        T = eltype(L)
        dim = size(L,1)
        boundary_pts = div(size(L,2) - size(L,1), 2) # assume equal number of boundary points at each end
        left_cache = Vector{T}(undef, boundary_pts)
        right_cache = Vector{T}(undef, boundary_pts)
        new{T,typeof(L)}(L,dim,boundary_pts,leftBC,rightBC,left_cache,right_cache)
    end
    GenericDerivativeOperator(L, BC) = GenericDerivativeOperator(L, BC[1], BC[2])
end

#########################################
# High level routines
size(LB::GenericDerivativeOperator) = (LB.dimension, LB.dimension)
size(LB::GenericDerivativeOperator, i::Int) = i <= 2 ? LB.dimension : 1
function mul!(y::AbstractVector, LB::GenericDerivativeOperator, x::AbstractVector)
    applyQ!(LB, x)
    xbar = Vcat(LB.left_cache, x, LB.right_cache) # lazy concat
    mul!(y, LB.L, xbar)
end
function *(LB::GenericDerivativeOperator, x::AbstractVector)
    y = zeros(promote_type(eltype(LB), eltype(x)), size(LB, 1))
    mul!(y, LB, x)
end
function convert(::Type{AbstractMatrix}, LB::GenericDerivativeOperator)
    Lmat = convert(AbstractMatrix, LB.L)
    Qmat = constructQ(LB)
    return Lmat * Qmat
end

#########################################
# Low level routines
function applyQ!(LB::GenericDerivativeOperator{T,LT}, x::AbstractVector{T}) where {T,LT}
    # Left boundary
    if LB.leftBC == :Dirichlet0
        @inbounds for i = 1:LB.boundary_pts
            LB.left_cache[i] = zero(T)
        end
    elseif LB.leftBC == :Neumann0
        @inbounds for i = 1:LB.boundary_pts
            LB.left_cache[end - i + 1] = x[i]
        end
    end
    # Right boundary
    if LB.rightBC == :Dirichlet0
        @inbounds for i = 1:LB.boundary_pts
            LB.right_cache[i] = zero(T)
        end
    elseif LB.rightBC == :Neumann0
        @inbounds for i = 1:LB.boundary_pts
            LB.right_cache[i] = x[end - i + 1]
        end
    end
    return LB
end
function constructQ(LB::GenericDerivativeOperator{T,LT}) where {T,LT}
    mat = spzeros(T, LB.dimension + 2*LB.boundary_pts, LB.dimension)
    # Fill out interior
    for i = 1:LB.dimension
        mat[LB.boundary_pts + i, i] = one(T)
    end
    # Left boundary
    if LB.leftBC == :Dirichlet0
        # No change
    elseif LB.leftBC == :Neumann0
        for i = 1:LB.boundary_pts
            mat[LB.boundary_pts - i + 1, i] = one(T)
        end
    end
    # Right boundary
    if LB.rightBC == :Dirichlet0
        # No change
    elseif LB.rightBC == :Neumann0
        for i = 1:LB.boundary_pts
            mat[LB.boundary_pts + LB.dimension + i, LB.dimension - i + 1] = one(T)
        end
    end
    return mat
end
