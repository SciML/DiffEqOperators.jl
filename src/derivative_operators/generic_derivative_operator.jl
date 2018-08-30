struct GenericDerivativeOperator{T,LT<:AbstractDiffEqLinearOperator{T}} <: AbstractDiffEqLinearOperator{T}
    L::LT
    dimension::Int
    leftBC
    rightBC
    left_cache::Vector{T}
    right_cache::Vector{T}
    function GenericDerivativeOperator(L, leftBC, rightBC)
        T = eltype(L)
        dim = size(L,1)
        boundary_pts = div(size(L,2) - size(L,1), 2) # assume equal number of boundary points at each end
        left_cache = Vector{T}(undef, boundary_pts)
        right_cache = Vector{T}(undef, boundary_pts)
        new{T,typeof(L)}(L,dim,leftBC,rightBC,left_cache,right_cache)
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
    mul!(y, LB.L, x)
end
function convert(::Type{AbstractMatrix}, LB::GenericDerivativeOperator)
    Lmat = convert(AbstractMatrix, LB.L)
    Qmat = constructQ(LB)
    return Lmat * Qmat
end

#########################################
# Low level routines
function applyQ!(LB::GenericDerivativeOperator, x::AbstractVector)
    # TODO
end
function constructQ(LB::GenericDerivativeOperator)
    # TODO
end
