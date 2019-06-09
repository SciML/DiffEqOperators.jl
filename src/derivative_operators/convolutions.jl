#= Worker functions=#
low(i::Int, mid::Int, bpc::Int) = Int(mid + (i-1)*(1-mid)/bpc)
high(i::Int, mid::Int, bpc::Int, slen::Int, L::Int) = Int(slen - (slen-mid)*(i-L+bpc)/(bpc))

# used in general dirichlet BC. To simulate a constant value beyond the boundary
limit(i, N) = N>=i>=1 ? i : (i<1 ? 1 : N)

# used in Neumann 0 BC
function reflect(idx, L)
    abs1 = abs(L-idx)
    if L - abs1 > 0
        return L-abs1
    else
        return abs(L-abs1)+2
    end
end

# gives the index for periodic BC
function rem1(idx,L)
    r = idx%L
    if r > 0
        return r
    else
        return r+L
    end
end

function LinearAlgebra.mul!(x_temp::AbstractVector{T}, A::Union{DerivativeOperator{T},UpwindOperator{T}}, x::AbstractVector{T}) where T<:Real
    convolve_BC_left!(x_temp, x, A)
    convolve_interior!(x_temp, x, A)
    convolve_BC_right!(x_temp, x, A)
    rmul!(x_temp, @.(1/(A.dx^A.derivative_order)))
end

# Against a standard vector, assume already padded and just apply the stencil
function convolve_interior!(x_temp::AbstractVector{T}, x::AbstractVector{T}, A::DerivativeOperator{T,S}) where {T<:Real,S<:SVector}
    @assert length(x_temp) == length(x)
    coeffs = A.stencil_coefs
    mid = div(A.stencil_length,2) + 1
    Threads.@threads for i in A.boundary_length : length(x_temp)-A.boundary_length
        xtempi = zero(T)
        @inbounds for idx in 1:A.stencil_length
            xtempi += coeffs[idx] * x[i - (mid-idx)]
        end
        x_temp[i] = xtempi
    end
end

function convolve_BC_left!(x_temp::AbstractVector{T}, x::AbstractVector, A::DerivativeOperator{T,S}) where {T<:Real,S<:SVector}
    coeffs = A.low_boundary_coefs
    Threads.@threads for i in 1 : A.boundary_length
        xtempi = coeffs[i][1]*x[1]
        @inbounds for idx in 2:A.stencil_length
            xtempi += coeffs[i][idx] * x[idx]
        end
        x_temp[i] = xtempi
    end
end

function convolve_BC_right!(x_temp::AbstractVector{T}, x::AbstractVector, A::DerivativeOperator{T,S}) where {T<:Real,S<:SVector}
    coeffs = A.low_boundary_coefs
    bc_start = length(x) - A.stencil_length
    Threads.@threads for i in 1 : A.boundary_length
        xtempi = coeffs[i][end]*x[end]
        @inbounds for idx in A.stencil_length:-1:2
            xtempi += coeffs[i][end-idx] * x[end-idx]
        end
        x_temp[i] = xtempi
    end
end

###########################################

# Against A BC-padded vector, specialize the computation to explicitly use the left, right, and middle parts
function convolve_interior!(x_temp::AbstractVector{T}, _x::RobinBCExtended, A::DerivativeOperator{T,S}) where {T<:Real,S<:SVector}
    coeffs = A.stencil_coefs
    x = _x.u
    mid = div(A.stencil_length,2) + 1
    # Just do the middle parts
    Threads.@threads for i in A.boundary_length : length(x_temp)-A.boundary_length
        xtempi = zero(T)
        @inbounds for idx in 1:A.stencil_length
            xtempi += coeffs[idx] * x[i - (mid-idx) + 1]
        end
        x_temp[i] = xtempi
    end
end

function convolve_BC_left!(x_temp::AbstractVector{T}, _x::RobinBCExtended, A::DerivativeOperator{T,S}) where {T<:Real,S<:SVector}
    coeffs = A.low_boundary_coefs
    Threads.@threads for i in 1 : A.boundary_length
        xtempi = coeffs[i][1]*x.l
        @inbounds for idx in 2:A.stencil_length
            xtempi += coeffs[i][idx] * x.u[idx-1]
        end
        x_temp[i] = xtempi
    end
end

function convolve_BC_right!(x_temp::AbstractVector{T}, _x::RobinBCExtended, A::DerivativeOperator{T,S}) where {T<:Real,S<:SVector}
    coeffs = A.low_boundary_coefs
    bc_start = length(x.u) - A.stencil_length
    Threads.@threads for i in 1 : A.boundary_length
        xtempi = coeffs[i][end]*x.r
        @inbounds for idx in A.stencil_length:-1:2
            xtempi += coeffs[i][end-idx] * x.u[end-idx+1]
        end
        x_temp[i] = xtempi
    end
end

##########################################

#= INTERIOR CONVOLUTION =#
function convolve_interior!(x_temp::AbstractVector{T}, x::AbstractVector{T}, A::UpwindOperator{T,S,LBC,RBC}) where {T<:Real,S<:SVector,LBC,RBC}
    N = length(x)
    stencil_length = length(A.up_stencil_coefs)
    stencil_rem = 1 - stencil_length%2
    Threads.@threads for i in A.boundary_point_count[1]+1 : N-A.boundary_point_count[2]
        xtempi = zero(T)
        if A.directions[][i] == false
            @inbounds for j in 1 : length(A.up_stencil_coefs)
                xtempi += A.up_stencil_coefs[j] * x[i+j-1-stencil_rem]
            end
        else
            @inbounds for j in -length(A.down_stencil_coefs)+1 : 0
                xtempi += A.down_stencil_coefs[j+stencil_length] * x[i+j+stencil_rem]
                # println("i = $i, j = $j, s_idx = $(j+stencil_length), x_idx = $(i+j+stencil_rem), $(A.down_stencil_coefs[j+stencil_length]) * $(x[i+j+stencil_rem]), xtempi = $xtempi")
            end
        end

        x_temp[i] = xtempi
    end
end
