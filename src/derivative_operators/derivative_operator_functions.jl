function LinearAlgebra.mul!(x_temp::AbstractArray{T}, A::DerivativeOperator{T,N}, M::AbstractArray{T}) where {T<:Real,N}

    # Check that x_temp has correct dimensions
    v = zeros(ndims(x_temp))
    v[N] = 2
    @assert [size(x_temp)...]+v == [size(M)...]

    # Check that axis of differentiation is in the dimensions of M and x_temp
    ndimsM = ndims(M)
    @assert N <= ndimsM

    dimsM = [axes(M)...]
    alldims = [1:ndims(M);]
    otherdims = setdiff(alldims, N)

    idx = Any[first(ind) for ind in axes(M)]
    itershape = tuple(dimsM[otherdims]...)
    nidx = length(otherdims)
    indices = Iterators.drop(CartesianIndices(itershape), 0)

    setindex!(idx, :, N)
    for I in indices
        Base.replace_tuples!(nidx, idx, idx, otherdims, I)
        mul!(view(x_temp, idx...), A, view(M, idx...))
    end
end

for MT in [2,3]
    @eval begin
        function LinearAlgebra.mul!(x_temp::AbstractArray{T,$MT}, A::DerivativeOperator{T,N,false,T2,S1}, M::AbstractArray{T,$MT}) where {T<:Real,N,T2,SL,S1<:SArray{Tuple{SL},T,1,SL}}

            # Check that x_temp has correct dimensions
            v = zeros(ndims(x_temp))
            v[N] = 2
            @assert [size(x_temp)...]+v == [size(M)...]

            # Check that axis of differentiation is in the dimensions of M and x_temp
            ndimsM = ndims(M)
            @assert N <= ndimsM

            # Determine padding for NNlib.conv!
            bpc = A.boundary_point_count
            pad = zeros(Int64,ndimsM)
            pad[N] = bpc

            # Reshape x_temp for NNlib.conv!
            _x_temp = reshape(x_temp, (size(x_temp)...,1,1))

            # Reshape M for NNlib.conv!
            _M = reshape(M, (size(M)...,1,1))

            # Setup W, the kernel for NNlib.conv!
            s = A.stencil_coefs
            sl = A.stencil_length
            Wdims = ones(Int64, ndims(_x_temp))
            Wdims[N] = sl
            W = zeros(Wdims...)
            Widx = Any[Wdims...]
            setindex!(Widx,:,N)
            W[Widx...] = s

            cv = DenseConvDims(_M, W, padding=pad)
            conv!(_x_temp, _M, W, cv)

            # Now deal with boundaries
            if bpc > 0
                dimsM = [axes(M)...]
                alldims = [1:ndims(M);]
                otherdims = setdiff(alldims, N)

                idx = Any[first(ind) for ind in axes(M)]
                itershape = tuple(dimsM[otherdims]...)
                nidx = length(otherdims)
                indices = Iterators.drop(CartesianIndices(itershape), 0)

                setindex!(idx, :, N)
                for I in indices
                    Base.replace_tuples!(nidx, idx, idx, otherdims, I)
                    convolve_BC_left!(view(x_temp, idx...), view(M, idx...), A)
                    convolve_BC_right!(view(x_temp, idx...), view(M, idx...), A)
                end
            end
            mul!(x_temp,x_temp,1/A.dx^A.derivative_order)
        end
    end
end

function *(A::DerivativeOperator{T,N},M::AbstractArray{T}) where {T<:Real,N}
    size_x_temp = [size(M)...]
    size_x_temp[N] -= 2
    x_temp = zeros(promote_type(eltype(A),eltype(M)), size_x_temp...)
    LinearAlgebra.mul!(x_temp, A, M)
    return x_temp
end

# Will eventually implement proper dispatch
function LinearAlgebra.mul!(x_temp::AbstractArray{T,2}, A::AbstractDiffEqCompositeOperator, M::AbstractArray{T,2}) where {T}

    ndimsM = ndims(M)
    Wdims = ones(Int64,ndimsM)
    pad = zeros(Int64, ndimsM)

    # compute dimensions of interior kernel W
    for L in A.ops
        axis = typeof(L).parameters[2]
        @assert axis <= ndimsM
        Wdims[axis] = max(Wdims[axis],L.stencil_length)
        pad[axis] = max(pad[axis], L.boundary_point_count)
    end

    # create zero-valued kernel
    W = zeros(T, Wdims...)
    mid_Wdims = div.(Wdims,2).+1
    idx = div.(Wdims,2).+1

    # add to kernel each stencil
    for L in A.ops
        s = L.stencil_coefs
        sl = L.stencil_length
        axis = typeof(L).parameters[2]
        offset = convert(Int64,(Wdims[axis] - sl)/2)
        for i in offset+1:Wdims[axis]-offset
            idx[axis]=i

            W[idx...] += (s[i-offset] / L.dx^L.derivative_order)

            idx[axis] = mid_Wdims[axis]
        end
    end

    # Reshape x_temp for NNlib.conv!
    _x_temp = reshape(x_temp, (size(x_temp)...,1,1))

    # Reshape M for NNlib.conv!
    _M = reshape(M, (size(M)...,1,1))

    _W = reshape(W, (size(W)...,1,1))

    # Call NNlib.conv!
    cv = DenseConvDims(_M, _W, padding=pad)
    conv!(_x_temp, _M, _W, cv)


    # convolve boundary and interior points near boundary
    # partition operator indices along axis of differentiation
    if pad[1] > 0 || pad[2] > 0
        ops_1 = Int64[]
        ops_1_max_bpc_idx = [0]
        ops_2 = Int64[]
        ops_2_max_bpc_idx = [0]
        for i in 1:length(A.ops)
            L = A.ops[i]
            if typeof(L).parameters[2] == 1
                push!(ops_1,i)
                if L.boundary_point_count == pad[1]
                    ops_1_max_bpc_idx[1] = i
                end
            else
                push!(ops_2,i)
                if L.boundary_point_count == pad[2]
                    ops_2_max_bpc_idx[1]= i
                end
            end
        end

        # need offsets since some axis may have ghost nodes and some may not
        offset_x = 0
        offset_y = 0

        if length(ops_2) > 0
            offset_x = 1
        end
        if length(ops_1) > 0
            offset_y =1
        end

    # convolve boundaries and unaccounted for interior in axis 1
        if length(ops_1) > 0
            for i in 1:size(x_temp)[2]
                convolve_BC_left!(view(x_temp,:,i), view(M,:,i+offset_x), A.ops[ops_1_max_bpc_idx...])
                convolve_BC_right!(view(x_temp,:,i), view(M,:,i+offset_x), A.ops[ops_1_max_bpc_idx...])
                if i <= pad[2] || i > size(x_temp)[2]-pad[2]
                    convolve_interior!(view(x_temp,:,i), view(M,:,i+offset_x), A.ops[ops_1_max_bpc_idx...])
                end
                #scale by dx

                for Lidx in ops_1
                    if Lidx != ops_1_max_bpc_idx[1]
                        convolve_BC_left_add!(view(x_temp,:,i), view(M,:,i+offset_x), A.ops[Lidx])
                        convolve_BC_right_add!(view(x_temp,:,i), view(M,:,i+offset_x), A.ops[Lidx])
                        if i <= pad[2] || i > size(x_temp)[2]-pad[2]
                            convolve_interior_add!(view(x_temp,:,i), view(M,:,i+offset_x), A.ops[Lidx])
                        elseif pad[1] - A.ops[Lidx].boundary_point_count > 0
                            #convolve_BC_interior_add_range!
                        end
                    end
                end
            end
        end
        # convolve boundaries and unaccounted for interior in axis 2
        if length(ops_2) > 0
            for i in 1:size(x_temp)[1]
                # in the case of no axis 1 operators, we need to over x_temp
                if length(ops_1) == 0
                    convolve_BC_left!(view(x_temp,i,:), view(M,i+offset_y,:), A.ops[ops_2_max_bpc_idx...])
                    convolve_BC_right!(view(x_temp,i,:), view(M,i+offset_y,:), A.ops[ops_2_max_bpc_idx...])
                    if i <= pad[1] || i > size(x_temp)[1]-pad[1]
                        convolve_interior!(view(x_temp,i,:), view(M,i+offset_y,:), A.ops[ops_2_max_bpc_idx...])
                    end
                    #scale by dx
                    # fix here as well
                else
                    convolve_BC_left_add!(view(x_temp,i,:), view(M,i+offset_y,:), A.ops[ops_2_max_bpc_idx...])
                    convolve_BC_right_add!(view(x_temp,i,:), view(M,i+offset_y,:), A.ops[ops_2_max_bpc_idx...])
                    if i <= pad[1] || i > size(x_temp)[1]-pad[1]
                        convolve_interior_add!(view(x_temp,i,:), view(M,i+offset_y,:), A.ops[ops_2_max_bpc_idx...])
                    end
                    #scale by dx
                    # fix here as well
                end
                for Lidx in ops_2
                    if Lidx != ops_2_max_bpc_idx[1]
                        convolve_BC_left_add!(view(x_temp,i,:), view(M,i+offset_y,:), A.ops[Lidx])
                        convolve_BC_right_add!(view(x_temp,i,:), view(M,i+offset_y,:), A.ops[Lidx])
                        if i <= pad[1] || i > size(x_temp)[1]-pad[1]
                            convolve_interior_add!(view(x_temp,i,:), view(M,i+offset_y,:), A.ops[Lidx])
                        elseif pad[2] - A.ops[Lidx].boundary_point_count > 0
                            #convolve_BC_interior_add_range!...
                        end
                    end
                end
            end
        end
    end
end


# Implementations of additive convolutions, need to remove redundancy later since we are
# only making these calls when grids are regular, non-winding, and coefficients among indices are equivalent
function convolve_interior_add!(x_temp::AbstractVector{T}, x::AbstractVector{T}, A::DerivativeOperator) where {T<:Real}
    @assert length(x_temp)+2 == length(x)
    stencil = A.stencil_coefs
    coeff   = A.coefficients
    mid = div(A.stencil_length,2)
    for i in (1+A.boundary_point_count) : (length(x_temp)-A.boundary_point_count)
        xtempi = zero(T)
        cur_stencil = eltype(stencil) <: AbstractVector ? stencil[i] : stencil
        cur_coeff   = typeof(coeff)   <: AbstractVector ? coeff[i] : true
        cur_stencil = use_winding(A) && cur_coeff < 0 ? reverse(cur_stencil) : cur_stencil
        for idx in 1:A.stencil_length
            xtempi += cur_coeff * cur_stencil[idx] * x[i - mid + idx]
        end
        x_temp[i] += xtempi
    end
end

function convolve_BC_left_add!(x_temp::AbstractVector{T}, x::AbstractVector{T}, A::DerivativeOperator) where {T<:Real}
    stencil = A.low_boundary_coefs
    coeff   = A.coefficients
    for i in 1 : A.boundary_point_count
        xtempi = stencil[i][1]*x[1]
        cur_stencil = stencil[i]
        cur_coeff   = typeof(coeff)   <: AbstractVector ? coeff[i] : true
        cur_stencil = use_winding(A) && cur_coeff < 0 ? reverse(cur_stencil) : cur_stencil
        for idx in 2:A.boundary_stencil_length
            xtempi += cur_coeff * cur_stencil[idx] * x[idx]
        end
        x_temp[i] += xtempi
    end
end

function convolve_BC_right_add!(x_temp::AbstractVector{T}, x::AbstractVector{T}, A::DerivativeOperator) where {T<:Real}
    stencil = A.high_boundary_coefs
    coeff   = A.coefficients
    for i in 1 : A.boundary_point_count
        xtempi = stencil[i][end]*x[end]
        cur_stencil = stencil[i]
        cur_coeff   = typeof(coeff)   <: AbstractVector ? coeff[i] : true
        cur_stencil = use_winding(A) && cur_coeff < 0 ? reverse(cur_stencil) : cur_stencil
        for idx in (A.boundary_stencil_length-1):-1:1
            xtempi += cur_coeff * cur_stencil[end-idx] * x[end-idx]
        end
        x_temp[end-A.boundary_point_count+i] += xtempi
    end
end
