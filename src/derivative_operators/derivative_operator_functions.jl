#
# The mul! methods that contract derivative operators with arrays.
#
# These methods handle the {N} dimension type parameter.
#
# There are four methods.  One is a general method for multidimensional
# arrays, and the others are short cuts for 1, 2 and 3 dimensions.  The 1 dimensional
# method for AbstractVector is defined in convolutions.jl, and the
# others are defined here.
#
# At the interior points of an evenly spaced grid, a finite difference
# operator is a convolution.  Where possible, the heavy lifting is
# done by the efficient conv!  routine from NNlib.  The routines
# defined in convolutions.jl cover the cases where that isn't possible.
#

# Fallback mul! implementation for a single DerivativeOperator operating on an AbstractArray
function LinearAlgebra.mul!(x_temp::AbstractArray{T}, A::DerivativeOperator{T,N}, M::AbstractArray{T}; overwrite = true) where {T,N}

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
        mul!(view(x_temp, idx...), A, view(M, idx...), overwrite = overwrite)
    end
end

# A more efficient mul! implementation for a single, regular-grid, centered difference,
# scalar coefficient, non-winding, DerivativeOperator operating on a 2D or 3D AbstractArray
for MT in [2,3]
    @eval begin
        function LinearAlgebra.mul!(x_temp::AbstractArray{T,$MT}, A::DerivativeOperator{T,N,false,T2,S1,S2,T3}, M::AbstractArray{T,$MT}) where
                                                                            {T,N,T2,SL,S1<:SArray{Tuple{SL},T,1,SL},S2,T3<:Union{Nothing,Number}}
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
            coeff = A.coefficients === nothing ? true : A.coefficients
            W[Widx...] = coeff*s

            cv = DenseConvDims(_M, W, padding=pad, flipkernel=true)
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
        end
    end
end

###########################################

function *(A::DerivativeOperator{T,N},M::AbstractArray{T}) where {T<:Real,N}
    size_x_temp = [size(M)...]
    size_x_temp[N] -= 2
    x_temp = zeros(promote_type(eltype(A),eltype(M)), size_x_temp...)
    LinearAlgebra.mul!(x_temp, A, M)
    return x_temp
end

function *(c::Number, A::DerivativeOperator{T,N,Wind}) where {T,N,Wind}
    coefficients = A.coefficients === nothing ? oneunit(T) .* c : c .* A.coefficients
    DerivativeOperator{T,N,Wind,typeof(A.dx),typeof(A.stencil_coefs),
                       typeof(A.low_boundary_coefs),typeof(coefficients),
                       typeof(A.coeff_func)}(
        A.derivative_order, A.approximation_order,
        A.dx, A.len, A.stencil_length,
        A.stencil_coefs,
        A.boundary_stencil_length,
        A.boundary_point_count,
        A.low_boundary_coefs,
        A.high_boundary_coefs,coefficients,A.coeff_func)
end

function *(c::AbstractVector{<:Number}, A::DerivativeOperator{T,N,Wind}) where {T,N,Wind}
    if length(c) != A.len
        throw(DimensionMismatch("length of c ($(length(c))) must match length of A ($A.len)"))
    end
    coefficients = A.coefficients === nothing ? c : c .* A.coefficients
    DerivativeOperator{T,N,Wind,typeof(A.dx),typeof(A.stencil_coefs),
                       typeof(A.low_boundary_coefs),typeof(coefficients),
                       typeof(A.coeff_func)}(
        A.derivative_order, A.approximation_order,
        A.dx, A.len, A.stencil_length,
        A.stencil_coefs,
        A.boundary_stencil_length,
        A.boundary_point_count,
        A.low_boundary_coefs,
        A.high_boundary_coefs,coefficients,A.coeff_func)
end

###########################################

# A more efficient mul! implementation for compositions of operators which may include regular-grid, centered difference,
# scalar coefficient, non-winding, DerivativeOperator, operating on a 2D or 3D AbstractArray
function LinearAlgebra.mul!(x_temp::AbstractArray{T,2}, A::AbstractDiffEqCompositeOperator, M::AbstractArray{T,2}) where {T}

    # opsA operators satisfy conditions for NNlib.conv! call, opsB operators do not
    opsA = DerivativeOperator[]
    opsB = DerivativeOperator[]
    for L in A.ops
        if (L.coefficients isa Number || L.coefficients === nothing) && use_winding(L) == false && L.dx isa Number
            push!(opsA, L)
        else
            push!(opsB,L)
        end
    end

    # Check that we can make at least one NNlib.conv! call
    if !isempty(opsA)
        ndimsM = ndims(M)
        Wdims = ones(Int64,ndimsM)
        pad = zeros(Int64, ndimsM)

        # compute dimensions of interior kernel W
        # Here we still use A.ops since operators in opsB may indicate that
        # we have more padding to account for
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
        for L in opsA
            s = L.stencil_coefs
            sl = L.stencil_length
            axis = typeof(L).parameters[2]
            offset = convert(Int64,(Wdims[axis] - sl)/2)
            coeff = L.coefficients isa Number ? L.coefficients : true
            for i in offset+1:Wdims[axis]-offset
                idx[axis]=i
                W[idx...] += coeff*s[i-offset]
                idx[axis] = mid_Wdims[axis]
            end
        end

        # Reshape x_temp for NNlib.conv!
        _x_temp = reshape(x_temp, (size(x_temp)...,1,1))

        # Reshape M for NNlib.conv!
        _M = reshape(M, (size(M)...,1,1))

        _W = reshape(W, (size(W)...,1,1))

        # Call NNlib.conv!
        cv = DenseConvDims(_M, _W, padding=pad, flipkernel=true)
        conv!(_x_temp, _M, _W, cv)


        # convolve boundary and interior points near boundary
        # partition operator indices along axis of differentiation
        if pad[1] > 0 || pad[2] > 0
            ops_1 = Int64[]
            ops_1_max_bpc_idx = [0]
            ops_2 = Int64[]
            ops_2_max_bpc_idx = [0]
            for i in 1:length(opsA)
                L = opsA[i]
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
                offset_y = 1
            end

            # convolve boundaries and unaccounted for interior in axis 1
            if length(ops_1) > 0
                for i in 1:size(x_temp)[2]
                    convolve_BC_left!(view(x_temp,:,i), view(M,:,i+offset_x), opsA[ops_1_max_bpc_idx...])
                    convolve_BC_right!(view(x_temp,:,i), view(M,:,i+offset_x), opsA[ops_1_max_bpc_idx...])
                    if i <= pad[2] || i > size(x_temp)[2]-pad[2]
                        convolve_interior!(view(x_temp,:,i), view(M,:,i+offset_x), opsA[ops_1_max_bpc_idx...])
                    end

                    for Lidx in ops_1
                        if Lidx != ops_1_max_bpc_idx[1]
                            convolve_BC_left!(view(x_temp,:,i), view(M,:,i+offset_x), opsA[Lidx], overwrite = false)
                            convolve_BC_right!(view(x_temp,:,i), view(M,:,i+offset_x), opsA[Lidx], overwrite = false)
                            if i <= pad[2] || i > size(x_temp)[2]-pad[2]
                                convolve_interior!(view(x_temp,:,i), view(M,:,i+offset_x), opsA[Lidx], overwrite = false)
                            elseif pad[1] - opsA[Lidx].boundary_point_count > 0
                                convolve_interior!(view(x_temp,:,i), view(M,:,i+offset_x), opsA[Lidx], overwrite = false, add_range = true, offset = pad[1] - opsA[Lidx].boundary_point_count)
                            end
                        end
                    end
                end
            end
            # convolve boundaries and unaccounted for interior in axis 2
            if length(ops_2) > 0
                for i in 1:size(x_temp)[1]
                    # in the case of no axis 1 operators, we need to overwrite x_temp
                    if length(ops_1) == 0
                        convolve_BC_left!(view(x_temp,i,:), view(M,i+offset_y,:), opsA[ops_2_max_bpc_idx...])
                        convolve_BC_right!(view(x_temp,i,:), view(M,i+offset_y,:), opsA[ops_2_max_bpc_idx...])
                        if i <= pad[1] || i > size(x_temp)[1]-pad[1]
                            convolve_interior!(view(x_temp,i,:), view(M,i+offset_y,:), opsA[ops_2_max_bpc_idx...])
                        end

                    else
                        convolve_BC_left!(view(x_temp,i,:), view(M,i+offset_y,:), opsA[ops_2_max_bpc_idx...], overwrite = false)
                        convolve_BC_right!(view(x_temp,i,:), view(M,i+offset_y,:), opsA[ops_2_max_bpc_idx...], overwrite = false)
                        if i <= pad[1] || i > size(x_temp)[1]-pad[1]
                            convolve_interior!(view(x_temp,i,:), view(M,i+offset_y,:), opsA[ops_2_max_bpc_idx...], overwrite = false)
                        end

                    end
                    for Lidx in ops_2
                        if Lidx != ops_2_max_bpc_idx[1]
                            convolve_BC_left!(view(x_temp,i,:), view(M,i+offset_y,:), opsA[Lidx], overwrite = false)
                            convolve_BC_right!(view(x_temp,i,:), view(M,i+offset_y,:), opsA[Lidx], overwrite = false)
                            if i <= pad[1] || i > size(x_temp)[1]-pad[1]
                                convolve_interior!(view(x_temp,i,:), view(M,i+offset_y,:), opsA[Lidx], overwrite = false)
                            elseif pad[2] - opsA[Lidx].boundary_point_count > 0
                                convolve_interior!(view(x_temp,i,:), view(M,i+offset_y,:), opsA[Lidx], overwrite = false, add_range = true, offset = pad[2] - opsA[Lidx].boundary_point_count)
                            end
                        end
                    end
                end
            end
        end

        # Here we compute mul! (additively) for every operator in opsB

        operating_dims = zeros(Int64,2)
        # need to consider all dimensions and operators to determine the truncation
        # of M to x_temp
        for L in A.ops
            if diff_axis(L) == 1
                operating_dims[1] = 1
            else
                operating_dims[2] = 1
            end
        end

        x_temp_1, x_temp_2 = size(x_temp)

        for L in opsB
            N = diff_axis(L)
            if N == 1
                if operating_dims[2] == 1
                    mul!(x_temp,L,view(M,1:x_temp_1+2,1:x_temp_2), overwrite = false)
                else
                    mul!(x_temp,L,M, overwrite = false)
                end
            else
                if operating_dims[1] == 1
                    mul!(x_temp,L,view(M,1:x_temp_1,1:x_temp_2+2), overwrite = false)
                else
                    mul!(x_temp,L,M, overwrite = false)
                end
            end
        end

    # The case where we call everything in A.ops using the fallback mul!
    else
        #operating_dims
        operating_dims = zeros(Int64,2)
        for L in A.ops
            if diff_axis(L) == 1
                operating_dims[1] = 1
            else
                operating_dims[2] = 1
            end
        end

        x_temp_1, x_temp_2 = size(x_temp)

        # Handle first case non-additively
        N = diff_axis(A.ops[1])
        if N == 1
            if operating_dims[2] == 1
                mul!(x_temp,A.ops[1],view(M,1:x_temp_1+2,1:x_temp_2))
            else
                mul!(x_temp,A.ops[1],M)
            end
        else
            if operating_dims[1] == 1
                mul!(x_temp,A.ops[1],view(M,1:x_temp_1,1:x_temp_2+2))
            else
                mul!(x_temp,A.ops[1],M)
            end
        end

        for L in A.ops[2:end]
            N = diff_axis(L)
            if N == 1
                if operating_dims[2] == 1
                    mul!(x_temp,L,view(M,1:x_temp_1+2,1:x_temp_2), overwrite = false)
                else
                    mul!(x_temp,L,M, overwrite = false)
                end
            else
                if operating_dims[1] == 1
                    mul!(x_temp,L,view(M,1:x_temp_1,1:x_temp_2+2), overwrite = false)
                else
                    mul!(x_temp,L,M, overwrite = false)
                end
            end
        end
    end
end

# A more efficient mul! implementation for compositions of operators which may include regular-grid, centered difference,
# scalar coefficient, non-winding, DerivativeOperator, operating on a 2D or 3D AbstractArray
function LinearAlgebra.mul!(x_temp::AbstractArray{T,3}, A::AbstractDiffEqCompositeOperator, M::AbstractArray{T,3}) where {T}

    # opsA operators satisfy conditions for NNlib.conv! call, opsB operators do not
    opsA = DerivativeOperator[]
    opsB = DerivativeOperator[]
    for L in A.ops
        if (L.coefficients isa Number || L.coefficients === nothing) && use_winding(L) == false && L.dx isa Number
            push!(opsA, L)
        else
            push!(opsB,L)
        end
    end

    # Check that we can make at least one NNlib.conv! call
    if !isempty(opsA)
        ndimsM = ndims(M)
        Wdims = ones(Int64,ndimsM)
        pad = zeros(Int64, ndimsM)

        # compute dimensions of interior kernel W
        # Here we still use A.ops since operators in opsB may indicate that
        # we have more padding to account for
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
        for L in opsA
            s = L.stencil_coefs
            sl = L.stencil_length
            axis = typeof(L).parameters[2]
            offset = convert(Int64,(Wdims[axis] - sl)/2)
            coeff = L.coefficients isa Number ? L.coefficients : true
            for i in offset+1:Wdims[axis]-offset
                idx[axis]=i
                W[idx...] += coeff*s[i-offset]
                idx[axis] = mid_Wdims[axis]
            end
        end

        # Reshape x_temp for NNlib.conv!
        _x_temp = reshape(x_temp, (size(x_temp)...,1,1))

        # Reshape M for NNlib.conv!
        _M = reshape(M, (size(M)...,1,1))

        _W = reshape(W, (size(W)...,1,1))

        # Call NNlib.conv!
        cv = DenseConvDims(_M, _W, padding=pad, flipkernel=true)
        conv!(_x_temp, _M, _W, cv)


        # convolve boundary and interior points near boundary
        # partition operator indices along axis of differentiation
        if pad[1] > 0 || pad[2] > 0 || pad[3] > 0
            ops_1 = Int64[]
            ops_1_max_bpc_idx = [0]
            ops_2 = Int64[]
            ops_2_max_bpc_idx = [0]
            ops_3 = Int64[]
            ops_3_max_bpc_idx = [0]

            for i in 1:length(opsA)
                L = opsA[i]
                if typeof(L).parameters[2] == 1
                    push!(ops_1,i)
                    if L.boundary_point_count == pad[1]
                        ops_1_max_bpc_idx[1] = i
                    end
                elseif typeof(L).parameters[2] == 2
                    push!(ops_2,i)
                    if L.boundary_point_count == pad[2]
                        ops_2_max_bpc_idx[1]= i
                    end
                else
                    push!(ops_3,i)
                    if L.boundary_point_count == pad[3]
                        ops_3_max_bpc_idx[1]= i
                    end
                end
            end

            # need offsets since some axis may have ghost nodes and some may not
            offset_x = 0
            offset_y = 0
            offset_z = 0

            if length(ops_1) > 0
                offset_x = 1
            end
            if length(ops_2) > 0
                offset_y = 1
            end
            if length(ops_3) > 0
                offset_z = 1
            end

            # convolve boundaries and unaccounted for interior in axis 1
            if length(ops_1) > 0
                for i in 1:size(x_temp)[2]
                    for j in 1:size(x_temp)[3]
                        convolve_BC_left!(view(x_temp,:,i,j), view(M,:,i+offset_y,j+offset_z), opsA[ops_1_max_bpc_idx...])
                        convolve_BC_right!(view(x_temp,:,i,j), view(M,:,i+offset_y,j+offset_z), opsA[ops_1_max_bpc_idx...])
                        if i <= pad[2] || i > size(x_temp)[2]-pad[2] || j <= pad[3] || j > size(x_temp)[3]-pad[3]
                            convolve_interior!(view(x_temp,:,i,j), view(M,:,i+offset_y,j+offset_z), opsA[ops_1_max_bpc_idx...])
                        end

                        for Lidx in ops_1
                            if Lidx != ops_1_max_bpc_idx[1]
                                convolve_BC_left!(view(x_temp,:,i,j), view(M,:,i+offset_y,j+offset_z), opsA[Lidx], overwrite = false)
                                convolve_BC_right!(view(x_temp,:,i,j), view(M,:,i+offset_y,j+offset_z), opsA[Lidx], overwrite = false)
                                if i <= pad[2] || i > size(x_temp)[2]-pad[2] || j <= pad[3] || j > size(x_temp)[3]-pad[3]
                                    convolve_interior!(view(x_temp,:,i,j), view(M,:,i+offset_y,j+offset_z), opsA[Lidx], overwrite = false)
                                elseif pad[1] - opsA[Lidx].boundary_point_count > 0
                                    convolve_interior!(view(x_temp,:,i,j), view(M,:,i+offset_y,j+offset_z), opsA[Lidx], overwrite = false, add_range = true, offset = pad[1] - opsA[Lidx].boundary_point_count)
                                end
                            end
                        end
                    end
                end
            end
            # convolve boundaries and unaccounted for interior in axis 2
            if length(ops_2) > 0
                for i in 1:size(x_temp)[1]
                    for j in 1:size(x_temp)[3]
                        # in the case of no axis 1 operators, we need to overwrite x_temp
                        if length(ops_1) == 0
                            convolve_BC_left!(view(x_temp,i,:,j), view(M,i+offset_x,:,j+offset_z), opsA[ops_2_max_bpc_idx...])
                            convolve_BC_right!(view(x_temp,i,:,j), view(M,i+offset_x,:,j+offset_z), opsA[ops_2_max_bpc_idx...])
                            if i <= pad[1] || i > size(x_temp)[1]-pad[1] || j <= pad[3] || j > size(x_temp)[3]-pad[3]
                                convolve_interior!(view(x_temp,i,:,j), view(M,i+offset_x,:,j+offset_z), opsA[ops_2_max_bpc_idx...])
                            end

                        else
                            convolve_BC_left!(view(x_temp,i,:,j), view(M,i+offset_x,:,j+offset_z), opsA[ops_2_max_bpc_idx...], overwrite = false)
                            convolve_BC_right!(view(x_temp,i,:,j), view(M,i+offset_x,:,j+offset_z), opsA[ops_2_max_bpc_idx...], overwrite = false)
                            if i <= pad[1] || i > size(x_temp)[1]-pad[1] || j <= pad[3] || j > size(x_temp)[3]-pad[3]
                                convolve_interior!(view(x_temp,i,:,j), view(M,i+offset_y,:,j+offset_z), opsA[ops_2_max_bpc_idx...], overwrite = false)
                            end

                        end
                        for Lidx in ops_2
                            if Lidx != ops_2_max_bpc_idx[1]
                                convolve_BC_left!(view(x_temp,i,:,j), view(M,i+offset_x,:,j+offset_z), opsA[Lidx], overwrite = false)
                                convolve_BC_right!(view(x_temp,i,:,j), view(M,i+offset_x,:,j+offset_z), opsA[Lidx], overwrite = false)
                                if i <= pad[1] || i > size(x_temp)[1]-pad[1] || j <= pad[3] || j > size(x_temp)[3]-pad[3]
                                    convolve_interior!(view(x_temp,i,:,j), view(M,i+offset_x,:,j+offset_z), opsA[Lidx], overwrite = false)
                                elseif pad[2] - opsA[Lidx].boundary_point_count > 0
                                    convolve_interior!(view(x_temp,i,:,j), view(M,i+offset_x,:,j+offset_z), opsA[Lidx], overwrite = false, add_range = true, offset = pad[2] - opsA[Lidx].boundary_point_count)
                                end
                            end
                        end
                    end
                end
            end
            # convolve boundaries and unaccounted for interior in axis 3
            if length(ops_3) > 0
                for i in 1:size(x_temp)[1]
                    for j in 1:size(x_temp)[2]
                        # in the case of no axis 1 and 2 operators, we need to overwrite x_temp
                        if length(ops_1) == 0 && length(ops_2) == 0
                            convolve_BC_left!(view(x_temp,i,j,:), view(M,i+offset_x,j+offset_y,:), opsA[ops_3_max_bpc_idx...])
                            convolve_BC_right!(view(x_temp,i,j,:), view(M,i+offset_x,j+offset_y,:), opsA[ops_3_max_bpc_idx...])
                            if i <= pad[1] || i > size(x_temp)[1]-pad[1]
                                convolve_interior!(view(x_temp,i,j,:), view(M,i+offset_x,j+offset_y,:), opsA[ops_3_max_bpc_idx...])
                            end

                        else
                            convolve_BC_left!(view(x_temp,i,j,:), view(M,i+offset_x,j+offset_y,:), opsA[ops_3_max_bpc_idx...], overwrite = false)
                            convolve_BC_right!(view(x_temp,i,j,:), view(M,i+offset_x,j+offset_y,:), opsA[ops_3_max_bpc_idx...], overwrite = false)
                            if i <= pad[1] || i > size(x_temp)[1]-pad[1] || j <= pad[2] || j > size(x_temp)[2]-pad[2]
                                convolve_interior!(view(x_temp,i,j,:), view(M,i+offset_x,j+offset_y,:), opsA[ops_3_max_bpc_idx...], overwrite = false)
                            end

                        end
                        for Lidx in ops_3
                            if Lidx != ops_3_max_bpc_idx[1]
                                convolve_BC_left!(view(x_temp,i,j,:), view(M,i+offset_x,j+offset_y,:), opsA[Lidx], overwrite = false)
                                convolve_BC_right!(view(x_temp,i,j,:), view(M,i+offset_x,j+offset_y,:), opsA[Lidx], overwrite = false)
                                if i <= pad[1] || i > size(x_temp)[1]-pad[1] || j <= pad[2] || j > size(x_temp)[2]-pad[2]
                                    convolve_interior!(view(x_temp,i,j,:), view(M,i+offset_x,j+offset_y,:), opsA[Lidx], overwrite = false)
                                elseif pad[3] - opsA[Lidx].boundary_point_count > 0
                                    convolve_interior!(view(x_temp,i,j,:), view(M,i+offset_x,j+offset_y,:), opsA[Lidx], overwrite = false, add_range = true, offset = pad[3] - opsA[Lidx].boundary_point_count)
                                end
                            end
                        end
                    end
                end
            end
        end

        # Here we compute mul! (additively) for every operator in opsB

        operating_dims = zeros(Int64,3)
        # need to consider all dimensions and operators to determine the truncation
        # of M to x_temp
        for L in A.ops
            operating_dims[diff_axis(L)] = 1
        end

        x_temp_1, x_temp_2, x_temp_3 = size(x_temp)

        for L in opsB
            N = diff_axis(L)
            if N == 1
                mul!(x_temp, L, view(M,1:x_temp_1+2,1:x_temp_2,1:x_temp_3), overwrite = false)
            elseif N == 2
                 mul!(x_temp, L, view(M,1:x_temp_1,1:x_temp_2+2,1:x_temp_3), overwrite = false)
            else
                mul!(x_temp, L, view(M,1:x_temp_1,1:x_temp_2,1:x_temp_3+2), overwrite = false)
            end
        end

    # The case where we call everything in A.ops using the fallback mul!
    else
        # operating_dims indicates which dimensions we are multiplying along
        operating_dims = zeros(Int64,3)
        for L in A.ops
            operating_dims[diff_axis(L)] = 1
        end

        x_temp_1, x_temp_2, x_temp_3 = size(x_temp)

        # Handle first case non-additively
        N = diff_axis(A.ops[1])

        if N == 1
            mul!(x_temp, A.ops[1], view(M,1:x_temp_1+2,1:x_temp_2,1:x_temp_3))
        elseif N == 2
             mul!(x_temp, A.ops[1], view(M,1:x_temp_1,1:x_temp_2+2,1:x_temp_3))
        else
            mul!(x_temp, A.ops[1], view(M,1:x_temp_1,1:x_temp_2,1:x_temp_3+2))
        end

        for L in A.ops[2:end]
            N = diff_axis(L)
            if N == 1
                mul!(x_temp, L, view(M,1:x_temp_1+2,1:x_temp_2,1:x_temp_3), overwrite = false)
            elseif N == 2
                 mul!(x_temp, L, view(M,1:x_temp_1,1:x_temp_2+2,1:x_temp_3), overwrite = false)
            else
                mul!(x_temp, L, view(M,1:x_temp_1,1:x_temp_2,1:x_temp_3+2), overwrite = false)
            end
        end
    end
end
