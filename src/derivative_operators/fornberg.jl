#############################################################
#= Fornberg algorithm

This implements the Fornberg (1988) algorithm (https://doi.org/10.1090/S0025-5718-1988-0935077-0)
and hermite-based finite difference Fornberg(2020) algorithm (https://doi.org/10.1093/imanum/draa006)
to obtain Finite Difference weights over arbitrary points to arbitrary order.

Inputs:
        order: The derivative order for which we need the coefficients
        x0   : The point in the array 'x' for which we need the coefficients
        x    : A dummy array with relative coordinates, e.g., central differences
               need coordinates centred at 0 while those at boundaries need
               coordinates starting from 0 to the end point
        dfdx : optional argument to consider weights of the first-derivative of function or not
                if    dfdx == false (default kwarg), implies Fornberg(1988)
                      dfdx == true,                  implies Fornberg(2020)

    Outputs:
        if dfdx == false (default kwarg),   _C : weights to approximate derivative of required order using function values only.
                                 else,   _D,_E : weights to approximate derivative of required order using function and its first- 
                                                 derivative values respectively.                                             

=#
function calculate_weights(order::Int, x0::T, x::AbstractVector; dfdx::Bool = false) where T<:Real
    N = length(x)
    @assert order < N "Not enough points for the requested order."
    M = order
    c1 = one(T)
    c4 = x[1] - x0
    C = zeros(T, N, M+1)
    C[1,1] = 1
    @inbounds for i in 1 : N-1
        i1 = i + 1
        mn = min(i, M)
        c2 = one(T)
        c5 = c4
        c4 = x[i1] - x0
        for j in 0 : i-1
            j1 = j + 1
            c3 = x[i1] - x[j1]
            c2 *= c3
            if j == i-1
                for s in mn : -1 : 1
                    s1 = s + 1
                    C[i1,s1] = c1*(s*C[i,s] - c5*C[i,s1]) / c2
                end
                C[i1,1] = -c1*c5*C[i,1] / c2
           end
            for s in mn : -1 : 1
                s1 = s + 1
                C[j1,s1] = (c4*C[j1,s1] - s*C[j1,s]) / c3
            end
            C[j1,1] = c4 * C[j1,1] / c3
        end
        c1 = c2
    end
    #=
        This is to fix the problem of numerical instability which occurs when the sum of the stencil_coefficients is not
        exactly 0.
        https://scicomp.stackexchange.com/questions/11249/numerical-derivative-and-finite-difference-coefficients-any-update-of-the-fornb
        Stack Overflow answer on this issue.
        http://epubs.siam.org/doi/pdf/10.1137/S0036144596322507 - Modified Fornberg Algorithm
    =#
    _C = C[:,end]
    _C[div(N,2)+1] -= sum(_C)
     if dfdx == false
        return _C
     else
        A = x .- x';
        s = sum(1 ./ (A + I(N)), dims = 1) .- 1;
        cp = factorial.(0:M);
        cc = C./cp'
        c̃ = zeros(N, M+2);
        for k in 1:M+1
           c̃[:,k+1] = sum(cc[:,1:k].*cc[:,k:-1:1], dims = 2);
        end
        E = c̃[:,1:M+1] - (x .- x0).*c̃[:,2:M+2];
        D = c̃[:,2:M+2] + 2*E.*s';
        D = D.*cp';
        E = E.*cp';

        _D = D[:,end];   _E = E[:,end]
        return _D, _E
    end
end
