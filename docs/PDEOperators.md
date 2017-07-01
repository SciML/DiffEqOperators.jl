In this tutorial we will explore the basic functionalities of PDEOperator which is used to obtain the discretizations of PDEs of appropriate derivative and approximation order.

So an operator API is as follows:-

    A = LinearOperator{T}
        (
            derivative_order,
            approximation_order,
            grid_step_size,
            grid_size,
            :LBC,
            :RBC;
            bndry_fn=(LBV, RBV)
        );
Currently we support the `Dirichlet 0/1`, `Neumann`, `periodic` and `Robin` boundary conditions.

Taking a specific example
    
    A = LinearOperator{Float64}(2,2,1/99,10,:D1,:D1; bndry_fn=(u[1],u[end]))
generates an operator which produces the 2nd order approximation of the Laplacian. We can checkout the stencil as follows:-

    julia> A.stencil_coefs
    3-element SVector{3,Float64}:
      1.0
     -2.0
      1.0

We can get the linear operator as a matrix as follows:-

    julia> full(A)
    10×10 Array{Float64,2}:
     -2.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
      1.0  -2.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
      0.0   1.0  -2.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0
      0.0   0.0   1.0  -2.0   1.0   0.0   0.0   0.0   0.0   0.0
      0.0   0.0   0.0   1.0  -2.0   1.0   0.0   0.0   0.0   0.0
      0.0   0.0   0.0   0.0   1.0  -2.0   1.0   0.0   0.0   0.0
      0.0   0.0   0.0   0.0   0.0   1.0  -2.0   1.0   0.0   0.0
      0.0   0.0   0.0   0.0   0.0   0.0   1.0  -2.0   1.0   0.0
      0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0  -2.0   1.0
      0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0  -2.0

Note that we don't need to define the `bndry_fn` only for `:D0` and `:periodic` boundary conditions.


Now coming to the main functionality of PDEOperators ie. taking finite difference discretizations of functions.

    julia> x = collect(0 : 1/99 : 1);
    julia> u0 = x.^2 -x;
    julia> res = A*u0
    100-element Array{Float64,1}:
     -98.0
       2.0
       2.0
       2.0
       2.0
       2.0
       2.0
       ⋮  
       2.0
       2.0
       2.0
       2.0
       2.0
       2.0
     -98.0

The derivative values at the boundaries are in accordance with the `Dirichlet` boundary condition.

**Note:** Please take care that the boundary values passed to the operator match the initial boundary conditions. The operator with the boundary condition is meant to enforce the boundary condition rather bring the boundaries to that state. Right now we support only **constant** boundaries conditions, time dependent conditions will supported in later versions.

**Note:** If you want to parallelize the operation of PDEOperator, please start Julia by specifying the number of threads using `export JULIA_NUM_THREADS=<desired number of threads>`