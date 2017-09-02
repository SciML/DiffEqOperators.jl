# Basics of DiffEqOperators

In this tutorial we will explore the basic functionalities of PDEOperator which is used to obtain the discretizations of PDEs of appropriate derivative and approximation order.

So an operator API is as follows:-

    A = DerivativeOperator{T}
        (
            derivative_order,
            approximation_order,
            grid_step_size,
            grid_size,
            :LBC,
            :RBC;
            BC=(LBV, RBV)
        );
Currently we support the `Dirichlet 0/1`, `Neumann 0/1`, `periodic` and `Robin` boundary conditions.

Taking a specific example
    
    A = DerivativeOperator{Float64}(2,2,1/99,10,:Dirichlet,:Dirichlet; BC=(u[1],u[end]))

this is the time independent Dirichlet BC. You can also specify a time dependent Dirichlet BC as follows:-
    
    A = DerivativeOperator{Float64}(2,2,1/99,10,:Dirichlet,:Dirichlet; bndry_fn=(t->(u[1]*cos(t)),u[end]))

We have generated an operator which produces the 2nd order approximation of the Laplacian. We can checkout the stencil as follows:-

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

Note that we **don't** need to define the `BC` only for `:D0` and `:periodic` boundary conditions so you can ignore it.


Now coming to the main functionality of DiffEqOperators ie. taking finite difference discretizations of functions.

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

You can also take derivatives of matrices using `A*M` or `M*A` where the order of multiplication decides the axis along which we want to take derivatives.

    julia> xarr = linspace(0,1,51)
    julia> yarr = linspace(0,1,101)
    julia> dx = xarr[2]-xarr[1]
    julia> dy = yarr[2]-yarr[1]
    julia> F = [x^2+y for x = xarr, y = yarr]
    julia> A = DerivativeOperator{Float64}(2,2,dx,length(yarr),:None,:None)
    julia> B = DerivativeOperator{Float64}(2,2,dy,length(yarr),:None,:None)

    julia> # A*F calculates derivatives along the x axis ie. keeping y constant
    julia> A*F
    51×101 Array{Float64,2}:
     2.0  2.0  2.0  2.0  2.0  2.0  2.0  …  2.0  2.0  2.0  2.0  2.0  2.0  2.0
     2.0  2.0  2.0  2.0  2.0  2.0  2.0     2.0  2.0  2.0  2.0  2.0  2.0  2.0
     2.0  2.0  2.0  2.0  2.0  2.0  2.0     2.0  2.0  2.0  2.0  2.0  2.0  2.0
     2.0  2.0  2.0  2.0  2.0  2.0  2.0     2.0  2.0  2.0  2.0  2.0  2.0  2.0
     2.0  2.0  2.0  2.0  2.0  2.0  2.0     2.0  2.0  2.0  2.0  2.0  2.0  2.0
     2.0  2.0  2.0  2.0  2.0  2.0  2.0  …  2.0  2.0  2.0  2.0  2.0  2.0  2.0
     2.0  2.0  2.0  2.0  2.0  2.0  2.0     2.0  2.0  2.0  2.0  2.0  2.0  2.0
     2.0  2.0  2.0  2.0  2.0  2.0  2.0     2.0  2.0  2.0  2.0  2.0  2.0  2.0
     2.0  2.0  2.0  2.0  2.0  2.0  2.0     2.0  2.0  2.0  2.0  2.0  2.0  2.0
     ⋮                        ⋮         ⋱       ⋮                        ⋮  
     2.0  2.0  2.0  2.0  2.0  2.0  2.0     2.0  2.0  2.0  2.0  2.0  2.0  2.0
     2.0  2.0  2.0  2.0  2.0  2.0  2.0     2.0  2.0  2.0  2.0  2.0  2.0  2.0
     2.0  2.0  2.0  2.0  2.0  2.0  2.0     2.0  2.0  2.0  2.0  2.0  2.0  2.0
     2.0  2.0  2.0  2.0  2.0  2.0  2.0  …  2.0  2.0  2.0  2.0  2.0  2.0  2.0
     2.0  2.0  2.0  2.0  2.0  2.0  2.0     2.0  2.0  2.0  2.0  2.0  2.0  2.0
     2.0  2.0  2.0  2.0  2.0  2.0  2.0     2.0  2.0  2.0  2.0  2.0  2.0  2.0
     2.0  2.0  2.0  2.0  2.0  2.0  2.0     2.0  2.0  2.0  2.0  2.0  2.0  2.0
     2.0  2.0  2.0  2.0  2.0  2.0  2.0     2.0  2.0  2.0  2.0  2.0  2.0  2.0
     2.0  2.0  2.0  2.0  2.0  2.0  2.0  …  2.0  2.0  2.0  2.0  2.0  2.0  2.0

    julia> # F*B calculates derivatives along the y axis ie. keeping x constant
    julia> F*B
    101×51 Array{Float64,2}:
      0.0           1.04083e-13   6.93889e-14  …   2.22045e-12   2.22045e-12
      0.0           3.46945e-14   0.0              0.0           0.0        
      0.0          -3.46945e-14  -6.93889e-14      0.0           0.0        
      6.93889e-14   0.0           6.93889e-14      0.0           0.0        
      0.0           0.0           0.0              0.0           0.0        
     -6.93889e-14  -6.93889e-14  -6.93889e-14  …   0.0           0.0        
      1.38778e-13   1.38778e-13   1.38778e-13      0.0           0.0        
     -1.38778e-13  -1.38778e-13  -2.77556e-13      0.0           0.0        
      0.0           0.0           0.0              0.0           0.0        
      ⋮                                        ⋱                 ⋮          
      0.0           0.0           0.0              4.44089e-12   4.44089e-12
     -1.11022e-12  -1.11022e-12  -1.11022e-12     -4.44089e-12  -4.44089e-12
      1.11022e-12   1.11022e-12   1.11022e-12      2.22045e-12   2.22045e-12
      0.0           0.0           0.0          …   0.0           0.0        
      0.0           0.0           0.0              0.0           0.0        
      0.0           0.0           0.0              0.0           0.0        
      0.0           0.0           0.0              0.0           0.0        
      0.0           0.0           0.0              0.0           0.0        
      0.0           0.0           0.0          …   8.88178e-12   8.88178e-12



**Note:** Please take care that the boundary values passed to the operator match the initial boundary conditions. The operator with the boundary condition is meant to enforce the boundary condition rather bring the boundaries to that state. ~~Right now we support only **constant** boundaries conditions, time dependent conditions will supported in later versions.~~
Support for time dependent Dirichlet BC has been added. 

**Note:** If you want to parallelize the operation of PDEOperator, please start Julia by specifying the number of threads using `export JULIA_NUM_THREADS=<desired number of threads>`
