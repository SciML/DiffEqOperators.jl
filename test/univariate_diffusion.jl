workspace()
using DiffEqOperators, DifferentialEquations, StaticArrays

# Note: despite already having all notation to run as a test, at this
# point I still prefer to # the tests, and check them one by one
# manually. I made that choice because it's much faster to run new tests
# This can be easily undune by removing each of the # before
# @testset, @test and end

# NEXT steps:

# ISSUE1: shouldn't sigma'*A*u*sigma be NxN? result is Nx1!
# ISSUE2: A has very high values in the boundaries (look at full(A))
# ISSUE3: For some reason, derivative opperator function doesn't accept
# upperBC instead of :Neumann0.
# ISSUE4: look at 3.1.3 for comments. Also applies to 3.1.4 (I already made
# the changes in 3.1.4, but originally is was defining mat and  sp_mat
# like in 3.1.3)
# ISSUE5: redefine y in 3.1.4 and understand its 2nd test (and why/if its
# relevant here)
# ISSUE6: redefine the number on test 1 of 3.1.5 (why 13.717407?) and why
# look only at A[10,20] for the third test?


# After the basics are done, ask Jesse:
# 1) Possible interesting extension: BC as functions
# example: A = DerivativeOperator{Float64}(2,2,1/99,10,:Dirichlet,:Dirichlet; bndry_fn=(t->(u[1]*cos(t)),u[end]))
# 2)

# INDEX/ROADMAP:
# 1) Test intro
# 2) Generate results with DiffEqOperators
# 3) Start testing:
# 3.1) Test reflecting boudaries:
#(from neumann.jl):
# 3.1.1) testing first derivative
# 3.1.2) testing interior
# (from derivative_operators_interface.jl):
# 3.1.3) Testing for full and sparce matrices
# 3.1.4) Testing "default" results vs high order
# 3.1.4) Testing indexing tests
# 3.1.5) (not started) Testing operations on matrices
# (from ?)
# 3.1.6 (not started) testing type
# 3.1.7 (not started) testing lenght
# from Pang and ?
# 3.1.8 Comparing solution to discretized A:

# 1)
#@testset "Univariate Diffusion" begin

# 2) set x, sigma(x), lowerBC; upperBC, u0
N = 100
d_order = 2
approx_order = 2
lowerx = 0
upperx = 10
dt = (upperx-lowerx)/(N-1)
x = collect(lowerx : dt : upperx);
u0 = -(x - 0.5).^2; # any function of x
fsigma(a,b) = a*x + b # sigma(x)
sigma = fsigma(0,1) # unit variance
lowerBC = :Neumann
upperBC = :Neumann0
# later, for tests:
d_order_high = 4
approx_order_high = 10

# DiffEqOperators output
A = DerivativeOperator{Float64}(d_order,approx_order,dt,N,
						lowerBC,upperBC;BC=(u0[1],u0[end]));
res = A*u0
discr = sigma'*res*sigma

# 3) TESTS:
# 3.1) Neumann boundaries:

# 3.1.1) Testing first derivative
# Neumann calls for zero derivative only at the boundary points, here is zero everywhere because process is driftless
FD = DerivativeOperator{Float64}(1,approx_order,dt,N,
                        lowerBC,:Neumann0;BC=(u0[1],u0[end]))
first_deriv = FD*res
# @test first_deriv ≈ zeros(n,1) atol=10.0^-1

# 3.1.2) Testing interior
boundary_points = A.boundary_point_count
# @test res[boundary_points[1] + 1: N - boundary_points[2]] ≈
#            2.0*ones(N - sum(boundary_points)) atol=10.0^approx_order

# 3.1.3) Testing for full and sparce matrices
   #@test sparse(A) == full(A)
   #@test full(A, N) == -Strang(N)

   # Why Chris defines mat and sp_mat? Why test mat = fsp_mat and then
   # test full(A) = sp_mat?
      #mat = full(A)
      #sp_mat = sparse(A)
      #@test mat == sp_mat;
      #@test full(A, 10) == -Strang(10); # Strang Matrix is defined with the center term +ve
      #@test full(A, N) == -Strang(N);   # Strang Matrix is defined with the center term +ve
      # @test full(A) == sp_mat

# 3.1.4) Using high order and looking how similar are the results
   #? y = collect(1:1.0:N).^4 - 2*collect(1:1.0:N).^3 + collect(1:1.0:N).^2;
   y = convert(Array{BigFloat, 1}, y)
   A = DerivativeOperator{BigFloat}(d_order_high,approx_order_high,
               BigFloat((upperx-lowerx)/(N-1)),N,lowerBC,upperBC)
   boundary_points = A.boundary_point_count

   #@test  full(A, N) == sparse(A);

   res = A*y
   #2nd test: #@test res[boundary_points[1] + 1: N - boundary_points[2]] ≈ 24.0*ones(N - sum(boundary_points)) atol=10.0^-approx_order;
   #@test sigma'*A*y*sigma ≈ sigma'*full(A, N)*y*sigma atol=10.0^-approx_order;
   #@test sigma'*A*y*sigma ≈ sigma'*sparse(A)*y*sigma atol=10.0^-approx_order;
   #@test sigma'sparse(A)*y*sigma ≈ sigma'full(A, N)*y*sigma atol=10.0^-approx_order;

# 3.1.5) Testing indexing tests
    A = DerivativeOperator{Float64}(d_order_high,approx_order_high,dt,N,lowerBC,upperBC)
    #? @test A[1,1] ≈ 13.717407 atol=1e-4
    #@test A[:,1] == (full(A))[:,1]
    #? @test A[10,20] == 0

    for i in 1:N
        @test A[i,i] == A.stencil_coefs[div(A.stencil_length, 2) + 1]
    end

    # Indexing Tests
    N = 1000
    d_order = 2
    approx_order = 2

    A = DerivativeOperator{Float64}(d_order_high,approx_order_high,1.0,N,:Dirichlet0,:Dirichlet0)
    M = full(A)

    @test A[1,1] == -2.0
    @test A[1:4,1] == M[1:4,1]
    @test A[5,2:10] == M[5,2:10]
    @test A[60:100,500:600] == M[60:100,500:600]

# 3.1.6) (not started)Testing operations on matrices

# 3.1.7 (not started) testing type
# think how to write something like:
#@test sol.t == collect(0:0.25:1.0) ??
#@test typeof(sol) <: DESolution ??
# 3.1.8 (not started) testing lenght
# think how to write something like:
#@test length(sol.u) == 5 ??

# 3.1.9 Comparing solution to discretized A:
function LinearOperator(x,mu,sigma_2)
    I = length(x); # size of x
    # Uniform grid:
    Delta_2 = (x[2]-x[1])^2;
    # Equations 7-9 in the notes
    X = sigma_2/(2*Delta_2);
    Y = - sigma_2/Delta_2;
    A = Tridiagonal(X[2:I],Y[1:I], X[1:I-1])
    # reflecting component:
    A[1,1] = Y[1] + X[1];
    A[I,I] = Y[I] + X[I];
    return(A)
end
# function parameters:
H = LinearOperator(x,zeros(N),A_mulB!(sigma,sigma))
# is it A that should be equal to H? I don't think so...
# @test A-H ≈ 0 atol=10.0^approx_order

#end
