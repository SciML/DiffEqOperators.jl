# using SparseArrays, DiffEqOperators, LinearAlgebra, Random

# # A (N,N+2) strang_matrix
# function strang_matrix(N)
#   A = zeros(N,N+2)
#   v = [1.0, -2.0, 1.0]
#   for i in 1:N
#     A[i,i:i+2] = v
#   end
#   A
# end

# # tests for full and sparse function
# @testset "Full and Sparse functions:" begin
#     N = 100
#     d_order = 2
#     approx_order = 2
#     x = collect(1:1.0:N).^2

#     A = InteriorOperator{Float64}(d_order,approx_order,1.0,N)
#     mat = convert(Array,A)
#     @test size(mat) = (N, N+2)

#     @test convert(Array, A, 10) == strang_matrix(10) # Strang Matrix is defined with the center term +ve
#     @test convert(Array, A, N) == strang_matrix(N) # Strang Matrix is defined with the center term +ve
#     @test opnorm(A, Inf) == opnorm(mat, Inf)

#     # testing correctness
#     N = 1000
#     d_order = 4
#     approx_order = 10
#     y = collect(1:1.0:N).^4 - 2*collect(1:1.0:N).^3 + collect(1:1.0:N).^2;
#     y = convert(Array{BigFloat, 1}, y)

#     A = InteriorOperator{BigFloat}(d_order,approx_order,one(BigFloat),N)
#     boundary_points = A.boundary_point_count
#     mat = convert(Array, A, N)
#     sp_mat = sparse(A)
#     @test mat == sp_mat

#     res = A*y
#     @test res[boundary_points[1] + 1: N - boundary_points[2]] ≈ 24.0*ones(N - sum(boundary_points)) atol=10.0^-approx_order
#     @test A*y ≈ mat*y atol=10.0^-approx_order
#     @test A*y ≈ sp_mat*y atol=10.0^-approx_order
#     @test sp_mat*y ≈ mat*y atol=10.0^-approx_order
# end

# @testset "Indexing tests" begin
#     N = 1000
#     d_order = 2
#     approx_order = 4

#     A = InteriorOperator{Float64}(d_order,approx_order,1.0,N)
#     @test A[1,1] ≈ -0.083333 atol=1e-4
#     @test A[:,1] == (convert(Array,A))[:,1]
#     @test A[10,20] == 0

#     for i in 1:N
#         @test A[i,i] == A.stencil_coefs[div(A.stencil_length, 2) + 1]
#     end

#     # Indexing Tests
#     N = 1000
#     d_order = 2
#     approx_order = 2

#     A = InteriorOperator{Float64}(d_order,approx_order,1.0,N)
#     M = convert(Array,A)

#     @test A[1,1] == -2.0
#     @test A[1:4,1] == M[1:4,1]
#     @test A[5,2:10] == M[5,2:10]
#     @test A[60:100,500:600] == M[60:100,500:600]
# end

# # @testset begin "Operations on matrices"
# #     N = 51
# #     M = 101
# #     d_order = 2
# #     approx_order = 2

# #     xarr = range(0,stop=1,length=N)
# #     yarr = range(0,stop=1,length=M)
# #     dx = xarr[2]-xarr[1]
# #     dy = yarr[2]-yarr[1]
# #     F = [x^2+y for x = xarr, y = yarr]

# #     A = InteriorOperator{Float64}(d_order,approx_order,dx,length(xarr)
# #     B = InteriorOperator{Float64}(d_order,approx_order,dy,length(yarr)

# #     @test A*F ≈ 2*ones(N,M) atol=1e-2
# #     @test F*B ≈ 8*ones(N,M) atol=1e-2
# #     @test A*F*B ≈ zeros(N,M) atol=1e-2

# #     G = [x^2+y^2 for x = xarr, y = yarr]

# #     @test A*G ≈ 2*ones(N,M) atol=1e-2
# #     @test G*B ≈ 8*ones(N,M) atol=1e-2
# #     @test A*G*B ≈ zeros(N,M) atol=1e-2
# # end

# # @testset "Linear combinations of operators" begin
# #     # Only tests the additional functionality defined in "operator_combination.jl"
# #     N = 10
# #     Random.seed!(0); LA = DiffEqArrayOperator(rand(N,N))
# #     LD = InteriorOperator{Float64}(2,2,1.0,N)
# #     @test_broken begin
# #       L = 1.1*LA - 2.2*LD + 3.3*I
# #       # Builds convert(L) the brute-force way
# #       fullL = zeros(N,N)
# #       v = zeros(N)
# #       for i = 1:N
# #           v[i] = 1.0
# #           fullL[:,i] = L*v
# #           v[i] = 0.0
# #       end
# #       @test convert(L) ≈ fullL
# #       @test exp(L) ≈ exp(fullL)
# #       for p in [1,2,Inf]
# #           @test opnorm(L,p) ≈ opnorm(fullL,p)
# #           @test opnormbound(L,p) ≈ 1.1*opnorm(LA,p) + 2.2*opnorm(LD,p) + 3.3
# #       end
# #   end
# # end
