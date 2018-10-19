using Test, LinearAlgebra
using DiffEqOperators, OrdinaryDiffEq

@testset "Matrix Free Operator" begin
  f = cumsum!
  A = MatrixFreeOperator(f)
  b = rand(5)
  prob = ODEProblem(A, b, (0,1.))
  @test_nowarn solve(prob, Tsit5())
  prob = SplitODEProblem(A, (du,u,p,t)->0, b, (0,1.))
  Base.size(::typeof(A), n) = n==1 || n == 2 ? 5 : 1
  LinearAlgebra.opnorm(::typeof(A), n::Real) = n==Inf ? 5 : nothing
  LinearAlgebra.ishermitian(::typeof(A)) = false
  @test_nowarn solve(prob, LawsonEuler(krylov=true, m=5), dt=0.1)
  f = (du, u, p, t) -> cumsum!(du, u)
  args = (1, 1)
  A = MatrixFreeOperator(f,args)
  b = rand(5)
  Base.size(::typeof(A), n) = n==1 || n == 2 ? 5 : 1
  LinearAlgebra.opnorm(::typeof(A), n::Real) = n==Inf ? 5 : nothing
  LinearAlgebra.ishermitian(::typeof(A)) = false
  @test_nowarn solve(prob, LawsonEuler(krylov=true, m=5), dt=0.1)
end
