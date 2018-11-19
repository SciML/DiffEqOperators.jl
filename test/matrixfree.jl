using Test, LinearAlgebra
using DiffEqOperators, OrdinaryDiffEq

@testset "Matrix Free Operator" begin
  f = (du, u, p) -> begin
    cumsum!(du, u)
    @. du = p*du
  end
  p = 1.
  A = MatrixFreeOperator(f, (p,), size=(5,5), opnorm=(p)->5)
  b = rand(5)
  @test is_constant(A)
  prob = ODEProblem(A, b, (0,1.), p)
  @test_nowarn solve(prob, Tsit5())

  f = (du, u, p, t) -> cumsum!(du, u)
  args = (1, 1)
  A = MatrixFreeOperator(f,args, size=(5,5), opnorm=(p)->5)
  @test is_constant(A) == false
  b = rand(5)
  @test_nowarn solve(prob, LawsonEuler(krylov=true, m=5), dt=0.1)

  A1 = [[4 -6]; [1 -1]]
  A2 = [[1 2]; [3 2]]
  THRESHOLD = 1
  # analytic solution for the DE
  u1(t) = t < THRESHOLD ? exp(t)*(-1+3exp(t)) : exp(-3-t)*(exp(5)*(-2+7exp(1))+exp(5t)*(-3+8exp(1)))/5
  u2(t) = t < THRESHOLD ? -exp(t)/2+exp(2t)   : (2-7exp(1))*exp(2-t)/5+exp(-3+4t)*(-3+8exp(1))*(3/10)
  sol_analytic(t) = [u1(t), u2(t)]
  # setups for DE solvers
  function Q!(df, f, p, t)
    i = t >= THRESHOLD ? 2 : 1
    A = p[i]
    mul!(df, A, f)
  end
  p = (A1,A2)
  O = MatrixFreeOperator(Q!, (p, 0.), size=(2,2), opnorm=(p)->10)
  # solve DE numerically
  T = 2
  f_0 = [2.0; 1/2]
  prob = ODEProblem(O,f_0,(0.0,T),p)
  sol = solve(prob, LinearExponential(krylov=:simple, m=3), tstops=THRESHOLD)
  @test sol(THRESHOLD) ≈ sol_analytic(THRESHOLD) atol = 1e-3
  @test sol(T) ≈ sol_analytic(T) atol = 1e-3
end
