using Test, LinearAlgebra
using DiffEqOperators, OrdinaryDiffEq

@testset "Matrix Free Operator constructors" begin
  iden_op = MatrixFreeOperator(identity)
  @test iden_op == MatrixFreeOperator(identity, (nothing,), size = nothing, opnorm = true, ishermitian = false)
  @test_throws AssertionError MatrixFreeOperator(identity, ())
  @test_throws AssertionError MatrixFreeOperator(identity, (0,0,0))
  @test_throws AssertionError MatrixFreeOperator(identity, [1])
end

@testset "Matrix Free Operator methods" begin
  iden_op = MatrixFreeOperator(identity)
  iden_op1 = MatrixFreeOperator(identity, (2,2), size = (2,2),
                                opnorm = identity)
  f = (du, u, p) -> begin
    cumsum!(du, u)
    @. du = p*du
  end
  g = (du, u, p, t) -> begin
    cumsum!(du, u)
    @. du = p*du * t
  end
  p = 1.
  t = 1.
  A = MatrixFreeOperator(f, (p,), size=(5,5), opnorm=5)
  B = MatrixFreeOperator(hcat, (1.,0.))
  C = MatrixFreeOperator(g, (p,t), size=(5,5), opnorm=5)

  # Base
  @test size(iden_op1) == (2,2)
  @test size(iden_op1, 2) == 2
  @test size(iden_op1, 3) == 1
  @test_throws ErrorException size(iden_op1, 0)
  @test_throws ErrorException size(iden_op1, -1)
  @test_throws ErrorException size(iden_op)
  @test_throws ErrorException size(iden_op, 5)

  # Linear Algebra
  @test ishermitian(iden_op) == false
  @test opnorm(iden_op) == true
  @test opnorm(A, 5) == 5
  @test opnorm(iden_op1, Inf) == Inf

  # DiffEqBase
  DiffEqBase.numargs(iden_op) == 4

  # Interface
  @test DiffEqOperators.isconstant(iden_op) == true
  @test DiffEqOperators.isconstant(iden_op1) == false
  @test update_coefficients!(iden_op, 0, 0, 0) == MatrixFreeOperator(identity)
  update_coefficients!(B, 0, 1., 1.)
  @test B.args == (1., 1.)
  @test A([.1, .1], 1., 0.) == [.1, .2] # with one arg
  @test A(zeros(2), [.1, .1], 1., 0.) == [.1, .2]
  @test C([.1, .1], 1., 2.) == [.2, .4] # with two args
  @test C(zeros(2), [.1, .1], 1., 2.) == [.2, .4]
  A.args = (1.,)
  C.args = (1., 2.)
  @test mul!(zeros(2), A, [.1, .1]) == [.1, .2]
  @test mul!(zeros(2), C, [.1, .1]) == [.2, .4]
  @test_throws DimensionMismatch mul!(zeros(2,3), A, zeros(3,2))
  @test mul!(zeros(2,2), A, [.1 .1; .1 .1]) == [.1 .1; .2 .2]
  @test mul!(zeros(2,2), C, [.1 .1; .1 .1]) == [.2 .2; .4 .4]
end

@testset "Matrix Free Operator example" begin
  f = (du, u, p) -> begin
    cumsum!(du, u)
    @. du = p*du
  end
  p = 1.
  A = MatrixFreeOperator(f, (p,), size=(5,5), opnorm=5)
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
  O = MatrixFreeOperator(Q!, (p, 0.), size=(2,2))

  # solve DE numerically
  T = 2
  f_0 = [2.0; 1/2]
  prob = ODEProblem(O,f_0,(0.0,T),p)
  sol = solve(prob, LinearExponential(krylov=:simple, m=3), tstops=THRESHOLD)
  @test sol(THRESHOLD) ≈ sol_analytic(THRESHOLD) atol = 1e-3
  @test sol(T) ≈ sol_analytic(T) atol = 1e-3
end
