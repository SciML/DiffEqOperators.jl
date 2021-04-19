using Test, LinearAlgebra, Random, DiffEqOperators
using DiffEqBase
using DiffEqBase: isconstant
using DiffEqOperators: DiffEqScaledOperator, DiffEqOperatorCombination, DiffEqOperatorComposition

@testset "Operator Compostion" begin
  Random.seed!(0)
  A1 = rand(2,3)
  A2 = rand(3,2)
  b = rand()
  B = rand(2,2)
  L0 = DiffEqArrayOperator(Diagonal([1.0, 1.0])) - I + zeros(2, 2) # this operator is 0
  L = DiffEqArrayOperator(A1) * DiffEqArrayOperator(A2) + DiffEqScalar(b) * DiffEqArrayOperator(B) + L0

  # Structure
  @test isa(L, DiffEqOperatorCombination)
  L1, L2, L3 = getops(L)
  @test isa(L1, DiffEqOperatorComposition)
  @test isa(L2, DiffEqScaledOperator)
  @test isa(L3, DiffEqOperatorCombination)

  # Verify that L3 and L0 == 0
  @test all(Matrix(L3) .== 0)
  @test all(Matrix(L0) .== 0)

  # Operations
  Lfull = Matrix(L)
  @test opnorm(L) ≈ opnorm(Lfull)
  @test size(L) == size(Lfull)
  @test L[1,2] ≈ Lfull[1,2]
  u = [1.0, 2.0]; du = zeros(2)
  @test L * u ≈ Lfull * u
  mul!(du, L, u); @test du ≈ Lfull * u
  Lf = factorize(L)
  ldiv!(du, Lf, u); @test Lfull * du ≈ u
  @test exp(L) ≈ exp(Lfull)
end

@testset "Mutable Composite Operators" begin
  A = zeros(2,2); fA = (A,u,p,t) -> fill!(A, t)
  B = zeros(2,2); fB = (B,u,p,t) -> fill!(B, 2t)
  L = DiffEqArrayOperator(A; update_func=fA) * DiffEqArrayOperator(B; update_func=fB)
  @test isconstant(L) == false
  u = [1.0, 2.0]
  t = 1.1
  @test L(u,nothing,t) ≈ fill(t, (2,2)) * (fill(2t, (2,2)) * u)
end
