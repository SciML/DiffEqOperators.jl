using Test
using DiffEqOperators

@testset "Matrix Free Operator" begin
  f = cumsum!
  args = nothing
  A = MatrixFreeOperator(f,args)
  b = rand(5)
  buffer = similar(b)
  @test A*b == f(buffer,b)
  A = MatrixFreeOperator(f,args)
  B = rand(5,5)
  buffer = similar(B)
  @test A*B ≈ f(buffer,B,dims=1)
end
