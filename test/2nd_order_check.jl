using DiffEqOperators, Test, LinearAlgebra

@testset "Second Order Value Check" begin
  ord = 2               # Order of approximation
  Δx = 0.025π
  N = Int(2*(π/Δx)) -2
  x = -π:Δx:π-Δx
  D1  = CenteredDifference(1,ord,Δx,N,:periodic,:periodic);    # 1nd Derivative
  D2  = CenteredDifference(2,ord,Δx,N,:periodic,:periodic);    # 2nd Derivative

  u0 = cos.(x)
  du_true = -cos.(x)

  M = Matrix(Tridiagonal([1.0 for i in 1:N+1],[-2.0 for i in 1:N+2],[1.0 for i in 1:N+1]))
  # Do the reflections, different for x and y operators
  M[end,1] = 1.0
  M[1,end] = 1.0
  M = M/(Δx^2)
  @test M*u0 ≈ D2*u0

  A = zeros(size(M))
  for i in 1:N+2, j in 1:N+2
      i == j && (A[i,j] = -0.5)
      abs(i - j) == 2 && (A[i,j] = 0.25)
  end
  A[end-1,1] = 0.25
  A[end,2] = 0.25
  A[1,end-1] = 0.25
  A[2,end] = 0.25
  A = A/(Δx^2)

  @test A*u0 ≈ D1*(D1*u0)

  κ(x) = 1.0
  tmp1 = similar(x)
  tmp2 = similar(x)
  function f(t,u,du)
      mul!(tmp1,  D1, u)
      @. tmp2 = κ(x)*tmp1
      mul!(du,D1,tmp2)
  end

  du = similar(u0)
  f(0,u0,du)
  du ≈ A*u0

  error = du_true - du
  du2 = D2*u0
  error2 = du_true - du2

  @test maximum(error) < 0.004
  @test maximum(error2) < 0.001
end
