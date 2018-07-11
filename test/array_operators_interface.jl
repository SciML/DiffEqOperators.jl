using DiffEqOperators, Random, LinearAlgebra
using Test

N = 5
srand(0); A = rand(N,N); u = rand(N)
L = DiffEqArrayOperator(A)

@test L * u ≈ A * u
@test lu(L) \ u ≈ A \ u
@test opnorm(L) ≈ opnorm(A)
@test exp(L) ≈ exp(A)
@test L[2,3] == A[2,3]

update_func = (_A,u,p,t) -> _A .= t * A
t = 3.0
Atmp = zeros(N,N)
Lt = DiffEqArrayOperator(Atmp; update_func=update_func)
@test Lt(u,nothing,t) ≈ (t*A) * u
