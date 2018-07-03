using DiffEqOperators
using Test

N = 5
srand(0); A = rand(N,N); u = rand(N)
L = DiffEqArrayOperator(A)
a = 3.5
La = L * a

@test La * u ≈ (a*A) * u
@test lufact(La) \ u ≈ (a*A) \ u
@test opnorm(La) ≈ opnorm(a*A)
@test exp(La) ≈ exp(a*A)
@test La[2,3] ≈ A[2,3] # should this be La[2,3] == a*A[2,3]?

update_func = (_A,u,p,t) -> _A .= t * A
t = 3.0
Atmp = zeros(N,N)
Lt = DiffEqArrayOperator(Atmp, a, update_func)
@test Lt(u,nothing,t) ≈ (a*t*A) * u
