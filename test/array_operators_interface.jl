using DiffEqOperators
using Base.Test

N = 5
srand(0); A = rand(N,N); u = rand(N)
L = DiffEqArrayOperator(A)
a = 3.5
La = L * a

@test La * u ≈ (a*A) * u
@test lufact(La) \ u ≈ (a*A) \ u
@test norm(La) ≈ norm(a*A)
@test expm(La) ≈ expm(a*A)
@test La[2,3] ≈ A[2,3] # should this be La[2,3] == a*A[2,3]?

update_func = (_A,t,u) -> _A .= t * A
t = 3.0
Atmp = zeros(N,N)
Lt = DiffEqArrayOperator(Atmp, a, update_func) 
@test Lt(u,nothing,t) ≈ (a*t*A) * u
