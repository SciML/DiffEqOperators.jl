using LinearAlgebra, DiffEqOperators, Random, Test

A = rand(100,100)
A[1,1] = A[end,1] = A[1,end] = A[end,end] = 0.0
lower = Vector(Vector{Float64}, A[1,2:(end-1)], transpose(A[2:(end-1),1]))
upper = Vector(Vector{Float64}, A[end,2:(end-1)], transpose(A[2:(end-1),end]))

Apad = DiffEqOperators.BoundaryPaddedMatrix{Float64, typeof(A), typeof(lower[1])}(lower, upper, A[2:(end-1), 2:(end-1)])

@test A == Array(Apad)
