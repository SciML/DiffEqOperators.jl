using DiffEqBase, DiffEqOperators, LinearAlgebra, Zygote, Test
const A = rand(Float32, 300, 300)
f(du, u, p, t) = mul!(du, A, u)
f(u, p, t) = A * u
x = rand(Float32, 300)
v = rand(Float32, 300)
du = similar(x)

J = VecJacOperator(f, x)
actual_vjp = Zygote.jacobian(x -> f(x, nothing, nothing), x)[1]' * v
@test J * v ≈ actual_vjp
J = VecJacOperator(f, x; autodiff = false)
@test J * v ≈ actual_vjp