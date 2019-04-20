using DiffEqOperators, ForwardDiff, LinearAlgebra, Test
const A = rand(300,300)
f(du,u) = mul!(du,A,u)
f(u) = A*u
x = rand(300)
v = rand(300)
du = similar(x)

cache1 = ForwardDiff.Dual{DiffEqOperators.JacVecTag}.(x, v)
cache2 = ForwardDiff.Dual{DiffEqOperators.JacVecTag}.(x, v)
@test DiffEqOperators.jacvec!(du, f, x, v) ≈ ForwardDiff.jacobian(f,similar(x),x)*v
@test DiffEqOperators.jacvec!(du, f, x, v, cache1, cache2) ≈ ForwardDiff.jacobian(f,similar(x),x)*v
@test DiffEqOperators.jacvec(f, x, v) ≈ ForwardDiff.jacobian(f,similar(x),x)*v

f(du,u,p,t) = mul!(du,A,u)
f(u,p,t) = A*u
L = JacVecOperator(f,x)
@test L*x ≈ DiffEqOperators.jacvec(f, x, x)
@test L*v ≈ DiffEqOperators.jacvec(f, v, v)
@test mul!(du,L,x) ≈ DiffEqOperators.jacvec(f, x, x)

L2 = JacVecOperator{Float64}(f)
@test L2*x ≈ DiffEqOperators.jacvec(f, x, x)
@test L2*v ≈ DiffEqOperators.jacvec(f, v, v)
@test mul!(du,L2,x) ≈ DiffEqOperators.jacvec(f, x, x)

using OrdinaryDiffEq
function lorenz(du,u,p,t)
 du[1] = 10.0(u[2]-u[1])
 du[2] = u[1]*(28.0-u[3]) - u[2]
 du[3] = u[1]*u[2] - (8/3)*u[3]
end
u0 = [1.0;0.0;0.0]
tspan = (0.0,100.0)
ff = ODEFunction(lorenz,jac_prototype=JacVecOperator{Float64}(lorenz,u0))
prob = ODEProblem(ff,u0,tspan)
sol = solve(prob,Rosenbrock23())
sol = solve(prob,Rosenbrock23(linsolve=LinSolveGMRES(tol=1e-10)))
