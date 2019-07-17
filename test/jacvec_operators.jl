using DiffEqBase, DiffEqOperators, ForwardDiff, LinearAlgebra, Test
const A = rand(300,300)
f(du,u) = mul!(du,A,u)
f(u) = A*u
x = rand(300)
v = rand(300)
du = similar(x)

cache1 = ForwardDiff.Dual{DiffEqOperators.JacVecTag}.(x, v)
cache2 = ForwardDiff.Dual{DiffEqOperators.JacVecTag}.(x, v)
@test DiffEqOperators.num_jacvec!(du, f, x, v) ≈ ForwardDiff.jacobian(f,similar(x),x)*v rtol=1e-6
@test DiffEqOperators.num_jacvec!(du, f, x, v, similar(v), similar(v)) ≈ ForwardDiff.jacobian(f,similar(x),x)*v rtol=1e-6
@test DiffEqOperators.num_jacvec(f, x, v) ≈ ForwardDiff.jacobian(f,similar(x),x)*v rtol=1e-6

@test DiffEqOperators.auto_jacvec!(du, f, x, v) ≈ ForwardDiff.jacobian(f,similar(x),x)*v
@test DiffEqOperators.auto_jacvec!(du, f, x, v, cache1, cache2) ≈ ForwardDiff.jacobian(f,similar(x),x)*v
@test DiffEqOperators.auto_jacvec(f, x, v) ≈ ForwardDiff.jacobian(f,similar(x),x)*v

f(du,u,p,t) = mul!(du,A,u)
f(u,p,t) = A*u
L = JacVecOperator(f,x)
@test L*x ≈ DiffEqOperators.auto_jacvec(f, x, x)
@test L*v ≈ DiffEqOperators.auto_jacvec(f, x, v)
@test mul!(du,L,v) ≈ DiffEqOperators.auto_jacvec(f, x, v)
DiffEqBase.update_coefficients!(L,v,nothing,nothing)
@test mul!(du,L,v) ≈ DiffEqOperators.auto_jacvec(f, v, v)

L = JacVecOperator(f,x,autodiff=false)
DiffEqBase.update_coefficients!(L,x,nothing,nothing)
@test L*x ≈ DiffEqOperators.num_jacvec(f, x, x)
@test L*v ≈ DiffEqOperators.num_jacvec(f, x, v)
@test mul!(du,L,v) ≈ DiffEqOperators.num_jacvec(f, x, v) rtol=1e-6
DiffEqBase.update_coefficients!(L,v,nothing,nothing)
@test mul!(du,L,v) ≈ DiffEqOperators.num_jacvec(f, v, v) rtol=1e-6

L2 = JacVecOperator{Float64}(f)
DiffEqBase.update_coefficients!(L2,x,nothing,nothing)
@test L2*x ≈ DiffEqOperators.auto_jacvec(f, x, x)
@test L2*v ≈ DiffEqOperators.auto_jacvec(f, v, v)
@test mul!(du,L2,x) ≈ DiffEqOperators.auto_jacvec(f, x, x)

L2 = JacVecOperator{Float64}(f,autodiff=false)
DiffEqBase.update_coefficients!(L2,x,nothing,nothing)
@test L2*x ≈ DiffEqOperators.num_jacvec(f, x, x)
@test L2*v ≈ DiffEqOperators.num_jacvec(f, v, v) rtol=1e-6

using OrdinaryDiffEq
function lorenz(du,u,p,t)
 du[1] = 10.0(u[2]-u[1])
 du[2] = u[1]*(28.0-u[3]) - u[2]
 du[3] = u[1]*u[2] - (8/3)*u[3]
end
u0 = [1.0;0.0;0.0]
tspan = (0.0,100.0)
ff1 = ODEFunction(lorenz,jac_prototype=JacVecOperator{Float64}(lorenz,u0))
ff2 = ODEFunction(lorenz,jac_prototype=JacVecOperator{Float64}(lorenz,u0,autodiff=false))
for ff in [ff1, ff2]
  prob = ODEProblem(ff,u0,tspan)
  @test solve(prob,TRBDF2()).retcode == :Success
  @test_broken solve(prob,TRBDF2(linsolve=LinSolveGMRES(tol=1e-10))).retcode == :Success
  @test solve(prob,Exprb32()).retcode == :Success
  @test_broken sol = solve(prob,Rosenbrock23())
  @test_broken sol = solve(prob,Rosenbrock23(linsolve=LinSolveGMRES(tol=1e-10)))
end
