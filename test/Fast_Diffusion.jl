# # Fast-Diffusion Problem
#
# This example demonstrates the use of 'NonLinearDiffusion' operator to solve time-dependent non-linear diffusion PDEs with coefficient having dependence on the unknown function. 
# Here we consider a fast diffusion problem with Dirichlet BCs on unit interval:
# ∂ₜu = ∂ₓ(k*∂ₓu) 
# k = 1/u²   
# u(x=0,t) = exp(-t)
# u(x=1,t) = 1/(1.0 + exp(2t))
# u(x, t=0) = u₀(x)
using Test
using DiffEqOperators, OrdinaryDiffEq

@testset "1D Nonlinear fast diffusion equation" begin
    # The analytical solution for this is given by :

    u_analytic(x, t) = 1 / sqrt(x^2 + exp(2*t))

    #
    # Reproducing it numerically 
    #


    nknots = 100
    h = 1.0/(nknots+1)
    knots = range(h, step=h, length=nknots)
    n = 1                                   # Outer differential order
    m = 1                                   # Inner differential order
    approx_ord = 2                               

    u0 = u_analytic.(knots,0.0)
    du = similar(u0)
    t0 = 0.0
    t1 = 1.0

    function f(du,u,p,t)
        bc = DirichletBC(exp(-t),(1.0 + exp(2*t))^(-0.5))
        l = bc*u
        k = l.^(-2)                        # Diffusion Coefficient
        nonlinear_diffusion!(du,n,m,approx_ord,k,l,h,nknots)
    end

    prob = ODEProblem(f, u0, (t0, t1))
    alg = KenCarp4()
    sol = solve(prob,alg)

    for t in t0:0.1:t1
        @test u_analytic.(knots, t) ≈ sol(t) rtol=1e-3
    end

end