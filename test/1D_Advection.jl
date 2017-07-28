using Base.Test
using FactCheck
using DifferentialEquations

context("KdV equation (Single Solition)")do
    N,M = 100,5000
    Δx = 1/(N-1)
    Δt = 1/(M-1)
    a = 1

    x = collect(-π : Δx : π)
    ϕ(x,t,a) = sin.((x-a*t)) # solution of the single forward moving wave, a is the speed of wave
    u0 = ϕ(x,0,a);
    oriu = zeros(x);

    A = UpwindOperator{Float64}(1,2,Δx,length(x),.~BitVector(length(x)),:Dirichlet0,:nothing);

    function advection(t, u, du)
       A(t,u,du)
       # scale!(du,-1)
    end

    single_solition = ODEProblem(advection, u0, (0.,5.));
    soln = solve(single_solition,SSPRK22(),dense=false,saveat=0.001,dt=Δt);

    for t in 0:0.5:5
        @test soln(t) ≈ ϕ(x,t,a) atol = 0.01;
    end
end
