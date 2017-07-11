using Base.Test
using FactCheck
using SpecialMatrices
using DifferentialEquations

context("KdV equation (Single Solition)")do
    x = collect(-10 : 1/99 : 10);
    ϕ(x,t) = (-1/2)*sech.((x+t)/2).^2 # solution of the single backward moving wave
    u0 = ϕ(x,0)
    du3 = zeros(x);
    A = LinearOperator{Float64}(1,2,1/99,length(x),:None,:None);
    C = LinearOperator{Float64}(3,2,1/99,length(x),:None,:None);

    function KdV(t, u, du)
       C(t,u,du3)
       A(t, u, du)
       temp = -6*u.*du .+ du3
       copy!(du,temp)
    end

    single_solition = ODEProblem(A, u0, (0.,5.));
    soln = solve(single_solition, dense=false, tstops=0:0.01:5);

    for t in 0:0.5:5
        @test soln(t) ≈ ϕ(x,t) atol = 0.01;
    end
end


context("KdV equation (Double Solition)")do
    x = collect(-10 : 1/99 : 10);
    p1,p2 = 25.16
    ϕ(x,t) = 3*p1^2*sech(.5*(p1*(x+2+t))).^2+3*p2^2*sech(.5*(p2*(x+1+t))).^2;
    u0 = ϕ(x,0)
    du3 = zeros(x);
    A = LinearOperator{Float64}(1,2,1/99,length(x),:None,:None);
    C = LinearOperator{Float64}(3,2,1/99,length(x),:None,:None);

    function KdV(t, u, du)
           C(t,u,du3)
           A(t, u, du)
           temp = -u.*du .- du3
           copy!(du,temp)
    end

    double_solition = ODEProblem(A, u0, (0.,10.));
    soln = solve(double_solition, dense=false, tstops=0:0.01:10);

    # The solution is a forward moving solition wave with speed = 1
    for t in 0:0.1:9
        @test soln(t) ≈ ϕ(x,t)
    end
end
