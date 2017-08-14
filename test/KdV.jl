using Base.Test
using OrdinaryDiffEq, Sundials

@testset "KdV equation (Single Solition)" begin
    N,M = 1000,10
    Δx = 1/(N-1)
    Δt = 1/(M-1)

    x = -10:Δx:10;
    ϕ(x,t) = (-1/2)*sech.((x-t)/2).^2 # solution of the single forward moving wave
    u0 = ϕ(x,0);
    oriu = zeros(x);
    du3 = zeros(x);
    temp = zeros(x);

    # A = DerivativeOperator{Float64}(1,4,Δx,length(x),:periodic,:periodic);
    A = UpwindOperator{Float64}(1,1,Δx,length(x),BitVector(length(x)),:None,:None);
    # C = DerivativeOperator{Float64}(3,4,Δx,length(x),:periodic,:periodic);
    C = UpwindOperator{Float64}(3,1,Δx,length(x),BitVector(length(x)),:None,:None);

    function KdV(t, u, du)
       C(t,u,du3)
       A(t,u,du)
       @. temp = -6*u*du - du3
       copy!(du,temp)
    end

    single_solition = ODEProblem(KdV, u0, (0.,5.));
    soln = solve(single_solition,CVODE_BDF(),dense=false,saveat=0.03,maxiters=10000);

    for t in 0:0.5:5
        @test_skip soln(t) ≈ ϕ(x,t) atol = 0.01;
    end
end

# Conduct interesting experiments by referring to http://lie.math.brocku.ca/~sanco/solitons/kdv_solitons.php
@testset "KdV equation (Double Solition)" begin
    x = collect(-50 : 1/99 : 50);
    c1,c2 = 20,10

    # ϕ1(x,t) = 3*p1^2*sech(.5*(p1*(x+2+t))).^2+3*p2^2*sech(.5*(p2*(x+1+t))).^2;

    function ϕ(x,t)
        # t = t-10
        num1 = 2(c1-c2)*(c1*cosh.(√c2*(x-c2*t)/2).^2 + c2*sinh.(√c1*(x-c1*t)/2).^2)
        den11 = (√c1-√c2)*cosh.((√c1*(x-c1*t) + √c2*(x-c2*t))/2)
        den12 = (√c1+√c2)*cosh.((√c1*(x-c1*t) - √c2*(x-c2*t))/2)
        den1 = (den11+den12).^2
        return num1./den1
    end

    du3 = zeros(x);
    temp = zeros(x);
    A = DerivativeOperator{Float64}(1,2,1/99,length(x),:None,:None);
    C = DerivativeOperator{Float64}(3,2,1/99,length(x),:None,:None);

    function KdV(t, u, du)
       C(t,u,du3)
       A(t, u, du)
       @. temp = -6*u*du - du3
       copy!(du,temp)
    end

    u0 = ϕ(x,-5);
    double_solition = ODEProblem(KdV, u0, (-5.,5.));
    soln = solve(double_solition,CVODE_BDF(),dense=false,tstops=-5:0.1:5);

    # The solution is a forward moving soliton wave with speed = 1
    for t in 0:0.1:9
        @test_skip soln(t) ≈ ϕ(x,t)
    end
end
