using Test
using DiffEqOperators, OrdinaryDiffEq

@testset "KdV equation (Single Solition)" begin
    N = 21
    local Δx = 1/(N-1)
    r = 0.5

    # x = 10:Δx:30;
    x = -10:Δx:10;
    # ϕ(x,t) = (r/2)*sech.((sqrt(r)*(x-r*t)/2)-7).^2 # solution of the single forward moving wave
    ϕ(x,t) = (1/2)*sech.((x-t)/2).^2 # solution of the single forward moving wave

    local u0 = ϕ(x,0);
    oriu = zeros(size(x));

    const du3 = zeros(size(x));
    const temp = zeros(size(x));

    # A = DerivativeOperator{Float64}(1,2,Δx,length(x),:Dirichlet0,:Dirichlet0);
    local A = UpwindOperator{Float64}(1,3,Δx,length(x),true.|BitVector(undef,length(x)),
                                :Dirichlet0,:Dirichlet0);
    # C = DerivativeOperator{Float64}(3,2,Δx,length(x),:Dirichlet0,:Dirichlet0);
    C = UpwindOperator{Float64}(3,3,Δx,length(x),true.|BitVector(undef,length(x)),
                                :Dirichlet0,:Dirichlet0);

    function KdV(du, u, p, t)
       C(t,u,du3)
       A(t,u,du)
       @. temp = -0.5*u*du - 0.25*du3
       copyto!(du,temp)
    end

    single_solition = ODEProblem(KdV, u0, (0.,5.));
    soln = solve(single_solition,Tsit5(),abstol=1e-6,reltol=1e-6);

    #=
    for t in 0:0.5:5
        @show maximum(soln(t)-ϕ(x,t))
    end
    =#

    for t in 0:0.5:5
        @test_skip soln(t) ≈ ϕ(x,t) atol = 0.01;
    end
end

# Conduct interesting experiments by referring to
# http://lie.math.brocku.ca/~sanco/solitons/kdv_solitons.php
@testset "KdV equation (Double Solition)" begin
    N = 10
    Δx = 1/(N-1)

    x = -50:Δx:50;
    c1,c2 = 25,16
    # ϕ1(x,t) = 3*p1^2*sech(.5*(p1*(x+2+t))).^2+3*p2^2*sech(.5*(p2*(x+1+t))).^2;

    function ϕ(x,t)
        # t = t-10
        num1 = 2(c1-c2)*(c1*cosh.(√c2*(x-c2*t)/2).^2 + c2*sinh.(√c1*(x-c1*t)/2).^2)
        den11 = (√c1-√c2)*cosh.((√c1*(x-c1*t) + √c2*(x-c2*t))/2)
        den12 = (√c1+√c2)*cosh.((√c1*(x-c1*t) - √c2*(x-c2*t))/2)
        den1 = (den11+den12).^2
        return num1./den1
    end

    const du3 = zeros(x);
    const temp = zeros(x);

    A = UpwindOperator{Float64}(1,1,Δx,length(x),false.*BitVector(length(x)),
                                :Dirichlet0,:nothing);
    C = UpwindOperator{Float64}(3,1,Δx,length(x),false.*BitVector(length(x)),
                                :Dirichlet0,:nothing);

    function KdV(du, u, p, t)
       C(t,u,du3)
       A(t, u, du)
       @. temp = -6*u*du - du3
       copy!(du,temp)
    end

    u0 = ϕ(x,0);
    double_solition = ODEProblem(KdV, u0, (0.,5.));
    soln = solve(double_solition,Tsit5());

    # The solution is a forward moving soliton wave with speed = 1
    for t in 0:0.1:9
        @test_skip soln(t) ≈ ϕ(x,t)
    end
end
