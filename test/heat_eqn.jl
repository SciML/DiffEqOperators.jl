using Test, DiffEqOperators
using OrdinaryDiffEq

@testset "Parabolic Heat Equation with Dirichlet BCs" begin
    x = collect(-pi : 2pi/511 : pi)
    u0 = @. -(x - 0.5).^2 + 1/12
    @. u_analytic(x)  = -(x .- 0.5).^2 + 1/12
    A = CenteredDifference(2,2,2π/511,512);
    bc = DirichletBC(u_analytic(-pi - 2pi/511),u_analytic(pi + 2pi/511))
    step = (u,p,t) ->A*bc*u
    heat_eqn = ODEProblem(step, u0, (0.,10.))
    soln = solve(heat_eqn,Tsit5(),dense=false,tstops=0:0.01:10)

    for t in 0:0.1:10
        @test soln(t)[1] ≈ u0[1] rtol=0.05
        @test soln(t)[end] ≈ u0[end] rtol=0.05
    end

    # UpwindDifference with equal no. of primay wind and offside points should behave like a CenteredDifference
    A2 = UpwindDifference(2,1,2π/511,512,1,offside=1);
    step = (u,p,t) ->A2*bc*u
    heat_eqn = ODEProblem(step, u0, (0.,10.))
    soln = solve(heat_eqn,Tsit5(),dense=false,tstops=0:0.01:10)

    for t in 0:0.1:10
        @test soln(t)[1] ≈ u0[1] rtol=0.05
        @test soln(t)[end] ≈ u0[end] rtol=0.05
    end
end

@testset "Parabolic Heat Equation with Neumann BCs" begin
    N = 512
    dx = 2π/(N-1)
    x = collect(-pi : dx : pi)
    u0 = @. -(x - 0.5)^2 + 1/12
    B = CenteredDifference(1,2,dx,N-2)
    deriv_start, deriv_end = (B*u0)[1], (B*u0)[end]

    A = CenteredDifference(2,2,dx,N)
    bc = NeumannBC((deriv_start,deriv_end),dx,1)

    step = (u,p,t) ->A*bc*u
    heat_eqn = ODEProblem(step, u0, (0.,10.))
    soln = solve(heat_eqn,Tsit5(),dense=false,tstops=0:0.01:10)

    first_order_coeffs_start = [-11/6, 3.0, -3/2, 1/3] * (1/dx)
    first_order_coeffs_end = -reverse([-11/6, 3.0, -3/2, 1/3] * (1/dx))

    for t in 0:0.1:10
        @test sum(first_order_coeffs_start .* soln(t)[1:4]) ≈ deriv_start atol=1e-1
        @test sum(first_order_coeffs_end .* soln(t)[end-3:end]) ≈ deriv_end atol=1e-1
    end

    # UpwindDifference with equal no. of primay wind and offside points should behave like a CenteredDifference
    A2 = UpwindDifference(2,1,dx,N,1,offside=1)
    B2 = UpwindDifference(1,2,dx,N-2,1,offside=1)
    deriv_start, deriv_end = (B2*u0)[1], (B2*u0)[end]
    bc = NeumannBC((deriv_start,deriv_end),dx,1)

    step = (u,p,t) ->A2*bc*u
    heat_eqn = ODEProblem(step, u0, (0.,10.))
    soln = solve(heat_eqn,Tsit5(),dense=false,tstops=0:0.01:10)

    for t in 0:0.1:10
        @test sum(first_order_coeffs_start .* soln(t)[1:4]) ≈ deriv_start atol=1e-1
        @test sum(first_order_coeffs_end .* soln(t)[end-3:end]) ≈ deriv_end atol=1e-1
    end
    # Testing for 2 offside points against Standard Vector input
    B3 = UpwindDifference(1,4,dx,N-2,-1,offside=2)
    deriv_start, deriv_end = (-1*B3*u0)[1], (-1*B3*u0)[end]
    bc = NeumannBC((deriv_start,deriv_end),dx,1)

    step = (u,p,t) ->A2*bc*u
    heat_eqn = ODEProblem(step, u0, (0.,10.))
    soln = solve(heat_eqn,Tsit5(),dense=false,tstops=0:0.01:10)

    for t in 0:0.1:10
        @test sum(first_order_coeffs_start .* soln(t)[1:4]) ≈ deriv_start atol=1e-1
        @test sum(first_order_coeffs_end .* soln(t)[end-3:end]) ≈ deriv_end atol=1e-1
    end
end

@testset "Parabolic Heat Equation with Robin BCs" begin
    N = 512
    dx = 2π/(N-1)
    x = collect(-pi : dx : pi)
    u0 = @. -(x - 0.5)^2 + 1/12
    B = CenteredDifference(1,2,dx,N-2);
    deriv_start, deriv_end = (B*u0)[1], (B*u0)[end]
    params = [1.0,0.5]

    left_RBC = params[1]*u0[1] - params[2]*deriv_start
    right_RBC = params[1]*u0[end] + params[2]*deriv_end

    A = CenteredDifference(2,2,dx,N);
    bc = RobinBC((params[1],-params[2],left_RBC), (params[1],params[2],right_RBC),dx,1);
    step1(u,p,t)=A*bc*u
    heat_eqn = ODEProblem(step1, u0, (0.,10.));
    println("solve 1")
    soln = solve(heat_eqn,Tsit5());

    first_order_coeffs_start = [-11/6, 3.0, -3/2, 1/3] * (1/dx)
    first_order_coeffs_end = -reverse([-11/6, 3.0, -3/2, 1/3] * (1/dx))
    val = []

    for t in 0.2:0.1:9.8
        @test params[1]*soln(t)[1] - params[2]*sum(first_order_coeffs_start .* soln(t)[1:4]) ≈ left_RBC atol=1e-1
        # append!(val,params[1]*soln(t)[1] + -params[2]*sum(first_order_coeffs_start .* soln(t)[1:4]) - left_RBC)
        @test params[1]*soln(t)[end] + params[2]*sum(first_order_coeffs_end .* soln(t)[end-3:end]) ≈ right_RBC atol=1e-1
    end

    # UpwindDifference with equal no. of primay wind and offside points should behave like a CenteredDifference
    A2 = UpwindDifference(2,1,dx*ones(N+1),N,1,offside=1)
    B2 = UpwindDifference(1,2,dx*ones(N-1),N-2,1,offside=1)
    deriv_start, deriv_end = (B2*u0)[1], (B2*u0)[end]
    left_RBC = params[1]*u0[1] - params[2]*deriv_start
    right_RBC = params[1]*u0[end] + params[2]*deriv_end
    bc = RobinBC((params[1],-params[2],left_RBC), (params[1],params[2],right_RBC),dx,1);

    step2(u,p,t)=A2*bc*u
    heat_eqn = ODEProblem(step2, u0, (0.,1.));
    println("solve 2")
    soln = solve(heat_eqn,Tsit5());
    println("sol2 done!")

    for t in 0.2:0.1:1.0
        @test params[1]*soln(t)[1] - params[2]*sum(first_order_coeffs_start .* soln(t)[1:4]) ≈ left_RBC atol=1e-1
        @test params[1]*soln(t)[end] + params[2]*sum(first_order_coeffs_end .* soln(t)[end-3:end]) ≈ right_RBC atol=1e-1
    end
    # Testing for 2 offside points against Standard Vector input
    B3 = UpwindDifference(1,4,dx*ones(N-1),N-2,-1,offside=2)
    deriv_start, deriv_end = (-1*B3*u0)[1], (-1*B3*u0)[end]
    left_RBC = params[1]*u0[1] - params[2]*deriv_start
    right_RBC = params[1]*u0[end] + params[2]*deriv_end
    bc = RobinBC((params[1],-params[2],left_RBC), (params[1],params[2],right_RBC),dx,1);

    step3(u,p,t)=A2*bc*u
    heat_eqn = ODEProblem(step3, u0, (0.,1.));
    println("solve 3")
    soln = solve(heat_eqn,Tsit5());

    for t in 0.2:0.1:1.0
        @test params[1]*soln(t)[1] - params[2]*sum(first_order_coeffs_start .* soln(t)[1:4]) ≈ left_RBC atol=1e-1
        @test params[1]*soln(t)[end] + params[2]*sum(first_order_coeffs_end .* soln(t)[end-3:end]) ≈ right_RBC atol=1e-1
    end
end
