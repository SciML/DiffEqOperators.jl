
@testset "FD DO operators comparison" begin
    n = 100
    x=0.0:0.01:2π
    dx=diff(x)
    y = sin.(x);


    for dor in 1:4, aor in 2:2:8,
            LBC in (:None,:Dirichlet0, :Dirichlet, :Neumann0, :Neumann, :periodic),
            RBC in (:None,:Dirichlet0, :Dirichlet, :Neumann0, :periodic)

        D1 = DerivativeOperator{Float64}(dor,aor,dx[1],length(x),:None,:None)
        D2 = DiffEqOperators.FiniteDifference{Float64}(dor,aor,dx,length(x),:None,:None)

        @test full(D1) ≈ full(D2)
    end

end
