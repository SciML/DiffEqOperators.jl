
@testset "FD DO operators comparison" begin
    x=0.0:0.01:1.
    dx=diff(x)

    for d_order in 1:4, approx_order in [2,4,8], LBC in (:None,:Dirichlet0, :Dirichlet), RBC in (:None,:Dirichlet0, :Dirichlet)

        D1 = DerivativeOperator{Float64}(d_order,approx_order,dx[1],length(x),LBC,RBC)
        D2 = DiffEqOperators.FiniteDifference{Float64}(d_order,approx_order,dx,length(x),LBC,RBC)

        @test convert(Array, D1) ≈ convert(Array, D2)
    end

end
