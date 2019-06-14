using DiffEqOperators

n = 100
x=0.0:0.01:2π
xprime = x[2:(end-1)]
dx=diff(x)
y = exp.(π*x)
y_im = exp.(π*im*x)
yim_ = y_im[2:(end-1)]
y_ = y[2:(end-1)]

@test_broken for dor in 1:6, aor in 1:8

    D1 = DerivativeOperator{Float64}(dor,aor,dx[1],length(x))
    D2 = DiffEqOperators.FiniteDifference{Float64}(dor,aor,dx,length(x))
    D = (D1,D2)
    @test_broken convert(Array, D1) ≈ convert(Array, D2)

    #take derivatives
    yprime1 = D1*y
    yprime2 = D2*y

    #test result
    @test_broken yprime1 ≈ (π^dor)*y_ # test operator with known derivative of exp(kx)
    @test_broken yprime2 ≈ (π^dor)*y_ # test operator with known derivative of exp(kx)

    #test equivalance
    @test_broken yprime1 ≈ yprime2

    #take derivatives
    y_imprime1 = D1*y_im
    y_imprime2 = D2*y_im

    #test result
    @test_broken y_imprime1 ≈ ((pi*im)^dor)*yim_ # test operator with known derivative of exp(jkx)
    @test_broken y_imprime2 ≈ ((pi*im)^dor)*yim_ # test operator with known derivative of exp(jkx)

    #test equivalance
    @test_broken y_imprime1 ≈ y_imprime2

    #TODO: implement specific tests for the left and right boundary regions, waiting until after update
end
