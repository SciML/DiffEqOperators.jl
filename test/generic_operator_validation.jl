using DiffEqOperators


x=0.0:0.005:2π
xprime = x[2:(end-1)]
dx=x[2]-x[1]
y = exp.(π*x)
y_im = exp.(π*im*x)
yim_ = y_im[2:(end-1)]
y_ = y[2:(end-1)]

@test_broken for dor in 1:4, aor in 2:6

    D1 = CenteredDifference(dor,aor,dx,length(x))

    #take derivative
    yprime1 = D1*y

    #test result
    @test_broken yprime1 ≈ (π^dor)*y_ # test operator with known derivative of exp(kx)

    #take derivatives
    y_imprime1 = D1*y_im

    #test result
    @test_broken y_imprime1 ≈ ((pi*im)^dor)*yim_ # test operator with known derivative of exp(jkx)
end
