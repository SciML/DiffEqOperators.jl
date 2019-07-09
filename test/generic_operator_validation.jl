using DiffEqOperators

n = 100
x=0.0:0.01:2π
xprime = x[2:(end-1)]
dx=diff(x)
y = exp.(π*x)
y_im = exp.(π*im*x)
yim_ = y_im[2:(end-1)]
y_ = y[2:(end-1)]

for dor in 1:6, aor in 1:8

    D1 = CenteredDifference(dor,aor,dx,length(x))

    #take derivative
    yprime1 = D1*y

    #test result
    @test yprime1 ≈ (π^dor)*y_ # test operator with known derivative of exp(kx)

    #take derivatives
    y_imprime1 = D1*y_im

    #test result
    @test y_imprime1 ≈ ((pi*im)^dor)*yim_ # test operator with known derivative of exp(jkx)


    #TODO: implement specific tests for the left and right boundary regions, waiting until after update
end
