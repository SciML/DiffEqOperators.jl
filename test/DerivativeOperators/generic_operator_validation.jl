using DiffEqOperators, SparseArrays, LinearAlgebra

x = 0.0:0.01:π
x_ = x[2:end-1]
dx = diff(x)
y = sin.(2x)
y_ = y[2:(end-1)]
dy = [2cos.(2x_),-4sin.(2x_),-8cos.(2x_),16y_,32cos.(2x_)]



# test concretization against regular grid operator
for dor in 1:4, aor in 2:2:6
    Dr = CenteredDifference(dor,aor,dx[1],length(x)-2)
    Dir = CenteredDifference(dor,aor,dx,length(x)-2)

    @test sparse(Dr)≈sparse(Dir)
    @test Array(Dr)≈Array(Dir)

end

# test irregular grid operator with regular grid
for dor in 1:4, aor in 2:6

    D1 = CenteredDifference(dor,aor,dx,length(x)-2)

    #take derivative
    yprime1 = D1*y

    #test result
    @test yprime1 ≈ dy[dor] atol = 10.0^(1-aor)#2test operator with known derivative of exp(kx)

    #TODO: implement specific tests for the left and right boundary regions, waiting until after update
end


# test irregular grid

x = sin.(0.0:0.05:π)
x = cumsum(x)
x = x/x[end]*π
x_ = x[2:end-1]
dx = diff(x)
y  = sin.(2x)
y_ = y[2:(end-1)]
dy = [2cos.(2x_), -4sin.(2x_), -8cos.(2x_), 16y_, 32cos.(2x_)]

for dor in 1:4, aor in 4:10

    D1 = CenteredDifference(dor,aor,dx,length(x)-2)

    #take derivative
    yprime1 = D1*y

    #test result
    tol = 2*10.0^(2-aor)*maximum(dx)^(2-dor)
    # error estimate is fairly difficult for high order derivatives and small dx.

    # err = norm(yprime1.-dy[dor])
    # @show err
    # @show tol

    @test yprime1 ≈ dy[dor] atol = tol #2test operator with known derivative of exp(kx)

    #TODO: implement specific tests for the left and right boundary regions, waiting until after update
end
