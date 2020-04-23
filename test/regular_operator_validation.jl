using DiffEqOperators


nx = 200
x = range(0,2pi,length=nx)
dx = x[2]-x[1]
y = sin.(x)
dy = [cos.(x),-sin.(x),-cos.(x),y,cos.(x),-sin.(x)]

for dor in 1:6, aor in 2:2:10

    D1 = CenteredDifference(dor,aor,dx,nx-2)
    #take derivative
    dyt = D1*y
    #test result
    @test dy[dor][2:end-1] ≈ dyt atol=1.0^(1-aor)

    #TODO: implement specific tests for the left and right boundary regions, waiting until after update
end
