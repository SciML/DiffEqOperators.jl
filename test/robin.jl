using LinearAlgebra, DiffEqOperators, Random, Test

# Generate random parameters
al = rand(5)
bl = rand(5)
cl = rand(5)
dx_l = rand(5)
ar = rand(5)
br = rand(5)
cr = rand(5)
dx_r = rand(5)

# Construct 5 arbitrary RobinBC operators
for i in 1:5
    Q = RobinBC(al[i], bl[i], cl[i], dx_l[i], ar[i], br[i], cr[i], dx_r[i])

    Q_L, Q_b = Array(Q,5i)

    #Check that Q_L is is correctly computed
    @test Q_L[2:5i+1,1:5i] ≈ Array(I, 5i, 5i)
    @test Q_L[1,:] ≈ [1 / (1-al[i]*dx_l[i]/bl[i]); zeros(5i-1)]
    @test Q_L[5i+2,:] ≈ [zeros(5i-1); 1 / (1+ar[i]*dx_r[i]/br[i])]


    #Check that Q_b is computed correctly
    @test Q_b ≈ [cl[i]/(al[i]-bl[i]/dx_l[i]); zeros(5i); cr[i]/(ar[i]+br[i]/dx_r[i])]

    # Construct the extended operator and check that it correctly extends u to a (5i+2)
    # vector, along with encoding boundary condition information.
    u = rand(5i)

    Qextended = Q*u
    CorrectQextended = [(cl[i]-(bl[i]/dx_l[i])*u[1])/(al[i]-bl[i]/dx_l[i]); u; (cr[i]+ (br[i]/dx_r[i])*u[5i])/(ar[i]+br[i]/dx_r[i])]

    @test length(Qextended) ≈ 5i+2
    @test Qextended ≈ CorrectQextended

    # Check concretization
    @test Array(Qextended) ≈ CorrectQextended

    # Check that Q_L and Q_b correctly compute RobinBCExtended
    @test Q_L*u + Q_b ≈ CorrectQextended

    @test [Qextended[1]; Qextended.u; Qextended[5i+2]] ≈ CorrectQextended

end

#TODO: Implement tests for BC's that are contingent on the sign of the coefficient on the boundary near the boundary
