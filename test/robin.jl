using LinearAlgebra, DiffEqOperators, Random, Test

# Generate random parameters
al = rand(5)
bl = rand(5)
cl = rand(5)
dx = rand(5)
ar = rand(5)
br = rand(5)
cr = rand(5)


# Construct 5 arbitrary RobinBC operators
for i in 1:5
    Q = RobinBC((al[i], bl[i], cl[i]), (ar[i], br[i], cr[i]), dx[i])

    Q_L, Q_b = Array(Q,5i)

    #Check that Q_L is is correctly computed
    @test Q_L[2:5i+1,1:5i] ≈ Array(I, 5i, 5i)
    @test Q_L[1,:] ≈ [1 / (1-al[i]*dx[i]/bl[i]); zeros(5i-1)]
    @test Q_L[5i+2,:] ≈ [zeros(5i-1); 1 / (1+ar[i]*dx[i]/br[i])]

    #Check that Q_b is computed correctly
    @test Q_b ≈ [cl[i]/(al[i]-bl[i]/dx[i]); zeros(5i); cr[i]/(ar[i]+br[i]/dx[i])]

    # Construct the extended operator and check that it correctly extends u to a (5i+2)
    # vector, along with encoding boundary condition information.
    u = rand(5i)

    Qextended = Q*u
    CorrectQextended = [(cl[i]-(bl[i]/dx[i])*u[1])/(al[i]-bl[i]/dx[i]); u; (cr[i]+ (br[i]/dx[i])*u[5i])/(ar[i]+br[i]/dx[i])]
    @test length(Qextended) ≈ 5i+2

    # Check concretization
    @test Array(Qextended) ≈ CorrectQextended

    # Check that Q_L and Q_b correctly compute BoundaryPaddedVector
    @test Q_L*u + Q_b ≈ CorrectQextended

    @test [Qextended[1]; Qextended.u; Qextended[5i+2]] ≈ CorrectQextended

end

#3rd order RobinBC, calculated with left stencil [-11/6 3 -3/2 1/3], right stencil [-1/3 3/2 -3 11/6] and [α,β,γ] = [1 6 10]
u0 = -4/10
uend = 125/12
u = Vector(1.0:10.0)
Q = RobinBC((1.0, 6.0, 10.0), (1.0, 6.0, 10.0), 1.0, 3)
urobinextended = Q*u
@test urobinextended.l ≈ u0
@test urobinextended.r ≈ uend
# General BC should be equivalent
G = GeneralBC([-10.0, 1.0, 6.0], [-10.0, 1.0, 6.0], 1.0, 3)
ugeneralextended = G*u
@test ugeneralextended.l ≈ u0
@test ugeneralextended.r ≈ uend


#TODO: Implement tests for BC's that are contingent on the sign of the coefficient on the operator near the boundary
