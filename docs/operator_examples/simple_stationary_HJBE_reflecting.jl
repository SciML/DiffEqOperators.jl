using ToeplitzMatrices
K = 5 #98
K̂ = K + 2
#x = linspace(0.0, 1.0, K̂)
x = linspace(0.0, 1.0, K)
Δx = step(x)
x̂ = [x[1]-Δx; x; x[end]+Δx] #Ended domain

B = [-1 1 zeros(1,K);
      zeros(1,K) -1 1]

Q = [zeros(K)' ;I; zeros(K)']
Q[1] = 1
Q[end] = 1
R = [zeros(K) I zeros(K)]


A = Toeplitz([1; zeros(K̂-3)], [1; -2; 1; zeros(K̂-3)])/Δx^2
#A2 = A - A[:,[1,K̂]]*(B[:,[1,K̂]]\B) #We think this is useless?

#Now solve the stationary ODE r u(x) = x^alpha + \sigma * D_xx u(x)
α = 0.5
r = 0.05
σ = 0.1
b(x) = x^α

B = r * I - σ^2 * A * Q

u = B \ b.(x)

# Double check the operators
R*Q*u == u # true



using Plots
#plot(x̂, Q*u)
plot(x, u)
