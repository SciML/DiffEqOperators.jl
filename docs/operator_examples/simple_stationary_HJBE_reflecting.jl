using ToeplitzMatrices, Plots
gr()
n = 100;
x = linspace(0.0, 1.0, n)
Δx = step(x)


B = [-1 1 zeros(1,n-4)  0 0;
      0 0 zeros(1,n-4) -1 1]

Q = [zeros(n-2)' ;I; zeros(n-2)']
Q[1] = 1
Q[end] = 1
R = [zeros(n-2) I zeros(n-2)]

u = exp.(x)


Δ = Toeplitz([1; zeros(n-3)], [1; -2; 1; zeros(n-3)])/Δx^2
A = Δ - Δ[:,[1,n]]*(B[:,[1,n]]\B)

#Now solve the stationary ODE r u(x) = x^alpha + \sigma * D_xx u(x)
alpha = 0.5
r = 0.05
sigma = 0.1

B = r * I - sigma^2 * A * Q
u = B \ x[2:end-1].^alpha

# Double check the operators
R*Q*u == u # true

plot(x, Q*u)
