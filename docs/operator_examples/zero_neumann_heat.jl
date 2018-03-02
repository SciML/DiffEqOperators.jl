using OrdinaryDiffEq, ToeplitzMatrices, Plots
n = 100;
x = linspace(-1,1,n)
Δx = step(x)

B = [-1 1 zeros(1,n-4)  0 0;
      0 0 zeros(1,n-4) -1 1]/Δx

Δ = Toeplitz([1; zeros(n-3)], [1; -2; 1; zeros(n-3)])/Δx^2
Ã = Δ - Δ[:,[1,n]]*(B[:,[1,n]]\B)
A = Ã[:,2:n-1]

function heat(du,u,A,t)
    du .=  A*u
end
e
u₀ = @. exp(x) + ((-2 + x)*x)/(4e) - e*x*(2 + x)/4 # satisfies zero Neumann

prob = ODEProblem(heat, u₀[2:n-1], (0.0,8.0), A)
@time ũ  = solve(prob, Rodas4(autodiff=false); reltol=1e-5,abstol=1e-5)

u = [B; eye(n)[2:n-1,:]] \ [0; 0; ũ(8.0)]

plot(x, u) # ≈ mean(u₀)
