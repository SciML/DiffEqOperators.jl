using OrdinaryDiffEq, ToeplitzMatrices, Plots
## Non-zero Dirichlet

n = 100;
x = linspace(-1,1,n)
Δx = step(x)

B = [1 0 zeros(1,n-4)  0 0;
      0 0 zeros(1,n-4) 0 1]/Δx

Δ = Toeplitz([1; zeros(n-3)], [1; -2; 1; zeros(n-3)])/Δx^2
Ã = Δ - Δ[:,[1,n]]*(B[:,[1,n]]\B)
A = Ã[:,2:n-1]

u₀ = @. exp(x)
r = Δ[:,[1,n]]*(B[:,[1,n]]\(B*u₀))

p = (A,r)


function heat(du,u,p,t)
    (A,r) = p
    du .=  A*u .+ r
end

prob = ODEProblem(heat, u₀[2:n-1], (0.0,8.0), p)
@time ũ  = solve(prob, Rodas4(autodiff=false); reltol=1e-5,abstol=1e-5)
u_r = [B; eye(n)[2:n-1,:]] \ [B*u₀;  ũ(8.0)]
plot(x, u_r; label="Rodas4") # ≈ mean(u₀)


@time ũ  = solve(prob)
u_s = [B; eye(n)[2:n-1,:]] \ [B*u₀;  ũ(8.0)]
plot!(x, u_s; label="solve") # ≈ mean(u₀)

norm(u_r-u_s) # < 0.0009
