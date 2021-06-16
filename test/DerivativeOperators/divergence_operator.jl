using DiffEqOperators, Test

@testset "Divergence operation for a 3D Vector" begin
    
    s = x, y, z = (-5:1.25:5, -5:1.25:5, -5:1.25:5)
    dx = dy = dz = x[2] - x[1]
    
    # Vector u0 = (x^2 + y^2) ê₁ + (y^2 + z^2) ê₂  + (x^2 + z^2) ê₃
    
    u0 = Array{Array{Float64,1},3}(undef,length(x),length(y),length(z))
    for I in CartesianIndices(u0)
        u0[I] = zeros(Float64,3)
        u0[I][1] = x[I[1]]^2 + y[I[2]]^2
        u0[I][2] = y[I[2]]^2 + z[I[3]]^2
        u0[I][3] = x[I[1]]^2 + z[I[3]]^2
    end
    
    # Analytic Divergence of the given vector given by u_analytic = 2x + 2y + 2z
    
    u_analytic = Array{Float64}(undef,size(u0).-2)
    for I in CartesianIndices(u_analytic)
        u_analytic[I] = 2*x[I[1]+1] + 2*y[I[2]+1] + 2*z[I[3]+1]
    end
    
    A = Divergence(4,(dx,dy,dz),size(u0).-2)
    
    u = A*u0
    
    for I in CartesianIndices(u)
        @test u[I] ≈ u_analytic[I] atol=1e-3
    end    
end