using DiffEqOperators, Test

@testset "Curl operation for a 3D Vector" begin
    
    s = x, y, z = (-5:1.25:5, -5:1.25:5, -5:1.25:5)
    dx = dy = dz = x[2] - x[1]
    
    # Vector u0 = (y^2 + z^2) ê₁ + (x^2 + z^2) ê₂  + (x^2 + y^2) ê₃
    
    u0 = zeros(Float64,length(x),length(y),length(z),3)

    for k in 1:length(z), j in 1:length(y), i in 1:length(x)
        u0[i,j,k,1] = y[j]^2 + z[k]^2
        u0[i,j,k,2] = x[i]^2 + z[k]^2
        u0[i,j,k,3] = x[i]^2 + y[j]^2
    end
    
    # Analytic Curl of the given vector given by u_analytic = 2(y-z) ê₁ + 2(z-x) ê₂  + 2(x-y) ê₃
    u_analytic = zeros(Float64,length(x)-2,length(y)-2,length(z)-2,3)
    for k in 1:length(z)-2, j in 1:length(y)-2, i in 1:length(x)-2 
        u_analytic[i,j,k,1] = 2*(y[j+1] - z[k+1])
        u_analytic[i,j,k,2] = 2*(z[k+1] - x[i+1])
        u_analytic[i,j,k,3] = 2*(x[i+1] - y[j+1])
    end
    
    A = Curl(4,(dx,dy,dz),size(u0)[1:end-1].-2)
    
    u = A*u0
    
    for I in CartesianIndices(u)
        @test u[I] ≈ u_analytic[I] atol=1e-3
    end    
    
    # check for multiplication with constant

    u1 = 2*A*u0
    
    for I in CartesianIndices(u)
        @test u1[I] ≈ 2*u[I] atol=1e-3
    end
    
    # check for non-uniform grid
    
    dx = dy = dz = 1.25*ones(10)

    A = Curl(4,(dx,dy,dz),size(u0)[1:end-1].-2)
    
    u = A*u0
    
    for I in CartesianIndices(u)
        @test u[I] ≈ u_analytic[I] atol=1e-3
    end
end