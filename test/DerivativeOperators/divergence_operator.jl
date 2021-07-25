using DiffEqOperators, Test

@testset "Divergence operation for a 2D Vector" begin
    
    # For testing the faster implementation for a 2D Vector
    s = x, y = (-5:1.25:5, -5:1.25:5)
    dx = dy = x[2] - x[1]
    
    # Vector u0 = (x^2) ê₁ + (y^2) ê₂
    u0 = zeros(Float64,length(x),length(y),2)
    for j in 1:length(y), i in 1:length(x)
        u0[i,j,1] = x[i]^2
        u0[i,j,2] = y[j]^2
    end
    
    # Analytic Divergence of the given vector given by u_analytic = 2x + 2y
    
    u_analytic = zeros(Float64,size(u0)[1:end-1].-2)
    for j in 1:length(y)-2, i in 1:length(x)-2
        u_analytic[i,j] = 2*x[i+1] + 2*y[j+1]
    end
    
    A = Divergence(4,(dx,dy),size(u0)[1:end-1].-2)
    
    u = A*u0
    
    for I in CartesianIndices(u)
        @test u[I] ≈ u_analytic[I] atol = 1e-3
    end
    
    # check for non-uniform grid
    
    dx = dy = 1.25*ones(10)

    A = Divergence(4,(dx,dy),size(u0)[1:end-1].-2)
    
    u = A*u0
    
    for I in CartesianIndices(u)
        @test u[I] ≈ u_analytic[I] atol=1e-3
    end
end

@testset "Divergence operation for a 3D Vector" begin
    
    # For testing the faster implementation for a 3D Vector
    s = x, y, z = (-5:1.25:5, -5:1.25:5, -5:1.25:5)
    dx = dy = dz = x[2] - x[1]
    
    # Vector u0 = (x^2 + y^2) ê₁ + (y^2 + z^2) ê₂  + (x^2 + z^2) ê₃
    u0 = zeros(Float64,length(x),length(y),length(z),3)
    for k in 1:length(z), j in 1:length(y), i in 1:length(x)
        u0[i,j,k,1] = x[i]^2 + y[j]^2
        u0[i,j,k,2] = y[j]^2 + z[k]^2
        u0[i,j,k,3] = x[i]^2 + z[k]^2
    end
    
    # Analytic Divergence of the given vector given by u_analytic = 2x + 2y + 2z
    
    u_analytic = zeros(Float64,size(u0)[1:end-1].-2)
    for k in 1:length(z)-2, j in 1:length(y)-2, i in 1:length(x)-2
        u_analytic[i,j,k] = 2*x[i+1] + 2*y[j+1] + 2*z[k+1]
    end
    
    A = Divergence(4,(dx,dy,dz),size(u0)[1:end-1].-2)
    
    u = A*u0
    
    for I in CartesianIndices(u)
        @test u[I] ≈ u_analytic[I] atol = 1e-3
    end    
    
    # check for multiplication with constant

    u1 = 2*A*u0
    
    for I in CartesianIndices(u)
        @test u1[I] ≈ 2*u[I] atol=1e-3
    end
    
    # check for non-uniform grid
    
    dx = dy = dz = 1.25*ones(10)

    A = Divergence(4,(dx,dy,dz),size(u0)[1:end-1].-2)
    
    u = A*u0
    
    for I in CartesianIndices(u)
        @test u[I] ≈ u_analytic[I] atol=1e-3
    end
end

@testset "Divergence operation for a 4D Vector" begin
    
    # For Testing the fallback implementation
    s = x, y, z, w = (-5:1.25:5, -5:1.25:5, -5:1.25:5, -5:1.25:5)
    dx = dy = dz = dw = x[2] - x[1]
    
    # Vector u0 = (x^2 + y^2) ê₁ + (y^2 + z^2) ê₂  + (z^2 + w^2) ê₃ + (x^2 + w^2) ê₄
    u0 = zeros(Float64,length(x),length(y),length(z),length(w),4)
    for l in 1:length(w), k in 1:length(z), j in 1:length(y), i in 1:length(x)
        u0[i,j,k,l,1] = x[i]^2 + y[j]^2
        u0[i,j,k,l,2] = y[j]^2 + z[k]^2
        u0[i,j,k,l,3] = z[k]^2 + w[l]^2
        u0[i,j,k,l,4] = x[i]^2 + w[l]^2
    end
    
    # Analytic Divergence of the given vector given by u_analytic = 2x + 2y + 2z + 2w
    
    u_analytic = zeros(Float64,size(u0)[1:end-1].-2)
    for l in 1:length(w)-2, k in 1:length(z)-2, j in 1:length(y)-2, i in 1:length(x)-2
        u_analytic[i,j,k,l] = 2*x[i+1] + 2*y[j+1] + 2*z[k+1] + 2*w[l+1]
    end
    
    A = Divergence(4,(dx,dy,dz,dw),size(u0)[1:end-1].-2)
    
    u = A*u0
    
    for I in CartesianIndices(u)
        @test u[I] ≈ u_analytic[I] atol = 1e-3
    end    
end