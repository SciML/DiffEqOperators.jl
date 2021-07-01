using DiffEqOperators, Test

@testset "Dot, Cross and L2-Norm of vectors" begin
    s = x, y, z = (-5:1.25:5, -5:1.25:5, -5:1.25:5)
    dx = dy = dz = x[2] - x[1]
    
    # Vector u0 = (x^2) ê₁ + (y^2) ê₂  + (z^2) ê₃
    # Vector u1 = (x) ê₁ + (y) ê₂  + (z) ê₃

    u0 = Array{Array{Float64,1},3}(undef,length(x),length(y),length(z))
    for I in CartesianIndices(u0)
        u0[I] = zeros(Float64,3)
        u0[I][1] = x[I[1]]^2
        u0[I][2] = y[I[2]]^2
        u0[I][3] = z[I[3]]^2
    end
    u1 = Array{Array{Float64,1},3}(undef,length(x),length(y),length(z))
    for I in CartesianIndices(u0)
        u1[I] = zeros(Float64,3)
        u1[I][1] = x[I[1]]
        u1[I][2] = y[I[2]]
        u1[I][3] = z[I[3]]
    end
    
    # Analytic dot of u0 & u1 is given by u_analytic = x^3 + y^3 + z^3
    
    u_analytic = Array{Float64}(undef,size(u0))
    for I in CartesianIndices(u_analytic)
        u_analytic[I] = x[I[1]]^3 + y[I[2]]^3 + z[I[3]]^3
    end
    
    u = Array{Float64}(undef,size(u0))
    D = dot_product(u0,u1)
    dot_product!(u,u0,u1)

    for I in CartesianIndices(u)
        @test u[I] ≈ u_analytic[I] atol=1e-3
        @test D[I] ≈ u_analytic[I] atol=1e-3
    end

    # Analytic cross u0xu1 is given by u_analytic2 = yz(y-z)ê₁ + xz(z-x)ê₂ + xy(x-y)ê₃
    
    u_analytic2 = Array{Array{Float64,1},3}(undef,size(u0))
    for I in CartesianIndices(u_analytic2)
        u_analytic2[I] = zeros(Float64,3)
        u_analytic2[I][1] = y[I[2]]*z[I[3]]*(y[I[2]]-z[I[3]])
        u_analytic2[I][2] = x[I[1]]*z[I[3]]*(z[I[3]]-x[I[1]])
        u_analytic2[I][3] = x[I[1]]*y[I[2]]*(x[I[1]]-y[I[2]])
    end

    u2 = Array{Array{Float64,1},3}(undef,size(u0))

    C = cross_product(u0,u1)
    cross_product!(u2,u0,u1)

    for I in CartesianIndices(u2)
        @test u2[I] ≈ u_analytic2[I] atol=1e-3
        @test C[I] ≈ u_analytic2[I] atol=1e-3
    end

    # Analytic L2-norm of the u1 is given by u_analytic3 = (x^2 + y^2 + z^2)^(0.5)

    u_analytic3 = Array{Float64}(undef,size(u0))
    for I in CartesianIndices(u_analytic3)
        u_analytic3[I] = zero(Float64)
        u_analytic3[I] = (x[I[1]]^2 + y[I[2]]^2 + z[I[3]]^2)^0.5
    end

    u3 = Array{Float64}(undef,size(u0))

    N = square_norm(u1);
    square_norm!(u3,u1)
    for I in CartesianIndices(u_analytic3)
        @test N[I] ≈ u_analytic3[I] atol=1e-3
        @test u3[I] ≈ u_analytic3[I] atol=1e-3
    end
end