using Test, DiffEqOperators

#a = rand(10)
#b = rand(10)

a = rand(10)
b = rand(10)

q = BridgeBC(a, length(a), b, 1)
dummy_u = zeros(10)

ab_extended = q*dummy_u
@test a[end] == ab_extended[1]
@test b[1] == ab_extended[end]

# Multi dimensional easy connection test

@inline function _easy_bridge_test(a, b, a_extended, b_extended, dim1, dim2, hilo1, hilo2, dirichlet0)
    if hilo1 == "low"
        if hilo2 == "low"
            @test a_extended.lower == selectdim(b, dim2, 1)
            @test b_extended.lower == selectdim(a, dim1, 1)

            @test a_extended.upper == dirichlet0
            @test b_extended.upper == dirichlet0
        elseif hilo2 == "high"
            @test a_extended.lower == selectdim(b, dim2, size(b,dim2))
            @test b_extended.upper == selectdim(a, dim1, 1)

            @test a_extended.upper == dirichlet0
            @test b_extended.lower == dirichlet0
        end
    elseif hilo1 == "high"
        if hilo2 == "low"
            @test a_extended.upper == selectdim(b, dim2, 1)
            @test b_extended.lower == selectdim(a, dim1, size(a,dim1))

            @test a_extended.lower == dirichlet0
            @test b_extended.upper == dirichlet0
        elseif hilo2 == "high"
            @test a_extended.upper == selectdim(b, dim2, size(b,dim2))
            @test b_extended.upper == selectdim(a, dim1, size(a,dim1))

            @test a_extended.lower == dirichlet0
            @test b_extended.lower == dirichlet0
        end
    end
end


dirichlet0 = zeros(10)
for hilo1 in ["low", "high"], hilo2 in ["low", "high"]
    for dim1 in 1:2, dim2 in 1:2
        a = rand(10,10)
        b = rand(10,10)
        Q1, Q2 = Dirichlet0BC(Float64, size(a)),  Dirichlet0BC(Float64, size(b))
        Qa, Qb = BridgeBC(Q1[dim1], a, dim1, hilo1, dim2, hilo2, b, Q2[dim2])
        a_extended = Qa*a
        b_extended = Qb*b

         _easy_bridge_test(a, b, a_extended, b_extended, dim1, dim2, hilo1, hilo2, dirichlet0)

        a .= a.*2 #Check that the operator still works even after the values in a and b have changed
        b .= b.*2
        a_extended = Qa*a
        b_extended = Qb*b

        _easy_bridge_test(a, b, a_extended, b_extended, dim1, dim2,hilo1, hilo2, dirichlet0)
    end
end
#validate 3d
dirichlet0 = zeros(10,10)
for hilo1 in ["low", "high"], hilo2 in ["low", "high"]
    for dim1 in 1:3, dim2 in 1:3
        a = rand(10,10,10)
        b = rand(10,10,10)
        Q1, Q2 = Dirichlet0BC(Float64, size(a)),  Dirichlet0BC(Float64, size(b))
        Qa, Qb = BridgeBC(Q1[dim1], a, dim1, hilo1, dim2, hilo2, b, Q2[dim2])
        a_extended = Qa*a
        b_extended = Qb*b

         _easy_bridge_test(a, b, a_extended, b_extended, dim1, dim2, hilo1, hilo2, dirichlet0)

        a .= a.*2 #Check that the operator still works even after the values in a and b have changed
        b .= b.*2
        a_extended = Qa*a
        b_extended = Qb*b

        _easy_bridge_test(a, b, a_extended, b_extended, dim1, dim2,hilo1, hilo2, dirichlet0)
    end
end
